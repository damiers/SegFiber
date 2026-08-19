from pathlib import Path
import math

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ...ckpt import load_model_state
from ..arch import get_model
from ..config import to_plain_data
from ..core.registry import TRAINER_REGISTRY
from ..dataset import get_dataset
from ..loss import get_loss
from ..optimizer import get_optimizer
from ..scheduler import get_scheduler


@TRAINER_REGISTRY.register("segfiber")
class SegFiberTrainer:
    def run(self, config, context):
        self.config = config
        self.params = config["trainer"]["params"]
        self.context = context
        self.writer = SummaryWriter(context.paths.tensorboard) if context.is_main else None
        checkpoint = self._load_checkpoint()
        self._init_modules(checkpoint)
        self.start_epoch = checkpoint["epoch"] if checkpoint else 0
        self.best_val_loss = checkpoint.get("best_val_loss", math.inf) if checkpoint else math.inf

        for epoch in range(self.start_epoch, self.params["epochs"]):
            train_loss = self._train_epoch(epoch)
            val_loss = self._validate_epoch(epoch)
            if context.is_main:
                self.writer.add_scalar("loss/TrainEpochAVGLoss", train_loss, epoch + 1)
                if val_loss is not None:
                    self.writer.add_scalar("loss/ValEpochAVGLoss", val_loss, epoch + 1)
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(epoch, "best_val_model.pth")
                if (epoch + 1) % self.params["save_every"] == 0:
                    self._save_checkpoint(epoch, f"Epoch_{epoch + 1:03d}.pth")

        if self.writer is not None:
            self.writer.close()

    def _init_modules(self, checkpoint):
        train_dataset = get_dataset(self.config, "train")
        sampler = self.context.make_sampler(
            train_dataset,
            shuffle=self.params["shuffle"],
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.params["batch_size"],
            shuffle=self.params["shuffle"] if sampler is None else False,
            sampler=sampler,
            num_workers=self.params["workers"],
            pin_memory=self.context.device.type == "cuda",
            drop_last=True,
        )

        self.val_loader = None
        if self.context.is_main and self.config["dataset"].get("val"):
            self.val_loader = DataLoader(
                get_dataset(self.config, "val"),
                batch_size=self.params["batch_size"],
            )

        model = get_model(self.config)
        if checkpoint:
            model.load_state_dict(checkpoint["model"], strict=True)
        elif self.params.get("pretrain_checkpoint"):
            state = load_model_state(self.params["pretrain_checkpoint"])
            model.load_state_dict(state, strict=True)
        model = model.to(self.context.device)
        if self.context.world_size > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.raw_model = model
        self.model = self.context.wrap_model(model)
        self.loss = get_loss(self.config).to(self.context.device)
        self.optimizer = get_optimizer(self.config, self.model)
        self.scheduler = get_scheduler(
            self.config,
            self.optimizer,
            len(self.train_loader),
            self.context.world_size,
        )
        amp_enabled = self.params["fp16"] and self.context.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        if checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "fp16_scaler" in checkpoint:
                self.scaler.load_state_dict(checkpoint["fp16_scaler"])

    def _train_epoch(self, epoch):
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)
        self.model.train()
        total_loss = 0.0
        progress = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch [{epoch + 1}/{self.params['epochs']}]",
            disable=not self.context.is_main,
        )
        for batch_index, (inputs, targets) in progress:
            iteration = epoch * len(self.train_loader) + batch_index
            learning_rate = self.scheduler.step(iteration)
            inputs = inputs.to(self.context.device, non_blocking=True)
            targets = targets.to(self.context.device, non_blocking=True)
            amp_enabled = self.params["fp16"] and self.context.device.type == "cuda"
            with torch.amp.autocast(self.context.device.type, enabled=amp_enabled):
                loss = self.loss(self.model(inputs), targets)
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"Loss is not finite: {loss.item()}")
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            if self.context.is_main:
                self.writer.add_scalar("Loss/total", loss.item(), iteration)
                self.writer.add_scalar("Schedules/Learning Rate", learning_rate, iteration)
                progress.set_postfix(loss=f"{loss.item():.6f}")
        return self.context.reduce_mean(total_loss / len(self.train_loader))

    @torch.no_grad()
    def _validate_epoch(self, epoch):
        value = None
        if self.context.is_main and self.val_loader is not None:
            self.raw_model.eval()
            total_loss = 0.0
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.context.device, non_blocking=True)
                targets = targets.to(self.context.device, non_blocking=True)
                total_loss += self.loss(self.raw_model(inputs), targets).item()
            value = total_loss / len(self.val_loader)
        if self.context.world_size > 1:
            dist.barrier()
        return value

    def _load_checkpoint(self):
        paths = sorted(Path(self.context.paths.checkpoints).glob("Epoch_*.pth"))
        if not paths:
            print("No checkpoint found. Training from scratch.")
            return None
        path = paths[-1]
        print(f"Loading checkpoint from {path}")
        return torch.load(path, map_location="cpu", weights_only=True)

    def _save_checkpoint(self, epoch, filename):
        model = self.model.module if hasattr(self.model, "module") else self.model
        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "fp16_scaler": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": to_plain_data(self.config),
        }
        torch.save(checkpoint, self.context.paths.checkpoints / filename)
