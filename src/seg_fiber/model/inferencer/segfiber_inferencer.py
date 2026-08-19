import math
from pathlib import Path

import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from ..core.registry import INFERENCER_REGISTRY
from ..dataset import get_dataset
from ..dataset.whole_brain_dataset import collate_whole_brain
from ..utils.neurodb_sqlite import NeurodbSQLite
from .seger import Seger


class ShardSampler(Sampler):
    def __init__(self, dataset_size, rank, world_size):
        self.indices = range(rank, dataset_size, world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


@INFERENCER_REGISTRY.register("segfiber")
class SegFiberInferencer:
    def __init__(self, config, params):
        self.config = config
        self.params = params

    def run(self, context, reset=False):
        dataset = get_dataset(self.config, "infer")
        sampler = ShardSampler(len(dataset), context.rank, context.world_size)
        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=self.params["workers"],
            pin_memory=context.device.type == "cuda",
            collate_fn=collate_whole_brain,
        )
        seger = Seger(
            self.config,
            checkpoint=self.params["checkpoint"],
            background_threshold=self.params["background_threshold"],
            tile_batch_size=self.params["tile_batch_size"],
            device=context.device,
        )

        output_path = self._output_path()
        database = None
        version = None
        if context.is_main:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if reset and output_path.exists():
                output_path.unlink()
            database = NeurodbSQLite(output_path)
            _, version = database.get_max_sid_version()

        transfer_group = None
        if context.world_size > 1:
            transfer_group = dist.new_group(backend="gloo")
            dist.barrier(group=transfer_group)

        iterator = iter(loader)
        rounds = math.ceil(len(dataset) / context.world_size)
        progress = tqdm(total=len(dataset), desc="GlobalProgressBar") if context.is_main else None
        for _ in range(rounds):
            batch = next(iterator, None)
            payload = None
            if batch is not None:
                segments = seger.process(
                    batch["image"],
                    batch["offset"],
                    batch["rebatch"],
                    keep_branch=self.params["keep_branch"],
                )
                payload = (batch["index"], segments)

            if transfer_group is None:
                gathered = [payload]
            else:
                gathered = [None] * context.world_size if context.is_main else None
                dist.gather_object(
                    payload,
                    object_gather_list=gathered,
                    dst=0,
                    group=transfer_group,
                )

            if context.is_main:
                results = sorted(item for item in gathered if item is not None)
                for _, segments in results:
                    database.segs2db(segments, version=version)
                progress.update(len(results))

        if progress is not None:
            progress.close()
        if transfer_group is not None:
            dist.destroy_process_group(transfer_group)
        return output_path if context.is_main else None

    def _output_path(self):
        image_path = self.config["dataset"]["infer"]["params"]["image_path"]
        output_path = Path(self.params["output_path"]).expanduser()
        if output_path.suffix:
            if output_path.suffix != ".db":
                output_path = output_path.with_suffix(".db")
        else:
            output_path /= f"segerOut_{Path(image_path).stem}.db"
        if self.params["keep_branch"] and not output_path.stem.endswith("_keepBranch"):
            output_path = output_path.with_name(
                f"{output_path.stem}_keepBranch{output_path.suffix}"
            )
        return output_path
