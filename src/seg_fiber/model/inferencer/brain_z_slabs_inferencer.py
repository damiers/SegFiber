import math
from pathlib import Path

import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from ..core.registry import INFERENCER_REGISTRY
from ..dataset import get_dataset
from ..dataset.brain_z_slabs_dataset import collate_brain_z_slabs
from ..utils.neurodb_sqlite import NeurodbSQLite
from .seger import Seger


class ShardSampler(Sampler):
    def __init__(self, dataset_size, rank, world_size):
        self.indices = range(rank, dataset_size, world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


@INFERENCER_REGISTRY.register("brain_z_slabs")
class BrainZSlabInferencer:
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
            collate_fn=collate_brain_z_slabs,
        )
        seger = Seger(
            self.config,
            checkpoint=self.params["checkpoint"],
            background_threshold=self.params["background_threshold"],
            tile_batch_size=self.params["tile_batch_size"],
            device=context.device,
        )

        output_paths = self._output_paths(dataset.slab_count)
        databases = {}
        versions = {}
        if context.is_main:
            output_paths[0].parent.mkdir(parents=True, exist_ok=True)
            for slab_number, output_path in enumerate(output_paths, start=1):
                if reset and output_path.exists():
                    output_path.unlink()
                database = NeurodbSQLite(output_path)
                databases[slab_number] = database
                _, versions[slab_number] = database.get_max_sid_version()

        transfer_group = None
        if context.world_size > 1:
            transfer_group = dist.new_group(backend="gloo")
            dist.barrier(group=transfer_group)

        iterator = iter(loader)
        rounds = math.ceil(len(dataset) / context.world_size)
        progress = (
            tqdm(total=len(dataset), desc="GlobalProgressBar")
            if context.is_main
            else None
        )
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
                payload = (batch["index"], batch["slab_number"], segments)

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
                results = sorted(
                    (item for item in gathered if item is not None),
                    key=lambda item: item[0],
                )
                for _, slab_number, segments in results:
                    databases[slab_number].segs2db(
                        segments,
                        version=versions[slab_number],
                    )
                progress.update(len(results))

        if progress is not None:
            progress.close()
        if transfer_group is not None:
            dist.destroy_process_group(transfer_group)
        return output_paths if context.is_main else None

    def _output_paths(self, slab_count):
        output_dir = Path(self.params["output_path"]).expanduser()
        prefix = self.params["output_prefix"]
        width = len(str(slab_count))
        keep_branch = "_keepBranch" if self.params["keep_branch"] else ""
        return [
            output_dir / f"{prefix}z{number:0{width}d}{keep_branch}.db"
            for number in range(1, slab_count + 1)
        ]
