# SegFiber

SegFiber trains and runs whole-brain fiber segmentation models on the BAIME
project structure.

## Install

```bash
pip install -e .
```

## Configure

Copy and edit the packaged template:

```bash
cp src/seg_fiber/model/config/template.yaml config.yaml
```

Every model module uses a `{name, params}` declaration. Training artifacts keep
the legacy layout below `experiment.output_dir`:

```text
weights/<experiment.name>/
logs/<experiment.name>/
slurm/
```

## Train

```bash
segfiber train --config config.yaml
```

Use `--runtime ddp --devices 0,1` for local DDP. Edit and run
`script/train_slurm.sh` for Slurm.

## Infer

```bash
segfiber infer \
  --config config.yaml \
  --input brain.ims \
  --output segmentation.db \
  --checkpoint FT_C534_model_tiny.pth
```

The two packaged checkpoint names are `FT_C534_model_tiny.pth` and
`universal_tiny.pth`. They use the canonical state-dict format without a
`module.` prefix.

Local and Slurm DDP inference split patches without duplication. Each worker
segments its own patches, and rank 0 writes results to the original NeuroDB
SQLite schema in global patch order. Large patches are internally inferred in
batches of eight tiles by default; adjust `inference.params.tile_batch_size`
to match the available GPU memory.

To split a brain into Z slabs and write one database per slab, use the
`brain_z_slabs` dataset and inferencer:

```bash
segfiber infer \
  --config src/seg_fiber/model/config/brain_z_slabs.yaml \
  --input brain.ims \
  --output out/slabs
```

Merge completed Z-slab databases without connecting endpoints across slab
boundaries:

```bash
segfiber merge \
  --input-dir out/slabs \
  --output out/segmentation.db \
  --reset
```
