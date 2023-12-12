## Getting Started with Compositor

This document provides a brief intro of the usage of Compositor.

Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.


### Training & Evaluation in Command Line

We provide script `train_net.py`, that is made to train all the configs provided in Compositor.

> Train Compositor w/ MaskFormer

To train a Compositor w/ MaskFormer model with "train_net.py", first setup the corresponding datasets following
[DATASETS.md](DATASETS.md), and change directory to Compositor_Mask2Former,
then run:
```
python train_net.py --num-gpus 8 \
  --config-file configs/partimagenet/semantic-segmentation/compositor_R50_bs16_90k.yaml 
```

The configs are made for 8-GPU training.
Since we use ADAMW optimizer, it is not clear how to scale learning rate with batch size.
To train on 1 GPU, you need to figure out learning rate and batch size by yourself:
```
python train_net.py \
  --config-file configs/partimagenet/semantic-segmentation/compositor_R50_bs16_90k.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH SET_TO_SOME_REASONABLE_VALUE SOLVER.BASE_LR SET_TO_SOME_REASONABLE_VALUE
```

> Train Compositor w/ kMaX-DeepLab

To train a Compositor w/ kMaX-DeepLab model with "train_net.py", change directory to Compositor_kMaX-DeepLab,
then run
```
python train_net.py --num-gpus 8 \
  --config-file configs/partimagenet/semantic_segmentation/compositor_r50.yaml \
```

> Evaluate Compositor w/ Mask2Former or kMaX-DeepLab

To evaluate a model's performance, use
```
python train_net.py \
  --config-file /path/to/config \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file \
```

For more options, see `python train_net.py -h`.