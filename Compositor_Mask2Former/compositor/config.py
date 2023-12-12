# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_compositor_config(cfg):
    """
    Add config for COMPOSITOR.
    """

    # SEM_SEG_HEADS NUM_CLASSES
    cfg.MODEL.SEM_SEG_HEAD.NUM_PART_CLASSES = 41
    cfg.MODEL.SEM_SEG_HEAD.NUM_OBJECT_CLASSES = 159

    # compositor model config
    cfg.MODEL.COMPOSITOR = CN()

    # loss
    cfg.MODEL.COMPOSITOR.DEEP_SUPERVISION = True
    cfg.MODEL.COMPOSITOR.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.COMPOSITOR.PART_CLASS_WEIGHT = 2.0
    cfg.MODEL.COMPOSITOR.PART_DICE_WEIGHT = 5.0
    cfg.MODEL.COMPOSITOR.PART_MASK_WEIGHT = 5.0
    cfg.MODEL.COMPOSITOR.OBJECT_CLASS_WEIGHT = 2.0
    cfg.MODEL.COMPOSITOR.OBJECT_DICE_WEIGHT = 5.0
    cfg.MODEL.COMPOSITOR.OBJECT_MASK_WEIGHT = 5.0

    # transformer config
    cfg.MODEL.COMPOSITOR.NHEADS = 8
    cfg.MODEL.COMPOSITOR.DROPOUT = 0.1
    cfg.MODEL.COMPOSITOR.DIM_FEEDFORWARD = 2048
    cfg.MODEL.COMPOSITOR.ENC_LAYERS = 0
    cfg.MODEL.COMPOSITOR.DEC_LAYERS = 6
    cfg.MODEL.COMPOSITOR.PRE_NORM = False

    cfg.MODEL.COMPOSITOR.HIDDEN_DIM = 256
    cfg.MODEL.COMPOSITOR.NUM_PART_QUERIES = 100
    cfg.MODEL.COMPOSITOR.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.COMPOSITOR.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.COMPOSITOR.ENFORCE_INPUT_PROJ = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.COMPOSITOR.SIZE_DIVISIBILITY = 32

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.COMPOSITOR.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.COMPOSITOR.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.COMPOSITOR.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.COMPOSITOR.IMPORTANCE_SAMPLE_RATIO = 0.75
