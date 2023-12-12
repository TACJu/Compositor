# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_compositor_config(cfg):
    """
    Add config for COMPOSITOR.
    """
    # NOTE: configs from original maskformer

    # SEM_SEG_HEADS NUM_CLASSES
    cfg.MODEL.SEM_SEG_HEAD.NUM_PART_CLASSES = 41
    cfg.MODEL.SEM_SEG_HEAD.NUM_OBJECT_CLASSES = 159

    # kMaX-DeepLab model config
    cfg.MODEL.COMPOSITOR = CN()

    # whether to share matching results
    cfg.MODEL.COMPOSITOR.SHARE_FINAL_MATCHING = True

    # channel-last format
    cfg.MODEL.COMPOSITOR.CHANNEL_LAST_FORMAT = False

    # loss
    cfg.MODEL.COMPOSITOR.DEEP_SUPERVISION = True
    cfg.MODEL.COMPOSITOR.NO_OBJECT_WEIGHT = 1e-5
    cfg.MODEL.COMPOSITOR.PART_CLASS_WEIGHT = 3.0
    cfg.MODEL.COMPOSITOR.PART_DICE_WEIGHT = 3.0
    cfg.MODEL.COMPOSITOR.PART_MASK_WEIGHT = 0.3
    cfg.MODEL.COMPOSITOR.PART_INSDIS_WEIGHT = 1.0
    cfg.MODEL.COMPOSITOR.OBJECT_CLASS_WEIGHT = 3.0
    cfg.MODEL.COMPOSITOR.OBJECT_DICE_WEIGHT = 3.0
    cfg.MODEL.COMPOSITOR.OBJECT_MASK_WEIGHT = 0.3
    cfg.MODEL.COMPOSITOR.OBJECT_INSDIS_WEIGHT = 1.0

    cfg.MODEL.COMPOSITOR.PIXEL_INSDIS_TEMPERATURE = 1.5
    cfg.MODEL.COMPOSITOR.PIXEL_INSDIS_SAMPLE_K = 4096
    cfg.MODEL.COMPOSITOR.MASKING_VOID_PIXEL = True

    # transformer decoder config
    cfg.MODEL.COMPOSITOR.TRANS_DEC = CN()
    cfg.MODEL.COMPOSITOR.TRANS_DEC.NAME = "CompositorTransformerDecoder"
    cfg.MODEL.COMPOSITOR.TRANS_DEC.DEC_LAYERS = [2, 2, 2]
    cfg.MODEL.COMPOSITOR.TRANS_DEC.NUM_PART_QUERIES = 50
    cfg.MODEL.COMPOSITOR.TRANS_DEC.NUM_OBJECT_QUERIES = 20
    cfg.MODEL.COMPOSITOR.TRANS_DEC.IN_CHANNELS = [2048, 1024, 512]
    cfg.MODEL.COMPOSITOR.TRANS_DEC.DROP_PATH_PROB = 0.0

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.COMPOSITOR.SIZE_DIVISIBILITY = -1

