# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

__all__ = ["CompositorSemanticDatasetMapper"]

class CompositorSemanticDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_part_label,
        ignore_object_label,
        size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_part_label = ignore_part_label
        self.ignore_object_label = ignore_object_label
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_part_label = meta.ignore_part_label
        ignore_object_label = meta.ignore_object_label

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_part_label": ignore_part_label,
            "ignore_object_label": ignore_object_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "CompositorSemanticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_part_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_part_gt = utils.read_image(dataset_dict.pop("sem_seg_part_file_name")).astype("double")
        else:
            sem_seg_part_gt = None

        if "sem_seg_object_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_object_gt = utils.read_image(dataset_dict.pop("sem_seg_object_file_name")).astype("double")
        else:
            sem_seg_object_gt = None

        if sem_seg_part_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_part_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        if sem_seg_object_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_object_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        sem_seg_part_gt = transforms.apply_segmentation(sem_seg_part_gt)
        sem_seg_object_gt = transforms.apply_segmentation(sem_seg_object_gt)

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_part_gt is not None:
            sem_seg_part_gt = torch.as_tensor(sem_seg_part_gt.astype("long"))
        if sem_seg_object_gt is not None:
            sem_seg_object_gt = torch.as_tensor(sem_seg_object_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_part_gt is not None:
                sem_seg_part_gt = F.pad(sem_seg_part_gt, padding_size, value=self.ignore_part_label).contiguous()
            if sem_seg_object_gt is not None:
                sem_seg_object_gt = F.pad(sem_seg_object_gt, padding_size, value=self.ignore_object_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        sem_seg_part_gt = sem_seg_part_gt.numpy()
        sem_seg_object_gt = sem_seg_object_gt.numpy()
        part_instances = Instances(image_shape)
        object_instances = Instances(image_shape)
        part_classes = np.unique(sem_seg_part_gt)
        object_classes = np.unique(sem_seg_object_gt)

        # remove ignored region
        part_classes = part_classes[part_classes != self.ignore_part_label]
        object_classes = object_classes[object_classes != self.ignore_object_label]
        part_instances.gt_classes = torch.tensor(part_classes, dtype=torch.int64)
        object_instances.gt_classes = torch.tensor(object_classes, dtype=torch.int64)

        part_masks = []
        object_masks = []
        for class_id in part_classes:
            part_masks.append(sem_seg_part_gt == class_id)
        for class_id in object_classes:
            object_masks.append(sem_seg_object_gt == class_id)

        if len(part_masks) == 0:
            # Some image does not have annotation (all ignored)
            part_instances.gt_masks = torch.zeros((0, sem_seg_part_gt.shape[-2], sem_seg_part_gt.shape[-1]))
        else:
            part_masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in part_masks])
            )
            part_instances.gt_masks = part_masks.tensor

        if len(object_masks) == 0:
            # Some image does not have annotation (all ignored)
            object_instances.gt_masks = torch.zeros((0, sem_seg_object_gt.shape[-2], sem_seg_object_gt.shape[-1]))
        else:
            object_masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in object_masks])
            )
            object_instances.gt_masks = object_masks.tensor

        dataset_dict["part_instances"] = part_instances
        dataset_dict["object_instances"] = object_instances

        return dataset_dict
