# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher


@META_ARCH_REGISTRY.register()
class Compositor(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_part_queries: int,
        num_object_queries: int,
        metadata,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_part_queries: int, number of part queries
            num_object_queries: int, number of object queries
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_part_queries = num_part_queries
        self.num_object_queries = num_object_queries
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.COMPOSITOR.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.COMPOSITOR.NO_OBJECT_WEIGHT

        # loss weights
        part_class_weight = cfg.MODEL.COMPOSITOR.PART_CLASS_WEIGHT
        part_dice_weight = cfg.MODEL.COMPOSITOR.PART_DICE_WEIGHT
        part_mask_weight = cfg.MODEL.COMPOSITOR.PART_MASK_WEIGHT

        object_class_weight = cfg.MODEL.COMPOSITOR.OBJECT_CLASS_WEIGHT
        object_dice_weight = cfg.MODEL.COMPOSITOR.OBJECT_DICE_WEIGHT
        object_mask_weight = cfg.MODEL.COMPOSITOR.OBJECT_MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_part_class=part_class_weight,
            cost_part_mask=part_mask_weight,
            cost_part_dice=part_dice_weight,
            cost_object_class=object_class_weight,
            cost_object_mask=object_mask_weight,
            cost_object_dice=object_dice_weight,
            num_points=cfg.MODEL.COMPOSITOR.TRAIN_NUM_POINTS,
        )

        weight_dict = {"part_loss_ce": part_class_weight, "part_loss_mask": part_mask_weight, "part_loss_dice": part_dice_weight,
                       "object_loss_ce": object_class_weight, "object_loss_mask": object_mask_weight, "object_loss_dice": object_dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.COMPOSITOR.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["part_labels", "part_masks", "object_masks", "object_labels"]

        criterion = SetCriterion(
            sem_seg_head.num_part_classes,
            sem_seg_head.num_object_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.COMPOSITOR.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.COMPOSITOR.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.COMPOSITOR.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_part_queries": cfg.MODEL.COMPOSITOR.NUM_PART_QUERIES,
            "num_object_queries": cfg.MODEL.COMPOSITOR.NUM_OBJECT_QUERIES,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.COMPOSITOR.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        if self.training:
            # mask classification target
            if "part_instances" in batched_inputs[0] and "object_instances" in batched_inputs[0]:
                gt_part_instances = [x["part_instances"].to(self.device) for x in batched_inputs]
                gt_object_instances = [x["object_instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_part_instances, gt_object_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            part_mask_cls_results = outputs["part_pred_logits"]
            part_mask_pred_results = outputs["part_pred_masks"]
            object_mask_cls_results = outputs["object_pred_logits"]
            object_mask_pred_results = outputs["object_pred_masks"]

            # upsample masks
            part_mask_pred_results = F.interpolate(
                part_mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            object_mask_pred_results = F.interpolate(
                object_mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for part_mask_cls_result, part_mask_pred_result, object_mask_cls_result, object_mask_pred_result, input_per_image, image_size in zip(
                part_mask_cls_results, part_mask_pred_results, object_mask_cls_results, object_mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                part_r = retry_if_cuda_oom(self.semantic_inference)(part_mask_cls_result, part_mask_pred_result)
                object_r = retry_if_cuda_oom(self.semantic_inference)(object_mask_cls_result, object_mask_pred_result)
                part_r = retry_if_cuda_oom(sem_seg_postprocess)(part_r, image_size, height, width)
                object_r = retry_if_cuda_oom(sem_seg_postprocess)(object_r, image_size, height, width)
                processed_results[-1]["part_sem_seg"] = part_r
                processed_results[-1]["object_sem_seg"] = object_r

            return processed_results

    def prepare_targets(self, part_targets, object_targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for part_targets_per_image, object_targets_per_image in zip(part_targets, object_targets):
            # pad gt
            gt_part_masks = part_targets_per_image.gt_masks
            padded_part_masks = torch.zeros((gt_part_masks.shape[0], h_pad, w_pad), dtype=gt_part_masks.dtype, device=gt_part_masks.device)
            padded_part_masks[:, : gt_part_masks.shape[1], : gt_part_masks.shape[2]] = gt_part_masks

            gt_object_masks = object_targets_per_image.gt_masks
            padded_object_masks = torch.zeros((gt_object_masks.shape[0], h_pad, w_pad), dtype=gt_object_masks.dtype, device=gt_object_masks.device)
            padded_object_masks[:, : gt_object_masks.shape[1], : gt_object_masks.shape[2]] = gt_object_masks

            new_targets.append(
                {
                    "part_labels": part_targets_per_image.gt_classes,
                    "part_masks": padded_part_masks,
                    "object_labels": object_targets_per_image.gt_classes,
                    "object_masks": padded_object_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg
