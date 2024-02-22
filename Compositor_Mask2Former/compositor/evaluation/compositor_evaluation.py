# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
from typing import Optional, Union
import pycocotools.mask as mask_util
import torch
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2.evaluation import DatasetEvaluator


def load_image_into_numpy_array(
    filename: str,
    copy: bool = False,
    dtype: Optional[Union[np.dtype, str]] = None,
) -> np.ndarray:
    with PathManager.open(filename, "rb") as f:
        array = np.array(Image.open(f), copy=copy, dtype=dtype)
    return array


class CompositorEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        sem_seg_loading_fn=load_image_into_numpy_array,
        num_part_classes=None,
        num_object_classes=None,
        ignore_part_label=None,
        ignore_object_label=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            sem_seg_loading_fn: function to read sem seg file and load into numpy array.
                Default provided, but projects can customize.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)
        if num_part_classes is not None:
            self._logger.warn(
                "SemSegEvaluator(num_part_classes) is deprecated! It should be obtained from metadata."
            )
        if num_object_classes is not None:
            self._logger.warn(
                "SemSegEvaluator(num_object_classes) is deprecated! It should be obtained from metadata."
            )
        if ignore_part_label is not None:
            self._logger.warn(
                "SemSegEvaluator(ignore_part_label) is deprecated! It should be obtained from metadata."
            )
        if ignore_object_label is not None:
            self._logger.warn(
                "SemSegEvaluator(ignore_object_label) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self.input_file_to_part_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_part_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }
        self.input_file_to_object_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_object_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)
        self._contiguous_id_to_dataset_id = None
        self._part_class_names = meta.part_classes
        self._object_class_names = meta.object_classes
        self._num_part_classes = len(meta.part_classes)
        self._num_object_classes = len(meta.object_classes)
        self._ignore_part_label = meta.ignore_part_label
        self._ignore_object_label = meta.ignore_object_label

        self.sem_seg_loading_fn = sem_seg_loading_fn

    def reset(self):
        self._part_conf_matrix = np.zeros((self._num_part_classes + 1, self._num_part_classes + 1), dtype=np.int64)
        self._object_conf_matrix = np.zeros((self._num_object_classes + 1, self._num_object_classes + 1), dtype=np.int64)
        self._part_predictions = []
        self._object_predictions = []
        self._cc = np.zeros(1)
        self._fc = np.zeros(1)

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            part_output = output["part_sem_seg"].argmax(dim=0).to(self._cpu_device)
            object_output = output["object_sem_seg"].argmax(dim=0).to(self._cpu_device)
            part_pred = np.array(part_output, dtype=int)
            object_pred = np.array(object_output, dtype=int)
            part_gt_filename = self.input_file_to_part_gt_file[input["file_name"]]
            object_gt_filename = self.input_file_to_object_gt_file[input["file_name"]]
            part_gt = self.sem_seg_loading_fn(part_gt_filename, dtype=int)
            object_gt = self.sem_seg_loading_fn(object_gt_filename, dtype=int)

            part_gt = part_gt.copy()
            object_gt = object_gt.copy()

            part_gt[part_gt == self._ignore_part_label] = self._num_part_classes
            object_gt[object_gt == self._ignore_object_label] = self._num_object_classes 

            self._part_conf_matrix += np.bincount(
                (self._num_part_classes + 1) * part_pred.reshape(-1) + part_gt.reshape(-1),
                minlength=self._part_conf_matrix.size,
            ).reshape(self._part_conf_matrix.shape)
            self._object_conf_matrix += np.bincount(
                (self._num_object_classes + 1) * object_pred.reshape(-1) + object_gt.reshape(-1),
                minlength=self._object_conf_matrix.size,
            ).reshape(self._object_conf_matrix.shape)                                                                                                                                                                                                                                                                                                                                                                           

            self._part_predictions.extend(self.encode_json_sem_seg(part_pred, input["file_name"]))
            self._object_predictions.extend(self.encode_json_sem_seg(object_pred, input["file_name"]))

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            part_conf_matrix_list = all_gather(self._part_conf_matrix)
            self._part_predictions = all_gather(self._part_predictions)
            self._part_predictions = list(itertools.chain(*self._part_predictions))
            object_conf_matrix_list = all_gather(self._object_conf_matrix)
            self._object_predictions = all_gather(self._object_predictions)
            self._object_predictions = list(itertools.chain(*self._object_predictions))
            if not is_main_process():
                return

            self._part_conf_matrix = np.zeros_like(self._part_conf_matrix)
            for part_conf_matrix in part_conf_matrix_list:
                self._part_conf_matrix += part_conf_matrix
            self._object_conf_matrix = np.zeros_like(self._object_conf_matrix)
            for object_conf_matrix in object_conf_matrix_list:
                self._object_conf_matrix += object_conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            part_file_path = os.path.join(self._output_dir, "sem_seg_part_predictions.json")
            object_file_path = os.path.join(self._output_dir, "sem_seg_object_predictions.json")
            with PathManager.open(part_file_path, "w") as f:
                f.write(json.dumps(self._part_predictions))
            with PathManager.open(object_file_path, "w") as f:
                f.write(json.dumps(self._object_predictions))

        part_acc = np.full(self._num_part_classes, np.nan, dtype=float)
        part_iou = np.full(self._num_part_classes, np.nan, dtype=float)
        part_tp = self._part_conf_matrix.diagonal()[:-1].astype(float)
        part_pos_gt = np.sum(self._part_conf_matrix[:-1, :-1], axis=0).astype(float)
        part_class_weights = part_pos_gt / np.sum(part_pos_gt)
        part_pos_pred = np.sum(self._part_conf_matrix[:-1, :-1], axis=1).astype(float)
        part_acc_valid = part_pos_gt > 0
        part_acc[part_acc_valid] = part_tp[part_acc_valid] / part_pos_gt[part_acc_valid]
        part_iou_valid = (part_pos_gt + part_pos_pred) > 0
        part_union = part_pos_gt + part_pos_pred - part_tp
        part_iou[part_iou_valid] = part_tp[part_iou_valid] / part_union[part_iou_valid]
        part_macc = np.sum(part_acc[part_acc_valid]) / np.sum(part_acc_valid)
        part_miou = np.sum(part_iou[part_iou_valid]) / np.sum(part_iou_valid)
        part_fiou = np.sum(part_iou[part_iou_valid] * part_class_weights[part_iou_valid])
        part_pacc = np.sum(part_tp) / np.sum(part_pos_gt)

        object_acc = np.full(self._num_object_classes, np.nan, dtype=float)
        object_iou = np.full(self._num_object_classes, np.nan, dtype=float)
        object_tp = self._object_conf_matrix.diagonal()[:-1].astype(float)
        object_pos_gt = np.sum(self._object_conf_matrix[:-1, :-1], axis=0).astype(float)
        object_class_weights = object_pos_gt / np.sum(object_pos_gt)
        object_pos_pred = np.sum(self._object_conf_matrix[:-1, :-1], axis=1).astype(float)
        object_acc_valid = object_pos_gt > 0
        object_acc[object_acc_valid] = object_tp[object_acc_valid] / object_pos_gt[object_acc_valid]
        object_iou_valid = (object_pos_gt + object_pos_pred) > 0
        object_union = object_pos_gt + object_pos_pred - object_tp
        object_iou[object_iou_valid] = object_tp[object_iou_valid] / object_union[object_iou_valid]
        object_macc = np.sum(object_acc[object_acc_valid]) / np.sum(object_acc_valid)
        object_miou = np.sum(object_iou[object_iou_valid]) / np.sum(object_iou_valid)
        object_fiou = np.sum(object_iou[object_iou_valid] * object_class_weights[object_iou_valid])
        object_pacc = np.sum(object_tp) / np.sum(object_pos_gt)

        res = {}
        res["part_mIoU"] = 100 * part_miou
        res["part_fwIoU"] = 100 * part_fiou
        for i, name in enumerate(self._part_class_names):
            res["IoU-{}".format(name)] = 100 * part_iou[i]
        res["part_mACC"] = 100 * part_macc
        res["part_pACC"] = 100 * part_pacc
        for i, name in enumerate(self._part_class_names):
            res["ACC-{}".format(name)] = 100 * part_acc[i]
        res["object_mIoU"] = 100 * object_miou
        res["object_fwIoU"] = 100 * object_fiou
        for i, name in enumerate(self._object_class_names):
            res["object_IoU-{}".format(name)] = 100 * object_iou[i]
        res["object_mACC"] = 100 * object_macc
        res["object_pACC"] = 100 * object_pacc
        for i, name in enumerate(self._object_class_names):
            res["object_ACC-{}".format(name)] = 100 * object_acc[i]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results

    def encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert (
                    label in self._contiguous_id_to_dataset_id
                ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(int)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F",dtype='uint8'))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append(
                {"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle}
            )
        return json_list