# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_compositor_config

# dataset loading
from .data.dataset_mappers.compositor_semantic_dataset_mapper import CompositorSemanticDatasetMapper

# models
from .compositor_model import Compositor
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.compositor_evaluation import CompositorEvaluator
