"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.

Reference: https://github.com/google-research/deeplab2/blob/main/model/transformer_decoder/kmax.py
"""

from typing import List
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_tf_ as trunc_normal_

from detectron2.config import configurable
from detectron2.utils.registry import Registry

from kmax_deeplab.modeling.pixel_decoder.kmax_pixel_decoder import get_norm, ConvBN
from kmax_deeplab.modeling.transformer_decoder.kmax_transformer_decoder import (
    add_bias_towards_void,
    AttentionOperation,
)


TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module.
"""
def build_transformer_decoder(cfg, input_shape_from_backbone):
    """
    Build a instance embedding branch from `cfg.MODEL.KMAX_DEEPLAB.TRANS_DEC.NAME`.
    """
    name = cfg.MODEL.COMPOSITOR.TRANS_DEC.NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, input_shape_from_backbone)


class CompositorPredictor(nn.Module):
    def __init__(self, in_channel_pixel, num_part_classes=41+1, num_object_classes=159+1):
        super().__init__()
        self._pixel_space_head_conv0bnact = ConvBN(in_channel_pixel, in_channel_pixel, kernel_size=5, groups=in_channel_pixel, padding=2, bias=False,
                                                   norm='syncbn', act='gelu', conv_init='xavier_uniform')
        self._pixel_space_head_conv1bnact = ConvBN(in_channel_pixel, 256, kernel_size=1, bias=False, norm='syncbn', act='gelu')
        self._pixel_space_head_last_convbn = ConvBN(256, 128, kernel_size=1, bias=True, norm='syncbn', act=None)
        trunc_normal_(self._pixel_space_head_last_convbn.conv.weight, std=0.01)

        self._transformer_part_mask_head = ConvBN(256, 128, kernel_size=1, bias=False, norm='syncbn', act=None, conv_type='1d')
        self._transformer_part_class_head = ConvBN(256, num_part_classes, kernel_size=1, norm=None, act=None, conv_type='1d')
        trunc_normal_(self._transformer_part_class_head.conv.weight, std=0.01)

        self._transformer_object_mask_head = ConvBN(256, 128, kernel_size=1, bias=False, norm='syncbn', act=None, conv_type='1d')
        self._transformer_object_class_head = ConvBN(256, num_object_classes, kernel_size=1, norm=None, act=None, conv_type='1d')
        trunc_normal_(self._transformer_object_class_head.conv.weight, std=0.01)

        self._pixel_space_mask_batch_norm = get_norm('syncbn', channels=1)
        nn.init.constant_(self._pixel_space_mask_batch_norm.weight, 0.1)


    def forward(self, mask_embeddings, class_embeddings, pixel_feature, pred_part=True):
        # mask_embeddings/class_embeddings: B x C x N
        # pixel feature: B x C x H x W
        pixel_space_feature = self._pixel_space_head_conv0bnact(pixel_feature)
        pixel_space_feature = self._pixel_space_head_conv1bnact(pixel_space_feature)
        pixel_space_feature = self._pixel_space_head_last_convbn(pixel_space_feature)
        pixel_space_normalized_feature = F.normalize(pixel_space_feature, p=2, dim=1)

        if pred_part:
            cluster_class_logits = self._transformer_part_class_head(class_embeddings).permute(0, 2, 1).contiguous()
            cluster_mask_kernel = self._transformer_part_mask_head(mask_embeddings)
        else:
            cluster_class_logits = self._transformer_object_class_head(class_embeddings).permute(0, 2, 1).contiguous()
            cluster_mask_kernel = self._transformer_object_mask_head(mask_embeddings)
        cluster_class_logits = add_bias_towards_void(cluster_class_logits)
        mask_logits = torch.einsum('bchw,bcn->bnhw', pixel_space_normalized_feature, cluster_mask_kernel)
        
        mask_logits = self._pixel_space_mask_batch_norm(mask_logits.unsqueeze(dim=1)).squeeze(dim=1)


        return {
            'class_logits': cluster_class_logits,
            'mask_logits': mask_logits,
            'pixel_feature': pixel_space_normalized_feature}
    

# https://github.com/google-research/deeplab2/blob/7a01a7165e97b3325ad7ea9b6bcc02d67fecd07a/model/layers/dual_path_transformer.py#L107
class CompositorTransformerLayer(nn.Module):
    def __init__(
        self,
        num_part_classes=41,
        num_object_classes=159,
        in_channel_pixel=2048,
        in_channel_query=256,
        base_filters=128,
        num_heads=8,
        bottleneck_expansion=2,
        key_expansion=1,
        value_expansion=2,
        drop_path_prob=0.0,
    ):
        super().__init__()

        self._num_part_classes = num_part_classes
        self._num_object_classes = num_object_classes
        self._num_heads = num_heads
        self._bottleneck_channels = int(round(base_filters * bottleneck_expansion))
        self._total_key_depth = int(round(base_filters * key_expansion))
        self._total_value_depth = int(round(base_filters * value_expansion))

        # Per tf2 implementation, the same drop path prob are applied to:
        # 1. k-means update for object query
        # 2. self/cross-attetion for object query
        # 3. ffn for object query
        self.drop_path_kmeans = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity() 
        self.drop_path_attn = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity() 
        self.drop_path_ffn = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity() 

        initialization_std = self._bottleneck_channels ** -0.5
        self._part_query_conv1_bn_act = ConvBN(in_channel_query, self._bottleneck_channels, kernel_size=1, bias=False,
                                          norm='syncbn', act='gelu', conv_type='1d')

        self._pixel_conv1_bn_act = ConvBN(in_channel_pixel, self._bottleneck_channels, kernel_size=1, bias=False,
                                          norm='syncbn', act='gelu')

        self._part_query_qkv_conv_bn = ConvBN(self._bottleneck_channels, self._total_key_depth * 2 + self._total_value_depth, kernel_size=1, bias=False,
                                          norm='syncbn', act=None, conv_type='1d')
        trunc_normal_(self._part_query_qkv_conv_bn.conv.weight, std=initialization_std)

        self._pixel_v_conv_bn = ConvBN(self._bottleneck_channels, self._total_value_depth, kernel_size=1, bias=False,
                                          norm='syncbn', act=None)
        trunc_normal_(self._pixel_v_conv_bn.conv.weight, std=initialization_std)

        self._part_query_self_attention = AttentionOperation(channels_v=self._total_value_depth, num_heads=num_heads)

        self._part_query_conv3_bn = ConvBN(self._total_value_depth, in_channel_query, kernel_size=1, bias=False,
                                          norm='syncbn', act=None, conv_type='1d', norm_init=0.0)

        self._part_query_ffn_conv1_bn_act = ConvBN(in_channel_query, 2048, kernel_size=1, bias=False,
                                          norm='syncbn', act='gelu', conv_type='1d')
        self._part_query_ffn_conv2_bn = ConvBN(2048, in_channel_query, kernel_size=1, bias=False,
                                          norm='syncbn', act=None, conv_type='1d', norm_init=0.0)

        self._predictor = CompositorPredictor(in_channel_pixel=self._bottleneck_channels,
            num_part_classes=num_part_classes, num_object_classes=num_object_classes)
        self._kmeans_part_query_batch_norm_retrieved_value = get_norm('syncbn', self._total_value_depth)
        self._kmeans_part_query_conv3_bn = ConvBN(self._total_value_depth, in_channel_query, kernel_size=1, bias=False,
                                          norm='syncbn', act=None, conv_type='1d', norm_init=0.0)
        
        self._object_query_conv1_bn_act = ConvBN(in_channel_query, self._bottleneck_channels, kernel_size=1, bias=False,
                                          norm='syncbn', act='gelu', conv_type='1d')
        self._part_query_conv4_bn_act = ConvBN(in_channel_query, self._bottleneck_channels, kernel_size=1, bias=False,
                                          norm='syncbn', act='gelu', conv_type='1d')

        self._object_query_qkv_conv_bn = ConvBN(self._bottleneck_channels, self._total_key_depth * 2 + self._total_value_depth, kernel_size=1, bias=False,
                                          norm='syncbn', act=None, conv_type='1d')

        self._part_query_kv_conv_bn = ConvBN(self._bottleneck_channels, self._total_key_depth + self._total_value_depth, kernel_size=1, bias=False,
                                          norm='syncbn', act=None, conv_type='1d')

        self._object_query_attention = AttentionOperation(channels_v=self._total_value_depth, num_heads=num_heads)

        self._object_query_conv3_bn = ConvBN(self._total_value_depth, in_channel_query, kernel_size=1, bias=False,
                                          norm='syncbn', act=None, conv_type='1d', norm_init=0.0)

        self._object_query_ffn_conv1_bn_act = ConvBN(in_channel_query, 2048, kernel_size=1, bias=False,
                                          norm='syncbn', act='gelu', conv_type='1d')
        self._object_query_ffn_conv2_bn = ConvBN(2048, in_channel_query, kernel_size=1, bias=False,
                                          norm='syncbn', act=None, conv_type='1d', norm_init=0.0)


    def forward(self, pixel_feature, part_query_feature, object_query_feature):
        N, C, H, W = pixel_feature.shape
        _, D, P = part_query_feature.shape
        _, _, O = object_query_feature.shape
        pixel_space = self._pixel_conv1_bn_act(F.gelu(pixel_feature)) # N C H W
        part_query_space = self._part_query_conv1_bn_act(part_query_feature) # N x C x P

        # k-means cross-attention.
        pixel_value = self._pixel_v_conv_bn(pixel_space) # N C H W
        pixel_value = pixel_value.reshape(N, self._total_value_depth, H*W)
        # k-means assignment.
        part_prediction_result = self._predictor(
            mask_embeddings=part_query_space, class_embeddings=part_query_space, pixel_feature=pixel_space, pred_part=True)
        
        with torch.no_grad():
            clustering_result = part_prediction_result['mask_logits'].flatten(2).detach() # N L HW
            index = clustering_result.max(1, keepdim=True)[1]
            clustering_result = torch.zeros_like(clustering_result, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)

        with autocast(enabled=False):
        # k-means update.
            kmeans_update = torch.einsum('blm,bdm->bdl', clustering_result.float(), pixel_value.float()) # N x C x L

        kmeans_update = self._kmeans_part_query_batch_norm_retrieved_value(kmeans_update)
        kmeans_update = self._kmeans_part_query_conv3_bn(kmeans_update)
        part_query_feature = part_query_feature + self.drop_path_kmeans(kmeans_update)

        # query self-attention.
        part_query_qkv = self._part_query_qkv_conv_bn(part_query_space)
        part_query_q, part_query_k, part_query_v = torch.split(part_query_qkv,
         [self._total_key_depth, self._total_key_depth, self._total_value_depth], dim=1)
        part_query_q = part_query_q.reshape(N, self._num_heads, self._total_key_depth//self._num_heads, P)
        part_query_k = part_query_k.reshape(N, self._num_heads, self._total_key_depth//self._num_heads, P)
        part_query_v = part_query_v.reshape(N, self._num_heads, self._total_value_depth//self._num_heads, P)
        self_attn_update = self._part_query_self_attention(part_query_q, part_query_k, part_query_v)
        self_attn_update = self._part_query_conv3_bn(self_attn_update)
        part_query_feature = part_query_feature + self.drop_path_attn(self_attn_update)
        part_query_feature = F.gelu(part_query_feature)

        # FFN.
        ffn_update = self._part_query_ffn_conv1_bn_act(part_query_feature)
        ffn_update = self._part_query_ffn_conv2_bn(ffn_update)
        part_query_feature = part_query_feature + self.drop_path_ffn(ffn_update)
        part_query_feature = F.gelu(part_query_feature)

        # cross-attention.
        object_query_space = self._object_query_conv1_bn_act(object_query_feature) # N x C x O
        part_query_space = self._part_query_conv4_bn_act(part_query_feature) # N x C x P

        object_prediction_result = self._predictor(
            mask_embeddings=object_query_space, class_embeddings=object_query_space, pixel_feature=pixel_space, pred_part=False)

        part_query_kv = self._part_query_kv_conv_bn(part_query_space) # N C L
        part_query_k, part_query_v = torch.split(part_query_kv, [self._total_key_depth, self._total_value_depth], dim=1)
        part_query_k = part_query_k.reshape(N, self._num_heads, self._total_key_depth//self._num_heads, P)
        part_query_v = part_query_v.reshape(N, self._num_heads, self._total_value_depth//self._num_heads, P)

        # query self & cross-attention.
        object_query_qkv = self._object_query_qkv_conv_bn(object_query_space)
        object_query_q, object_query_k, object_query_v = torch.split(object_query_qkv,
         [self._total_key_depth, self._total_key_depth, self._total_value_depth], dim=1)
        object_query_q = object_query_q.reshape(N, self._num_heads, self._total_key_depth//self._num_heads, O)
        object_query_k = object_query_k.reshape(N, self._num_heads, self._total_key_depth//self._num_heads, O)
        object_query_v = object_query_v.reshape(N, self._num_heads, self._total_value_depth//self._num_heads, O)
        cross_attn_update = self._object_query_attention(object_query_q, torch.cat([object_query_k, part_query_k], dim=-1), torch.cat([object_query_v, part_query_v], dim=-1))
        cross_attn_update = self._object_query_conv3_bn(cross_attn_update)
        object_query_feature = object_query_feature + self.drop_path_attn(cross_attn_update)
        object_query_feature = F.gelu(object_query_feature)

        # FFN.
        ffn_update = self._object_query_ffn_conv1_bn_act(object_query_feature)
        ffn_update = self._object_query_ffn_conv2_bn(ffn_update)
        object_query_feature = object_query_feature + self.drop_path_ffn(ffn_update)
        object_query_feature = F.gelu(object_query_feature)

        return part_query_feature, object_query_feature, part_prediction_result, object_prediction_result


@TRANSFORMER_DECODER_REGISTRY.register()
class CompositorTransformerDecoder(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        dec_layers: List[int],
        in_channels: List[int],
        num_part_classes: int,
        num_object_classes: int,
        num_part_queries: int,
        num_object_queries: int,
        drop_path_prob: float,
    ):
        """
        NOTE: this interface is experimental.
        Args:
        """
        super().__init__()
        
        # define Transformer decoder here
        self._compositor_transformer_layers = nn.ModuleList()
        self._num_blocks = dec_layers
        os2channels = {32: in_channels[0], 16: in_channels[1], 8: in_channels[2]}

        for index, output_stride in enumerate([32, 16, 8]):
            for _ in range(self._num_blocks[index]):
                self._compositor_transformer_layers.append(
                    CompositorTransformerLayer(
                        num_part_classes=num_part_classes+1,
                        num_object_classes=num_object_classes+1,
                        in_channel_pixel=os2channels[output_stride],
                        in_channel_query=256,
                        base_filters=128,
                        num_heads=8,
                        bottleneck_expansion=2,
                        key_expansion=1,
                        value_expansion=2,
                        drop_path_prob=drop_path_prob)
                )

        self._num_part_queries = num_part_queries
        self._num_object_queries = num_object_queries
        # learnable query features
        self._part_cluster_centers = nn.Embedding(256, num_part_queries)
        self._object_cluster_centers = nn.Embedding(256, num_object_queries)
        trunc_normal_(self._part_cluster_centers.weight, std=1.0)
        trunc_normal_(self._object_cluster_centers.weight, std=1.0)

        self._part_class_embedding_projection = ConvBN(256, 256, kernel_size=1, bias=False, norm='syncbn', act='gelu',
                                                  conv_type='1d')
        self._part_mask_embedding_projection = ConvBN(256, 256, kernel_size=1, bias=False, norm='syncbn', act='gelu',
                                                  conv_type='1d')
        
        self._object_class_embedding_projection = ConvBN(256, 256, kernel_size=1, bias=False, norm='syncbn', act='gelu',
                                                  conv_type='1d')
        self._object_mask_embedding_projection = ConvBN(256, 256, kernel_size=1, bias=False, norm='syncbn', act='gelu',
                                                  conv_type='1d')

        self._predictor = CompositorPredictor(in_channel_pixel=in_channels[-1], num_part_classes=num_part_classes+1, num_object_classes=num_object_classes+1)


    @classmethod
    def from_config(cls, cfg, input_shape_from_backbone):
        ret = {}
        ret["dec_layers"] = cfg.MODEL.COMPOSITOR.TRANS_DEC.DEC_LAYERS
        ret["in_channels"] = cfg.MODEL.COMPOSITOR.TRANS_DEC.IN_CHANNELS   
        ret["num_part_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_PART_CLASSES
        ret["num_object_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_OBJECT_CLASSES
        ret["num_part_queries"] = cfg.MODEL.COMPOSITOR.TRANS_DEC.NUM_PART_QUERIES
        ret["num_object_queries"] = cfg.MODEL.COMPOSITOR.TRANS_DEC.NUM_OBJECT_QUERIES
        ret["drop_path_prob"] = cfg.MODEL.COMPOSITOR.TRANS_DEC.DROP_PATH_PROB
        return ret


    def forward(self, x, panoptic_features):
        B = x[0].shape[0]
        part_cluster_centers = self._part_cluster_centers.weight.unsqueeze(0).repeat(B, 1, 1) # B x C x L
        object_cluster_centers = self._object_cluster_centers.weight.unsqueeze(0).repeat(B, 1, 1) # B x C x L

        current_transformer_idx = 0

        part_predictions_class = []
        part_predictions_mask = []
        object_predictions_class = []
        object_predictions_mask = []
        predictions_pixel_feature = []

        for i, feat in enumerate(x):
            for _ in range(self._num_blocks[i]):
                part_cluster_centers, object_cluster_centers, part_prediction_result, object_prediction_result = self._compositor_transformer_layers[current_transformer_idx](
                    pixel_feature=feat, part_query_feature=part_cluster_centers, object_query_feature=object_cluster_centers
                )

                part_predictions_class.append(part_prediction_result['class_logits'])
                part_predictions_mask.append(part_prediction_result['mask_logits'])
                object_predictions_class.append(object_prediction_result['class_logits'])
                object_predictions_mask.append(object_prediction_result['mask_logits'])
                predictions_pixel_feature.append(object_prediction_result['pixel_feature'])

                current_transformer_idx += 1

        part_class_embeddings = self._part_class_embedding_projection(part_cluster_centers)
        part_mask_embeddings = self._part_mask_embedding_projection(part_cluster_centers)
        object_class_embeddings = self._object_class_embedding_projection(object_cluster_centers)
        object_mask_embeddings = self._object_mask_embedding_projection(object_cluster_centers)

        # Final predictions.
        part_prediction_result = self._predictor(
            class_embeddings=part_class_embeddings,
            mask_embeddings=part_mask_embeddings,
            pixel_feature=panoptic_features,
            pred_part=True,
        )
        part_predictions_class.append(part_prediction_result['class_logits'])
        part_predictions_mask.append(part_prediction_result['mask_logits'])

        object_prediction_result = self._predictor(
            class_embeddings=object_class_embeddings,
            mask_embeddings=object_mask_embeddings,
            pixel_feature=panoptic_features,
            pred_part=False,
        )
        object_predictions_class.append(object_prediction_result['class_logits'])
        object_predictions_mask.append(object_prediction_result['mask_logits'])
        predictions_pixel_feature.append(object_prediction_result['pixel_feature'])

        out = {
            'part_pred_logits': part_predictions_class[-1],
            'part_pred_masks': part_predictions_mask[-1],
            'object_pred_logits': object_predictions_class[-1],
            'object_pred_masks': object_predictions_mask[-1],
            'pixel_feature': predictions_pixel_feature[-1],
            'aux_outputs': self._set_aux_loss(
                part_predictions_class, object_predictions_class, part_predictions_mask, 
                object_predictions_mask, predictions_pixel_feature
            ),      
        }

        return out


    @torch.jit.unused
    def _set_aux_loss(self, part_outputs_class, object_outputs_class, part_outputs_seg_masks, object_outputs_seg_masks, outputs_pixel_feature):
        target_size = part_outputs_seg_masks[-1].shape[-2:]
        align_corners = (target_size[0] % 2 == 1)
        return [
            {"part_pred_logits": a, "object_pred_logits": b, "part_pred_masks": F.interpolate(c, size=target_size, mode="bilinear", align_corners=align_corners),
            "object_pred_masks": F.interpolate(d, size=target_size, mode="bilinear", align_corners=align_corners),
            "pixel_feature": F.interpolate(e, size=target_size, mode="bilinear", align_corners=align_corners)}
            for a, b, c, d, e in zip(part_outputs_class[:-1], object_outputs_class[:-1], part_outputs_seg_masks[:-1], 
                                        object_outputs_seg_masks[:-1], outputs_pixel_feature[:-1])
        ]
