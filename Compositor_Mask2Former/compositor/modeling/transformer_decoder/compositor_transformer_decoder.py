# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import warnings
import math
from typing import List, Optional, Tuple
from torch.types import _dtype as DType
import fvcore.nn.weight_init as weight_init
import torch
from torch import _VF
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from mask2former.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine
from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY


def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.COMPOSITOR.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)


class SoftClusterMultiheadAttention(nn.MultiheadAttention):

    def __init__(self, embed_dim, num_heads, dropout=0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, device, dtype)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = self.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights)
        else:
            attn_output, attn_output_weights = self.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def multi_head_attention_forward(self, query: Tensor, key: Tensor, value: Tensor, embed_dim_to_check: int, num_heads: int,
                                     in_proj_weight: Tensor, in_proj_bias: Optional[Tensor], bias_k: Optional[Tensor], bias_v: Optional[Tensor], 
                                     add_zero_attn: bool, dropout_p: float, out_proj_weight: Tensor, out_proj_bias: Optional[Tensor], training: bool = True,
                                     key_padding_mask: Optional[Tensor] = None, need_weights: bool = True, attn_mask: Optional[Tensor] = None, 
                                     use_separate_proj_weight: bool = False, q_proj_weight: Optional[Tensor] = None, k_proj_weight: Optional[Tensor] = None,
                                     v_proj_weight: Optional[Tensor] = None, static_k: Optional[Tensor] = None, static_v: Optional[Tensor] = None,
                                     average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:

        def _get_softmax_dim(name: str, ndim: int, stacklevel: int) -> int:
            warnings.warn(
                "Implicit dimension choice for {} has been deprecated. "
                "Change the call to include dim=X as an argument.".format(name),
                stacklevel=stacklevel,
            )
            if ndim == 0 or ndim == 1 or ndim == 3:
                ret = 0
            else:
                ret = 1
            return ret

        def softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[DType] = None) -> Tensor:
            r"""Applies a softmax function.
            Softmax is defined as:
            :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`
            It is applied to all slices along dim, and will re-scale them so that the elements
            lie in the range `[0, 1]` and sum to 1.
            See :class:`~torch.nn.Softmax` for more details.
            Args:
                input (Tensor): input
                dim (int): A dimension along which softmax will be computed.
                dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
                If specified, the input tensor is casted to :attr:`dtype` before the operation
                is performed. This is useful for preventing data type overflows. Default: None.
            .. note::
                This function doesn't work directly with NLLLoss,
                which expects the Log to be computed between the Softmax and itself.
                Use log_softmax instead (it's faster and has better numerical properties).
            """
            if dim is None:
                dim = _get_softmax_dim("softmax", input.dim(), _stacklevel)
            if dtype is None:
                ret = input.softmax(dim)
            else:
                ret = input.softmax(dim, dtype=dtype)
            return ret

        def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
            r"""
            During training, randomly zeroes some of the elements of the input
            tensor with probability :attr:`p` using samples from a Bernoulli
            distribution.
            See :class:`~torch.nn.Dropout` for details.
            Args:
                p: probability of an element to be zeroed. Default: 0.5
                training: apply dropout if is ``True``. Default: ``True``
                inplace: If set to ``True``, will do this operation in-place. Default: ``False``
            """
            if p < 0.0 or p > 1.0:
                raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
            return _VF.dropout_(input, p, training) if inplace else _VF.dropout(input, p, training)

        # This method is override for K-MaX
        def _scaled_dot_product_attention(
            q: Tensor,
            k: Tensor,
            v: Tensor,
            attn_mask: Optional[Tensor] = None,
            dropout_p: float = 0.0,
        ) -> Tuple[Tensor, Tensor]:
            r"""
            Computes scaled dot product attention on query, key and value tensors, using
            an optional attention mask if passed, and applying dropout if a probability
            greater than 0.0 is specified.
            Returns a tensor pair containing attended values and attention weights.
            Args:
                q, k, v: query, key and value tensors. See Shape section for shape details.
                attn_mask: optional tensor containing mask values to be added to calculated
                    attention. May be 2D or 3D; see Shape section for details.
                dropout_p: dropout probability. If greater than 0.0, dropout is applied.
            Shape:
                - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
                    and E is embedding dimension.
                - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                    and E is embedding dimension.
                - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                    and E is embedding dimension.
                - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
                    shape :math:`(Nt, Ns)`.
                - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
                    have shape :math:`(B, Nt, Ns)`
            """
            B, Nt, E = q.shape
            q = q / math.sqrt(E)
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(q, k.transpose(-2, -1))
            if attn_mask is not None:
                attn += attn_mask
            attn = softmax(attn, dim=1)
            attn = attn / attn.sum(dim=1, keepdim=True)
            if dropout_p > 0.0:
                attn = dropout(attn, p=dropout_p)
            # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
            output = torch.bmm(attn, v)
            return output, attn

        def _mha_shape_check(query: Tensor, key: Tensor, value: Tensor,
                     key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor], num_heads: int):
            # Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
            # and returns if the input is batched or not.
            # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

            # Shape check.
            if query.dim() == 3:
                # Batched Inputs
                is_batched = True
                assert key.dim() == 3 and value.dim() == 3, \
                    ("For batched (3-D) `query`, expected `key` and `value` to be 3-D"
                    f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")
                if key_padding_mask is not None:
                    assert key_padding_mask.dim() == 2, \
                        ("For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                        f" but found {key_padding_mask.dim()}-D tensor instead")
                if attn_mask is not None:
                    assert attn_mask.dim() in (2, 3), \
                        ("For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                        f" but found {attn_mask.dim()}-D tensor instead")
            elif query.dim() == 2:
                # Unbatched Inputs
                is_batched = False
                assert key.dim() == 2 and value.dim() == 2, \
                    ("For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
                    f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")

                if key_padding_mask is not None:
                    assert key_padding_mask.dim() == 1, \
                        ("For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
                        f" but found {key_padding_mask.dim()}-D tensor instead")

                if attn_mask is not None:
                    assert attn_mask.dim() in (2, 3), \
                        ("For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                        f" but found {attn_mask.dim()}-D tensor instead")
                    if attn_mask.dim() == 3:
                        expected_shape = (num_heads, query.shape[0], key.shape[0])
                        assert attn_mask.shape == expected_shape, \
                            (f"Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}")
            else:
                raise AssertionError(
                    f"query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor")

            return is_batched

        linear = torch._C._nn.linear

        def _pad_circular(input: Tensor, padding: List[int]) -> Tensor:
            """Circularly pads tensor.
            Tensor values at the beginning are used to pad the end, and values at the
            end are used to pad the beginning. For example, consider a single dimension
            with values [0, 1, 2, 3]. With circular padding of (1, 1) it would be
            padded to [3, 0, 1, 2, 3, 0], and with padding (1, 2) it would be padded to
            [3, 0, 1, 2, 3, 0, 1]. If negative padding is applied then the ends of the
            tensor get removed. With circular padding of (-1, -1) the previous example
            would become [1, 2]. Circular padding of (-1, 1) would produce
            [1, 2, 3, 1].
            The first and second dimensions of the tensor are not padded.
            Args:
                input: Tensor with shape :math:`(N, C, D[, H, W])`.
                padding: Tuple containing the number of elements to pad each side of
                    the tensor. The length of padding must be twice the number of
                    paddable dimensions. For example, the length of padding should be 4
                    for a tensor of shape :math:`(N, C, H, W)`, and the length should
                    be 6 for a tensor of shape :math:`(N, C, D, H, W)`.
            Examples::
                >>> x = torch.tensor([[[[0, 1, 2], [3, 4, 5]]]])  # Create tensor
                >>> # Example 1
                >>> padding = (1, 1, 1, 1)
                >>> y = F.pad(x, padding, mode='circular')
                >>> print(y)
                tensor([[[[5, 3, 4, 5, 3],
                        [2, 0, 1, 2, 0],
                        [5, 3, 4, 5, 3],
                        [2, 0, 1, 2, 0]]]])
                >>> print(y.shape)
                torch.Size([1, 1, 4, 5])
                >>> # Example 2
                >>> padding = (1, 1, 2, 2)
                >>> z = F.pad(x, padding, mode='circular')
                >>> print(z)
                tensor([[[[2, 0, 1, 2, 0],
                        [5, 3, 4, 5, 3],
                        [2, 0, 1, 2, 0],
                        [5, 3, 4, 5, 3],
                        [2, 0, 1, 2, 0],
                        [5, 3, 4, 5, 3]]]])
                >>> print(z.shape)
                torch.Size([1, 1, 6, 5])
            """
            in_shape = input.shape
            paddable_shape = in_shape[2:]
            ndim = len(paddable_shape)

            for idx, size in enumerate(paddable_shape):
                # Only supports wrapping around once
                assert padding[-(idx * 2 + 1)] <= size, "Padding value causes wrapping around more than once."
                assert padding[-(idx * 2 + 2)] <= size, "Padding value causes wrapping around more than once."
                # Negative padding should not result in negative sizes
                assert (
                    padding[-(idx * 2 + 1)] + padding[-(idx * 2 + 2)] + size >= 0
                ), "Negative padding value is resulting in an empty dimension."

            # Get shape of padded tensor
            out_shape = in_shape[:2]
            for idx, size in enumerate(paddable_shape):
                out_shape += (size + padding[-(idx * 2 + 1)] + padding[-(idx * 2 + 2)],)

            out = input.new_empty(out_shape)

            # Put original array in padded array
            if ndim == 1:
                out_d0 = max(padding[-2], 0)
                out_d1 = out_shape[2] - max(padding[-1], 0)

                in_d0 = max(-padding[-2], 0)
                in_d1 = in_shape[2] - max(-padding[-1], 0)

                out[..., out_d0:out_d1] = input[..., in_d0:in_d1]
            elif ndim == 2:
                out_d0 = max(padding[-2], 0)
                out_d1 = out_shape[2] - max(padding[-1], 0)

                out_h0 = max(padding[-4], 0)
                out_h1 = out_shape[3] - max(padding[-3], 0)

                in_d0 = max(-padding[-2], 0)
                in_d1 = in_shape[2] - max(-padding[-1], 0)

                in_h0 = max(-padding[-4], 0)
                in_h1 = in_shape[3] - max(-padding[-3], 0)

                out[..., out_d0:out_d1, out_h0:out_h1] = input[..., in_d0:in_d1, in_h0:in_h1]
            elif ndim == 3:
                out_d0 = max(padding[-2], 0)
                out_d1 = out_shape[2] - max(padding[-1], 0)

                out_h0 = max(padding[-4], 0)
                out_h1 = out_shape[3] - max(padding[-3], 0)

                out_w0 = max(padding[-6], 0)
                out_w1 = out_shape[4] - max(padding[-5], 0)

                in_d0 = max(-padding[-2], 0)
                in_d1 = in_shape[2] - max(-padding[-1], 0)

                in_h0 = max(-padding[-4], 0)
                in_h1 = in_shape[3] - max(-padding[-3], 0)

                in_w0 = max(-padding[-6], 0)
                in_w1 = in_shape[4] - max(-padding[-5], 0)

                out[..., out_d0:out_d1, out_h0:out_h1, out_w0:out_w1] = input[..., in_d0:in_d1, in_h0:in_h1, in_w0:in_w1]

            # The following steps first pad the beginning of the tensor (left side),
            # and then pad the end of the tensor (right side).
            # Note: Corners will be written more than once when ndim > 1.

            # Only in cases where padding values are > 0 are when additional copying
            # is required.

            # Pad first dimension (depth)
            if padding[-2] > 0:
                i0 = out_shape[2] - padding[-2] - max(padding[-1], 0)
                i1 = out_shape[2] - max(padding[-1], 0)
                o0 = 0
                o1 = padding[-2]
                out[:, :, o0:o1] = out[:, :, i0:i1]
            if padding[-1] > 0:
                i0 = max(padding[-2], 0)
                i1 = max(padding[-2], 0) + padding[-1]
                o0 = out_shape[2] - padding[-1]
                o1 = out_shape[2]
                out[:, :, o0:o1] = out[:, :, i0:i1]

            # Pad second dimension (height)
            if len(padding) > 2:
                if padding[-4] > 0:
                    i0 = out_shape[3] - padding[-4] - max(padding[-3], 0)
                    i1 = out_shape[3] - max(padding[-3], 0)
                    o0 = 0
                    o1 = padding[-4]
                    out[:, :, :, o0:o1] = out[:, :, :, i0:i1]
                if padding[-3] > 0:
                    i0 = max(padding[-4], 0)
                    i1 = max(padding[-4], 0) + padding[-3]
                    o0 = out_shape[3] - padding[-3]
                    o1 = out_shape[3]
                    out[:, :, :, o0:o1] = out[:, :, :, i0:i1]

            # Pad third dimension (width)
            if len(padding) > 4:
                if padding[-6] > 0:
                    i0 = out_shape[4] - padding[-6] - max(padding[-5], 0)
                    i1 = out_shape[4] - max(padding[-5], 0)
                    o0 = 0
                    o1 = padding[-6]
                    out[:, :, :, :, o0:o1] = out[:, :, :, :, i0:i1]
                if padding[-5] > 0:
                    i0 = max(padding[-6], 0)
                    i1 = max(padding[-6], 0) + padding[-5]
                    o0 = out_shape[4] - padding[-5]
                    o1 = out_shape[4]
                    out[:, :, :, :, o0:o1] = out[:, :, :, :, i0:i1]

            return out

        def _pad(input: Tensor, pad: List[int], mode: str = "constant", value: float = 0.0) -> Tensor:
            r"""Pads tensor.
            Padding size:
                The padding size by which to pad some dimensions of :attr:`input`
                are described starting from the last dimension and moving forward.
                :math:`\left\lfloor\frac{\text{len(pad)}}{2}\right\rfloor` dimensions
                of ``input`` will be padded.
                For example, to pad only the last dimension of the input tensor, then
                :attr:`pad` has the form
                :math:`(\text{padding\_left}, \text{padding\_right})`;
                to pad the last 2 dimensions of the input tensor, then use
                :math:`(\text{padding\_left}, \text{padding\_right},`
                :math:`\text{padding\_top}, \text{padding\_bottom})`;
                to pad the last 3 dimensions, use
                :math:`(\text{padding\_left}, \text{padding\_right},`
                :math:`\text{padding\_top}, \text{padding\_bottom}`
                :math:`\text{padding\_front}, \text{padding\_back})`.
            Padding mode:
                See :class:`torch.nn.ConstantPad2d`, :class:`torch.nn.ReflectionPad2d`, and
                :class:`torch.nn.ReplicationPad2d` for concrete examples on how each of the
                padding modes works. Constant padding is implemented for arbitrary dimensions.
                Replicate and reflection padding is implemented for padding the last 3
                dimensions of 5D input tensor, or the last 2 dimensions of 4D input
                tensor, or the last dimension of 3D input tensor.
            Note:
                When using the CUDA backend, this operation may induce nondeterministic
                behaviour in its backward pass that is not easily switched off.
                Please see the notes on :doc:`/notes/randomness` for background.
            Args:
                input (Tensor): N-dimensional tensor
                pad (tuple): m-elements tuple, where
                    :math:`\frac{m}{2} \leq` input dimensions and :math:`m` is even.
                mode: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
                    Default: ``'constant'``
                value: fill value for ``'constant'`` padding. Default: ``0``
            Examples::
                >>> t4d = torch.empty(3, 3, 4, 2)
                >>> p1d = (1, 1) # pad last dim by 1 on each side
                >>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
                >>> print(out.size())
                torch.Size([3, 3, 4, 4])
                >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
                >>> out = F.pad(t4d, p2d, "constant", 0)
                >>> print(out.size())
                torch.Size([3, 3, 8, 4])
                >>> t4d = torch.empty(3, 3, 4, 2)
                >>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
                >>> out = F.pad(t4d, p3d, "constant", 0)
                >>> print(out.size())
                torch.Size([3, 9, 7, 3])
            """
            assert len(pad) % 2 == 0, "Padding length must be divisible by 2"
            assert len(pad) // 2 <= input.dim(), "Padding length too large"
            if mode == "constant":
                return _VF.constant_pad_nd(input, pad, value)
            else:
                assert value == 0.0, 'Padding mode "{}"" doesn\'t take in value argument'.format(mode)
                if len(pad) == 2 and (input.dim() == 2 or input.dim() == 3):
                    if mode == "reflect":
                        return torch._C._nn.reflection_pad1d(input, pad)
                    elif mode == "replicate":
                        return torch._C._nn.replication_pad1d(input, pad)
                    elif mode == "circular":
                        return _pad_circular(input, pad)
                    else:
                        raise NotImplementedError

                elif len(pad) == 4 and (input.dim() == 3 or input.dim() == 4):
                    if mode == "reflect":
                        return torch._C._nn.reflection_pad2d(input, pad)
                    elif mode == "replicate":
                        return torch._C._nn.replication_pad2d(input, pad)
                    elif mode == "circular":
                        return _pad_circular(input, pad)
                    else:
                        raise NotImplementedError

                elif len(pad) == 6 and (input.dim() == 4 or input.dim() == 5):
                    if mode == "reflect":
                        return torch._C._nn.reflection_pad3d(input, pad)
                    elif mode == "replicate":
                        return torch._C._nn.replication_pad3d(input, pad)
                    elif mode == "circular":
                        return _pad_circular(input, pad)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError("Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now")

        pad = _pad

        def _in_projection_packed(
            q: Tensor,
            k: Tensor,
            v: Tensor,
            w: Tensor,
            b: Optional[Tensor] = None,
        ):
            r"""
            Performs the in-projection step of the attention operation, using packed weights.
            Output is a triple containing projection tensors for query, key and value.
            Args:
                q, k, v: query, key and value tensors to be projected. For self-attention,
                    these are typically the same tensor; for encoder-decoder attention,
                    k and v are typically the same tensor. (We take advantage of these
                    identities for performance if they are present.) Regardless, q, k and v
                    must share a common embedding dimension; otherwise their shapes may vary.
                w: projection weights for q, k and v, packed into a single tensor. Weights
                    are packed along dimension 0, in q, k, v order.
                b: optional projection biases for q, k and v, packed into a single tensor
                    in q, k, v order.
            Shape:
                Inputs:
                - q: :math:`(..., E)` where E is the embedding dimension
                - k: :math:`(..., E)` where E is the embedding dimension
                - v: :math:`(..., E)` where E is the embedding dimension
                - w: :math:`(E * 3, E)` where E is the embedding dimension
                - b: :math:`E * 3` where E is the embedding dimension
                Output:
                - in output list :math:`[q', k', v']`, each output tensor will have the
                    same shape as the corresponding input tensor.
            """
            E = q.size(-1)
            if k is v:
                if q is k:
                    # self-attention
                    return linear(q, w, b).chunk(3, dim=-1)
                else:
                    # encoder-decoder attention
                    w_q, w_kv = w.split([E, E * 2])
                    if b is None:
                        b_q = b_kv = None
                    else:
                        b_q, b_kv = b.split([E, E * 2])
                    return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
            else:
                w_q, w_k, w_v = w.chunk(3)
                if b is None:
                    b_q = b_k = b_v = None
                else:
                    b_q, b_k, b_v = b.chunk(3)
                return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

        def _in_projection(
            q: Tensor,
            k: Tensor,
            v: Tensor,
            w_q: Tensor,
            w_k: Tensor,
            w_v: Tensor,
            b_q: Optional[Tensor] = None,
            b_k: Optional[Tensor] = None,
            b_v: Optional[Tensor] = None,
        ):
            r"""
            Performs the in-projection step of the attention operation. This is simply
            a triple of linear projections, with shape constraints on the weights which
            ensure embedding dimension uniformity in the projected outputs.
            Output is a triple containing projection tensors for query, key and value.
            Args:
                q, k, v: query, key and value tensors to be projected.
                w_q, w_k, w_v: weights for q, k and v, respectively.
                b_q, b_k, b_v: optional biases for q, k and v, respectively.
            Shape:
                Inputs:
                - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
                    number of leading dimensions.
                - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
                    number of leading dimensions.
                - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
                    number of leading dimensions.
                - w_q: :math:`(Eq, Eq)`
                - w_k: :math:`(Eq, Ek)`
                - w_v: :math:`(Eq, Ev)`
                - b_q: :math:`(Eq)`
                - b_k: :math:`(Eq)`
                - b_v: :math:`(Eq)`
                Output: in output triple :math:`(q', k', v')`,
                - q': :math:`[Qdims..., Eq]`
                - k': :math:`[Kdims..., Eq]`
                - v': :math:`[Vdims..., Eq]`
            """
            Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
            assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
            assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
            assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
            assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
            assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
            assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
            return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

        is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

        # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
        # is batched, run the computation and before returning squeeze the
        # batch dimension so that the output doesn't carry this temporary batch dimension.
        if not is_batched:
            # unsqueeze if the input is unbatched
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        assert embed_dim == embed_dim_to_check, \
            f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        if use_separate_proj_weight:
            # allow MHA to have different embedding dimensions when separate projection weights are used
            assert key.shape[:2] == value.shape[:2], \
                f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
        else:
            assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

        #
        # compute in-projection
        #
        if not use_separate_proj_weight:
            q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        else:
            assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
            assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
            assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
            if in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = in_proj_bias.chunk(3)
            q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

        # prep attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)
            else:
                assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                    f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        # prep key padding mask
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)

        # add bias along batch dimension (currently second)
        if bias_k is not None and bias_v is not None:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert bias_k is None
            assert bias_v is None

        #
        # reshape q, k, v for multihead attention and make em batch first
        #
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if static_k is None:
            k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert static_k.size(0) == bsz * num_heads, \
                f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
            assert static_k.size(2) == head_dim, \
                f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
            k = static_k
        if static_v is None:
            v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert static_v.size(0) == bsz * num_heads, \
                f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
            assert static_v.size(2) == head_dim, \
                f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
            v = static_v

        # add zero attention along batch dimension (now first)
        if add_zero_attn:
            zero_attn_shape = (bsz * num_heads, 1, head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))

        # update source sequence length after adjustments
        src_len = k.size(1)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
                expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        # adjust dropout probability
        if not training:
            dropout_p = 0.0

        #
        # (deep breath) calculate attention and out projection
        #
        attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        if need_weights:
            # optionally average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.sum(dim=1) / num_heads

            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
                attn_output_weights = attn_output_weights.squeeze(0)
            return attn_output, attn_output_weights
        else:
            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
            return attn_output, None
        

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)

class SoftClusterAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = SoftClusterMultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)
    

class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class CompositorTransformerDecoder(nn.Module):
    
    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_part_classes: int,
        num_object_classes: int,
        hidden_dim: int,
        num_part_queries: int,
        num_object_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_part_classes: number of part classes
            num_object_classes: number of object classes
            hidden_dim: Transformer feature dimension
            num_part_queries: number of part queries
            num_object_queries: number of object queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_pixel2part_layers = nn.ModuleList()
        self.transformer_part_self_attention_layers = nn.ModuleList()
        self.transformer_part2object_layers = nn.ModuleList()
        self.transformer_object_self_attention_layers = nn.ModuleList()
        self.transformer_part_ffn_layers = nn.ModuleList()
        self.transformer_object_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_pixel2part_layers.append(
                SoftClusterAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_part_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_part2object_layers.append(
                SoftClusterAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_object_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_part_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_object_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_part_queries = num_part_queries
        self.num_object_queries = num_object_queries

        # learnable query features
        self.part_query_feat = nn.Embedding(num_part_queries, hidden_dim)
        self.object_query_feat = nn.Embedding(num_object_queries, hidden_dim)
        # learnable query p.e.
        self.part_query_embed = nn.Embedding(num_part_queries, hidden_dim)
        self.object_query_embed = nn.Embedding(num_object_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.part_class_embed = nn.Linear(hidden_dim, num_part_classes + 1)
            self.object_class_emded = nn.Linear(hidden_dim, num_object_classes + 1)
        self.part_mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.object_mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_part_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_PART_CLASSES
        ret["num_object_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_OBJECT_CLASSES
        ret["hidden_dim"] = cfg.MODEL.COMPOSITOR.HIDDEN_DIM
        ret["num_part_queries"] = cfg.MODEL.COMPOSITOR.NUM_PART_QUERIES
        ret["num_object_queries"] = cfg.MODEL.COMPOSITOR.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.COMPOSITOR.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.COMPOSITOR.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.COMPOSITOR.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.COMPOSITOR.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.COMPOSITOR.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.COMPOSITOR.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        return ret

    def forward(self, x, mask_features, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        part_query_embed = self.part_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        part_output = self.part_query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        object_query_embed = self.object_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        object_output = self.object_query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        part_predictions_class = []
        part_predictions_mask = []
        object_predictions_class = []
        object_predictions_mask = []

        # prediction heads on learnable query features
        part_outputs_class, object_outputs_class, part_outputs_mask, object_outputs_mask, object_attn_mask = self.forward_prediction_heads(part_output, object_output, mask_features, attn_mask_target_size=size_list[0])
        part_predictions_class.append(part_outputs_class)
        object_predictions_class.append(object_outputs_class)
        part_predictions_mask.append(part_outputs_mask)
        object_predictions_mask.append(object_outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels

            mask_index = torch.where(object_attn_mask.sum(1) == object_attn_mask.shape[1])
            object_attn_mask[mask_index[0],:,mask_index[1]] = False

            part_output = self.transformer_pixel2part_layers[i](
                part_output, src[level_index],
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=part_query_embed
            )

            part_output = self.transformer_part_self_attention_layers[i](
                part_output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=part_query_embed
            )

            object_output = self.transformer_part2object_layers[i](
                object_output, part_output,
                memory_mask=object_attn_mask,
                memory_key_padding_mask=None,
                pos=part_query_embed, query_pos=object_query_embed
            )
            
            # FFN
            part_output = self.transformer_part_ffn_layers[i](
                part_output
            )
            object_output = self.transformer_object_ffn_layers[i](
                object_output
            )

            part_outputs_class, object_outputs_class, part_outputs_mask, object_outputs_mask, object_attn_mask = self.forward_prediction_heads(part_output, object_output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            part_predictions_class.append(part_outputs_class)
            object_predictions_class.append(object_outputs_class)
            part_predictions_mask.append(part_outputs_mask)
            object_predictions_mask.append(object_outputs_mask)

        out = {
            'part_pred_logits': part_predictions_class[-1],
            'object_pred_logits': object_predictions_class[-1],
            'part_pred_masks': part_predictions_mask[-1],
            'object_pred_masks': object_predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                part_predictions_class if self.mask_classification else None, object_predictions_class if self.mask_classification else None, part_predictions_mask, object_predictions_mask
            )
        }
        return out

    def forward_prediction_heads(self, part_output, object_output, mask_features, attn_mask_target_size):
        part_decoder_output = self.decoder_norm(part_output)
        part_decoder_output = part_decoder_output.transpose(0, 1)
        part_outputs_class = self.part_class_embed(part_decoder_output) 
        part_mask_embed = self.part_mask_embed(part_decoder_output) 
        part_outputs_mask = torch.einsum("bqc,bchw->bqhw", part_mask_embed, mask_features) # bnc,bchw->bnhw

        object_decoder_output = self.decoder_norm(object_output)
        object_decoder_output = object_decoder_output.transpose(0, 1)
        object_outputs_class = self.object_class_emded(object_decoder_output)
        object_mask_embed = self.object_mask_embed(object_decoder_output)
        object_outputs_mask = torch.einsum("bqc,bchw->bqhw", object_mask_embed, mask_features) # bnc,bchw->bnhw

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        part_attn_mask = F.interpolate(part_outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        part_attn_mask = (part_attn_mask.sigmoid().flatten(2) < 0.5).bool()
        object_attn_mask = ((~part_attn_mask).sum(2) == 0).bool()
        object_attn_mask = object_attn_mask.unsqueeze(1).unsqueeze(1).repeat(1,self.num_heads,self.num_object_queries,1).flatten(0,1)
        object_attn_mask = object_attn_mask.detach()

        return part_outputs_class, object_outputs_class, part_outputs_mask, object_outputs_mask, object_attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, part_outputs_class, object_outputs_class, part_outputs_seg_masks, object_outputs_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"part_pred_logits": a, "object_pred_logits": b, "part_pred_masks": c, "object_pred_masks": d}
                for a, b, c, d in zip(part_outputs_class[:-1], object_outputs_class[:-1], part_outputs_seg_masks[:-1], object_outputs_masks[:-1])
            ]
        else:
            return [{"part_pred_masks": c, "object_pred_masks": d} for c, d in zip(part_outputs_seg_masks[:-1], object_outputs_masks[:-1])]
