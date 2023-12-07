#!/usr/bin/env python3
# Code by authors of ViR: Towards Efficient Vision Retention Backbones

import logging
import math
from math import log
from functools import partial
from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import math

import torch.nn.functional as F
import torch.utils.checkpoint

from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import PatchEmbed , Mlp, DropPath, trunc_normal_, lecun_normal_, resample_patch_embed, \
    resample_abs_pos_embed, PatchDropout
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import named_apply
from timm.models._registry import register_model
import numpy as np

__all__ = ['VisionRetention']  # model_registry will add each entrypoint fn to this


_logger = logging.getLogger(__name__)


class Retention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)


    def parallel(self, x, mask, act='softmax'):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        retention = q @ k.transpose(-2, -1)
        retention = retention * mask
        retention = retention.softmax(dim=-1)
        retention = (retention @ v).transpose(1, 2).reshape(B, N, C)
        retention = self.out_proj(retention)
        return retention, None


    def recurrent(self, x, gamma, state_prev=None, act='softmax'):
        B, C = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        states = k.unsqueeze(-2).transpose(-1, -2) @ v.unsqueeze(-2)
        if state_prev is not None:
            gamma = gamma.reshape(1, -1, 1, 1)
            states = states + state_prev * gamma

        retention = torch.matmul(q.unsqueeze(2), states)
        retention = retention.flatten(1)
        retention = self.out_proj(retention)
        return retention, states


    def chunkwise(self, x, mask, gamma, state_prev=None, act='softmax'):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        retention = q @ k.transpose(-2, -1)
        retention = retention * mask
        retention = (retention @ v).transpose(1, 2).reshape(B, N, C)
        inner_pos = (torch.arange(k.size(2), device=k.device, dtype=k.dtype) + 1).reshape(1, 1, -1, 1)
        gamma = gamma.reshape(1, -1, 1, 1)
        states = k.unsqueeze(-2).transpose(-1, -2) @ v.unsqueeze(-2)
        state_decays = gamma ** (k.size(2) - inner_pos)
        state = (states * state_decays.unsqueeze(-1)).sum(dim=2)

        if state_prev is not None:
            chunk_decay = gamma ** k.size(2)
            state = state + state_prev * chunk_decay
            cross_retention = (q @ state_prev) * (gamma**inner_pos)
            retention = retention + cross_retention.transpose(1, 2).reshape(B, N, C)

        retention = self.out_proj(retention)
        return retention, state


    def forward(self, x, mask=None, gamma=None, state = None, mode='parallel'):
        if mode == 'parallel':
            x, _ = self.parallel(x, mask=mask)
        elif mode == 'recurrent':
            x, state = self.recurrent(x, gamma=gamma, state_prev=state)
        elif mode == 'chunkwise':
            x, state = self.chunkwise(x, mask=mask, gamma=gamma, state_prev=state)
        return x, state


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.retention = Retention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, state_prev=None, mode='parallel', mask=None, gamma=None):
        x_r, state = self.retention(self.norm1(x), mask=mask, gamma=gamma, state = state_prev, mode=mode)
        x = x + self.drop_path1(self.ls1(x_r))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, state


class ResPostBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.init_values = init_values

        self.retention = Retention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.init_weights()

    def init_weights(self):
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x):
        x, mask = x
        x = x + self.drop_path1(self.norm1(self.retention(x, mask)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x, mask
    

class DecayMask(nn.Module):
    def __init__(self,
                 num_heads=12,
                 ):
        super().__init__()
        self.decay_gammas = 1 - torch.exp(torch.linspace(math.log(1 / 32), math.log(1 / 512),
                                                         steps=num_heads,device='cuda'))
        self.decay_gammas_mask = self.decay_gammas.unsqueeze(1).unsqueeze(2)
        
    def forward(self, N, dtype):
        token_index = torch.arange(N, device='cuda', dtype=dtype)
        token_d = torch.abs(token_index.unsqueeze(-1) - token_index.unsqueeze(0))
        mask = torch.ones_like(token_d, dtype=torch.bool).triu_(diagonal=1)
        mask = self.decay_gammas_mask**token_d.masked_fill(mask, float("inf")).unsqueeze(0)
        return mask, self.decay_gammas


class VisionRetention(nn.Module):
    """ Vision Retention Networks

    A PyTorch impl of : `Vision Retention Networks`
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'token_last',
            encode_mode: str = 'autoregressive',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = 1e-5,
            class_token: bool = True,
            no_embed_class: bool = True,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: str = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[Callable] = None,
            act_layer: Optional[Callable] = None,
            block_fn: Callable = Block,
            mlp_layer: Callable = Mlp,
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token', 'token_last', 'final')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False
        self.chunkwise_recurrent = False
        self.num_heads = num_heads
        self.encode_mode = encode_mode
        if self.encode_mode == 'bidirectional':
            self.num_tokens = ((img_size[0] // 16) ** 2) * 2
        elif self.encode_mode == 'autoregressive':
            self.num_tokens = ((img_size[0] // 16) ** 2)
        if class_token:
            self.num_tokens += 1


        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,
        )
        self.decay_mask = DecayMask(num_heads=num_heads)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)


    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        #trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}


    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token', 'token_last', 'final')
            self.global_pool = global_pool


    def _pos_embed(self, x):
        if self.no_embed_class:
            x = x + self.pos_embed
            if self.cls_token is not None:
                if self.global_pool == 'token':
                    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
                elif self.global_pool == 'token_last':
                    x = torch.cat((x, self.cls_token.expand(x.shape[0], -1, -1)), dim=1)
        else:
            if self.cls_token is not None:
                if self.global_pool == 'token':
                    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
                elif self.global_pool == 'token_last':
                    x = torch.cat((x, self.cls_token.expand(x.shape[0], -1, -1)), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)


    def forward_head(self, x):
        if self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool == 'final' or self.global_pool == 'token_last':
            x = x[:, -1]
        elif self.global_pool == 'token':
            x = x[:, 0]
        else:
            raise NotImplementedError(f'Pool method {self.global_pool} not currently implemented!')
        x = self.fc_norm(x)
        x = self.head(x)

        return x


    def forward_parallel(self, x, mask, gamma, mode = 'chunkwise'):
        for blk, state_prev in zip(self.blocks, [None] * len(self.blocks)):
            x, _ = blk(x, state_prev, 'parallel', mask, gamma)
        x = self.norm(x)
        x = self.forward_head(x)
        return x


    def forward_retention(self, x, mask, gamma, chunk_size = 16, mode = 'chunkwise'):
        states_prev = []
        outputs = []
        if mode == 'recurrent':
            for idx in range(self.num_tokens):
                states = []
                if len(states_prev) == 0:
                    states_prev = [None] * len(self.blocks)
                x_r = x[:, idx, :]
                for blk, state_prev in zip(self.blocks, states_prev):
                    x_r, state = blk(x_r, state_prev, mode, mask, gamma)
                    states.append(state)
                x_r = self.norm(x_r)
                outputs.append(x_r)
                states_prev = states
            x = torch.stack(outputs, dim=1)
        elif mode == 'chunkwise':
            for idx in range(0, self.num_tokens, chunk_size):
                states = []
                if len(states_prev) == 0:
                    states_prev = [None] * len(self.blocks)
                x_r = x[:, idx : idx + chunk_size, :]
                mask, gamma = self.decay_mask(x_r.shape[1], x.dtype)
                for blk, state_prev in zip(self.blocks, states_prev):
                    x_r, state = blk(x_r, state_prev, mode, mask, gamma)
                    states.append(state)
                x_r = self.norm(x_r)
                outputs.append(x_r)
                states_prev = states
            x = torch.cat(outputs, dim=1)
        return x


    def forward_features(self, x, chunk_size = 16, mode = 'parallel'):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.encode_mode == 'bidirectional':
            x = torch.cat([x,torch.flip(x,dims=[1])], dim=1)

        if mode == 'parallel':
            mask, gamma = self.decay_mask(x.shape[1], x.dtype)
            x = self.forward_parallel(x, mask, gamma)
        elif mode == 'recurrent':
            mask, gamma = self.decay_mask(x.shape[1], x.dtype)
            x = self.forward_retention(x, mask, gamma, chunk_size = chunk_size, mode='recurrent')
        elif mode == 'chunkwise':
            mask, gamma = self.decay_mask(x.shape[1], x.dtype)
            x = self.forward_retention(x, mask=None, gamma=None, chunk_size = chunk_size, mode='chunkwise')
        return x

    def forward(self, x, chunk_size = 16, mode = 'parallel'):
         x = self.forward_features(x, chunk_size = chunk_size, mode = mode)
         return x


def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()
    
def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()



# def get_init_weights_vit(head_bias: float = 0.):
#     return partial(init_weights_vir, head_bias=head_bias)

def get_init_weights_vit(mode='jax', head_bias: float = 0.):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


def resize_pos_embed(
        posemb,
        posemb_new,
        num_prefix_tokens=1,
        gs_new=(),
        interpolation='bicubic',
        antialias=False,
):
    """ Rescale the grid of position embeddings when loading from state_dict.
    """
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
        ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info(f'Resized position embedding: {posemb.shape} ({[gs_old, gs_old]}) to {posemb_new.shape} ({gs_new}).')
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=interpolation, antialias=antialias, align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


def checkpoint_filter_fn(
        state_dict,
        model,
        adapt_layer_scale=False,
        interpolation='bicubic',
        antialias=True,
):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    import re
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            O, I, H, W = model.patch_embed.proj.weight.shape
            if len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = model.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        elif adapt_layer_scale and 'gamma_' in k:
            k = re.sub(r'gamma_([0-9])', r'ls\1.gamma', k)
        elif 'pre_logits' in k:
            continue
        out_dict[k] = v
    return out_dict


def _create_vision_retention(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        VisionRetention,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs,
    )


@register_model
def vir_small_patch16_224(pretrained=False, **kwargs) -> VisionRetention:
    """ ViR-Small (ViR-S/16)
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, init_values=1e-5)
    model = _create_vision_retention('vit_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vir_base_patch16_224(pretrained=False, **kwargs) -> VisionRetention:
    """ ViR-Base (ViR-B/32)
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, init_values=1e-5)
    model = _create_vision_retention('vit_base_patch16_224.augreg_in1k', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vir_large_patch16_224(pretrained=False, **kwargs) -> VisionRetention:
    """ ViR-Large model (ViR-L/16)
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, init_values=1e-5)
    model = _create_vision_retention('vit_large_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vir_large_patch14_224(pretrained=False, **kwargs) -> VisionRetention:
    """ ViR-Large model (ViR-L/14)
    """
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, init_values=1e-5)
    model = _create_vision_retention('vit_large_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model



if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Starting inference")
    model = vir_base_patch16_224().cuda()
    model.eval()
    img = torch.randn(1, 3, 224, 224).cuda()
    with torch.no_grad():
        out_parallel = model(img, mode='parallel')
    print("Finished inference")
