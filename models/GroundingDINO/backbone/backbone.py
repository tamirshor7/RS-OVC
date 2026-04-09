# ------------------------------------------------------------------------
# Adapted from "https://github.com/niki-amini-naieni/CountGD"
# ------------------------------------------------------------------------

"""
Backbone modules.
"""

from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from util.misc import NestedTensor, is_main_process
from transformers import AutoImageProcessor, AutoModel
from models.GroundingDINO.backbone.position_encoding import build_position_encoding
from models.GroundingDINO.backbone.swin_transformer import build_swin_transformer



class DINOv3Backbone(nn.Module):
    """
    DINOv3 backbone wrapper compatible with Joiner.
    """
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.patch_size = self.model.config.patch_size
        self.hidden_dim = self.model.config.hidden_size
        self.num_channels = [self.hidden_dim] # Single scale output

        # Register denormalization buffers (ImageNet stats used in transforms)
        self.register_buffer("swin_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("swin_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, tensor_list: NestedTensor):
        images = tensor_list.tensors
        device = images.device
        B = images.shape[0]

        # 1. Denormalize (Swin -> Raw)
        images_denorm = images * self.swin_std + self.swin_mean
        
        # 2. Process for DINOv3
        target_size = {"shortest_edge": 224}
        if hasattr(self.processor, "size"):
             target_size = self.processor.size

        inputs = self.processor(
            images=images_denorm, 
            return_tensors="pt",
            do_resize=True,
            size={'height':512,'width':512},#target_size,
            do_rescale=False, 
            do_normalize=True
        ).to(device)
        
        
        outputs = self.model(**inputs)
        
        # 4. Reshape Output
        sequence_output = outputs.last_hidden_state
        
        # Remove CLS + Registers (assuming first tokens are special)
        # We calculate num patches based on input size to be safe
        H_in, W_in = inputs['pixel_values'].shape[-2:]
        H_feat = H_in // self.patch_size
        W_feat = W_in // self.patch_size
        num_patches = H_feat * W_feat
        
        L_total = sequence_output.shape[1]
        num_special = L_total - num_patches
        
        if num_special > 0:
             sequence_output = sequence_output[:, num_special:, :]
             
        # [B, L, D] -> [B, D, H, W]
        feature_4d = sequence_output.view(B, H_feat, W_feat, self.hidden_dim).permute(0, 3, 1, 2)
        
        # 5. Handle Mask
        m = tensor_list.mask
        if m is not None:
            m_float = m.float().unsqueeze(1)
            mask_feat = F.interpolate(m_float, size=(H_feat, W_feat), mode='nearest').to(torch.bool).squeeze(1)
        else:
            mask_feat = torch.zeros((B, H_feat, W_feat), dtype=torch.bool, device=device)

        # Return Dict expected by Joiner. Key "0" for the single feature map.
        # We use string keys like "0" or "layer0" to match BackboneBase convention if needed, 
        # but typically Joiner expects a dict.
        return {"0": NestedTensor(feature_4d, mask_feat)}

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_indices: list,
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.requires_grad_(False)

        return_layers = {}
        for idx, layer_index in enumerate(return_interm_indices):
            return_layers.update(
                {"layer{}".format(5 - len(return_interm_indices) + idx): "{}".format(layer_index)}
            )

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        dilation: bool,
        return_interm_indices: list,
        batch_norm=FrozenBatchNorm2d,
    ):
        if name in ["resnet18", "resnet34", "resnet50", "resnet101"]:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(),
                norm_layer=batch_norm,
            )
        else:
            raise NotImplementedError("Why you can get here with name {}".format(name))
        # num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        assert name not in ("resnet18", "resnet34"), "Only resnet50 and resnet101 are available."
        assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]
        num_channels_all = [256, 512, 1024, 2048]
        num_channels = num_channels_all[4 - len(return_interm_indices) :]
        super().__init__(backbone, train_backbone, num_channels, return_interm_indices)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    """
    Useful args:
        - backbone: backbone name
        - lr_backbone:
        - dilation
        - return_interm_indices: available: [0,1,2,3], [1,2,3], [3]
        - backbone_freeze_keywords:
        - use_checkpoint: for swin only for now

    """
    
    if args.backbone not in ['rsfm']:
        position_embedding = build_position_encoding(args)
    train_backbone = True
    if not train_backbone:
        raise ValueError("Please set lr_backbone > 0")
    return_interm_indices = args.return_interm_indices
    assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]
    args.backbone_freeze_keywords
    use_checkpoint = getattr(args, "use_checkpoint", False)

    if args.backbone in ["resnet50", "resnet101"]:
        backbone = Backbone(
            args.backbone,
            train_backbone,
            args.dilation,
            return_interm_indices,
            batch_norm=FrozenBatchNorm2d,
        )
        bb_num_channels = backbone.num_channels
    elif args.backbone in [
        "swin_T_224_1k",
        "swin_B_224_22k",
        "swin_B_384_22k",
        "swin_L_224_22k",
        "swin_L_384_22k",
    ]:
        pretrain_img_size = int(args.backbone.split("_")[-2])
        backbone = build_swin_transformer(
            args.backbone,
            pretrain_img_size=pretrain_img_size,
            out_indices=tuple(return_interm_indices),
            dilation=False,
            use_checkpoint=use_checkpoint,
        )

        bb_num_channels = backbone.num_features[4 - len(return_interm_indices) :]
    
    
    elif args.backbone == 'dinov3':
        pretrained_model_name = "facebook/dinov3-vitl16-pretrain-sat493m"
        backbone = DINOv3Backbone(pretrained_model_name)
        bb_num_channels = backbone.num_channels # [1024]
    else:
        raise NotImplementedError("Unknown backbone {}".format(args.backbone))

    if args.backbone not in ['dinov3']:
        assert len(bb_num_channels) == len(
            return_interm_indices
        ), f"len(bb_num_channels) {len(bb_num_channels)} != len(return_interm_indices) {len(return_interm_indices)}"
        assert isinstance(
        bb_num_channels, List
        ), "bb_num_channels is expected to be a List but {}".format(type(bb_num_channels))
    else:
        bb_num_channels = [1024]
    model = Joiner(backbone, position_embedding)
    model.num_channels = bb_num_channels
    
  
    return model
