#transform timm vits to versions that can stop feeding forward specific patches

from types import MethodType
import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import Attention,Block

def vit_attention_forward(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    # attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    del q, k

    # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B, N, C)
    del v
    x = self.proj(x)
    x = self.proj_drop(x)
    return x,attn

def vit_block_forward(self,x):
    x_out,attn_out=self.attn(self.norm1(x))
    x = x + (x_out)
    x = x + (self.mlp(self.norm2(x)))
    return x,attn_out

def vit_forward_features(self, x,current_abs_index,next_relative_index):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        # sparse
        if next_relative_index.shape[1]==current_abs_index.shape[1]:
            pass
        else:
            current_abs_index=torch.gather(current_abs_index,1,next_relative_index)
            assert current_abs_index[0][0]==0

        x=torch.gather(x,1,current_abs_index.unsqueeze(-1).expand(-1,-1,x.size(-1)))

        x = self.pos_drop(x)
        attn_weights = []
        for blk in self.blocks:
            x, attn = blk(x)
            attn_weights.append(attn)
        x = self.norm(x)[:, 0]

        return x,attn_weights,current_abs_index

def vit_forward(self,x,current_abs_index,next_relative_index):
    x,attn_out,current_abs_index = self.forward_features(x,current_abs_index,next_relative_index)
    x = self.head(x)
    return x,attn_out,current_abs_index





class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B


def build_model(name, Pretrained=True):
    """
    Get a vision transformer model.

    This will insert
    current_abs_index (the absolute index of current patches)
    and next_relative_index  (the relative index of patches to retain)
    to the original input of attention.forward, block.forward/forward_feature, and net.forward

    Currently support almost all quantization in timm.quantization.transformers, including:
    - vit_tiny/small/base/large_patch16/patch32_224/384,
    - deit_tiny/small/base(_distilled)_patch16_224,
    """
    net = timm.create_model(name,pretrained=Pretrained)

    for name, module in net.named_modules():
        if isinstance(module, Attention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(vit_attention_forward, module)
        if isinstance(module,Block):
            module.forward = MethodType(vit_block_forward, module)
            net.forward_features = MethodType(vit_forward_features, net)
            net.forward = MethodType(vit_forward, net)

    net = net.cuda()
    net.eval()
    return net
