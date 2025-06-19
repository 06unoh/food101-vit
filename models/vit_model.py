import torch
from torch import nn

class PatchEmbedding(nn.Module):
  def __init__(self,img_size=224,patch_size=16,in_channels=3,embed_dim=768):
    super().__init__()
    self.img_size=img_size
    self.num_patches=(img_size//patch_size)**2

    self.proj=nn.Conv2d(in_channels, embed_dim,kernel_size=patch_size, stride=patch_size)

  def forward(self,x):
    x=self.proj(x)
    x=x.flatten(2)
    x=x.transpose(1,2)
    return x

class EncoderBlock(nn.Module):
  def __init__(self,embed_dim=768,num_heads=8,mlp_dim=768*4,dropout=0.2):
    super().__init__()
    self.norm1=nn.LayerNorm(embed_dim)
    self.attn=nn.MultiheadAttention(embed_dim,num_heads,dropout,batch_first=True)
    self.norm2=nn.LayerNorm(embed_dim)
    self.mlp=nn.Sequential(
        nn.Linear(embed_dim,mlp_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_dim,embed_dim),
        nn.Dropout(dropout),
    )

  def forward(self,x):
    x_res=x
    x=self.norm1(x)
    x,_=self.attn(x,x,x)
    x=x_res+x

    x_res=x
    x=self.norm2(x)
    x=self.mlp(x)
    x=x_res+x
    return x

class VisionTransformer(nn.Module):
  def __init__(self,img_size=224,patch_size=16,in_channels=3,embed_dim=768,mlp_dim=768*4,depth=12,dropout=0.2,num_classes=101,num_heads=8):
    super().__init__()
    self.norm=nn.LayerNorm(embed_dim)
    self.patch_embed=PatchEmbedding(img_size,patch_size,in_channels,embed_dim)
    self.cls_tokens=nn.Parameter(torch.zeros((1,1,embed_dim)))
    self.pos_emb=nn.Parameter(torch.zeros((1,self.patch_embed.num_patches+1,embed_dim)))
    self.dropout=nn.Dropout(dropout
                            )
    self.enc_block=nn.Sequential(*[
        EncoderBlock(embed_dim, num_heads,mlp_dim,dropout)
        for _ in range(depth)
    ])

    self.head=nn.Linear(embed_dim,num_classes)

    nn.init.trunc_normal_(self.cls_tokens,std=0.02)
    nn.init.trunc_normal_(self.pos_emb,std=0.02)
    self.apply(self._init_weight)

  def _init_weight(self,m):
    if isinstance(m, nn.Linear):
      nn.init.trunc_normal_(m.weight, std=0.02)
      if m.bias is not None:
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
      nn.init.zeros_(m.bias)
      nn.init.ones_(m.weight)


  def forward(self, x):
    x = self.patch_embed(x)  
    B=x.size(0)
    cls_tokens=self.cls_tokens.expand(B,-1,-1)
    x=torch.cat((cls_tokens,x),dim=1)
    x=x+self.pos_emb
    x=self.dropout(x)
    x=self.enc_block(x)
    x=self.norm(x)
    cls_output=x[:,0]
    logits=self.head(cls_output)
    return logits
