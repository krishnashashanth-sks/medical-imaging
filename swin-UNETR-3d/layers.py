import torch
import torch.nn as nn
from utils import window_partition3d,window_reverse3d
import torch.nn.functional as F

class Mlp(nn.Module):
  def __init__(self,in_features,hidden_features=None,out_features=None,act_layer=nn.GELU,drop=0.):
    super().__init__()
    out_features=out_features or in_features
    hidden_features=hidden_features or in_features
    self.fc1=nn.Linear(in_features,hidden_features)
    self.act=act_layer()
    self.fc2=nn.Linear(hidden_features,out_features)
    self.drop=nn.Dropout(drop)
  def forward(self,x):
    return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class WindowAttention3D(nn.Module):
  def __init__(self,dim,window_size,num_heads,qkv_bias=True,qk_scale=None,attn_drop=0.,proj_drop=0.):
    super().__init__()
    self.dim=dim
    self.window_size=window_size
    self.num_heads=num_heads
    head_dim=dim//num_heads
    self.qk_scale=qk_scale or head_dim**-0.5
    self.qkv=nn.Linear(dim,dim*3,bias=qkv_bias)
    self.attn_drop=nn.Dropout(attn_drop)
    self.proj=nn.Linear(dim,dim)
    self.proj_drop=nn.Dropout(proj_drop)
    self.softmax=nn.Softmax(dim=-1)
  def forward(self,x,mask=None):
    B_,N,C=x.shape
    qkv=self.qkv(x).reshape(B_,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
    q,k,v=qkv[0],qkv[1],qkv[2]
    q=q*self.qk_scale
    attn=(q@k.transpose(-2,-1))
    if mask is not None:
      nw=mask.shape[0]
      # attn_view becomes (actual_batch_size, num_windows_per_image, num_heads, N, N)
      attn = attn.view(B_ // nw, nw, self.num_heads, N, N)
      # mask_expanded becomes (1, num_windows_per_image, 1, N, N)
      mask_expanded = mask.unsqueeze(1).unsqueeze(0)
      # The addition broadcasts to (actual_batch_size, num_windows_per_image, num_heads, N, N)
      attn = attn + mask_expanded
      # Flatten attn back to (B_, num_heads, N, N) for matrix multiplication with v
      attn = attn.view(B_, self.num_heads, N, N)
      attn=self.softmax(attn)
    else:
      attn=self.softmax(attn)
    attn=self.attn_drop(attn)
    x=(attn @ v).transpose(1,2).reshape(B_,N,C)
    x=self.proj(x)
    x=self.proj_drop(x)
    return x

class PatchEmbed3D(nn.Module):
  def __init__(self,img_size=(64,64,64),patch_size=2,in_chans=1,embed_dim=96,norm_layer=None):
    super().__init__()
    patch_size_tuple=(patch_size,patch_size,patch_size) if isinstance(patch_size,int) else patch_size
    self.img_size=img_size
    self.patch_size=patch_size_tuple
    self.embed_dim=embed_dim
    self.num_patches=(
        (img_size[0]//patch_size_tuple[0])*
        (img_size[1]//patch_size_tuple[1])*
        (img_size[2]//patch_size_tuple[2])
    )
    self.proj=nn.Conv3d(in_chans,embed_dim,kernel_size=patch_size_tuple,stride=patch_size_tuple)
    if norm_layer:
      self.norm=norm_layer(embed_dim)
    else:
      self.norm=None
  def forward(self,x):
    B,C,D,H,W=x.shape
    assert D==self.img_size[0]and H==self.img_size[1] and W==self.img_size[2],\
    f"Input volume size ({D}*{H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
    x=self.proj(x).flatten(2).transpose(1,2)
    if self.norm:
      x=self.norm(x)
    return x
  
class PatchMerging3D(nn.Module):
  def __init__(self,input_resolution,dim,norm_layer=nn.LayerNorm):
    super().__init__()
    self.input_resolution=input_resolution
    self.dim=dim
    self.reduction=nn.Linear(8*dim,2*dim,bias=False)
    self.norm=norm_layer(8*dim)
  def forward(self,x):
    D,H,W=self.input_resolution
    B,L,C=x.shape
    assert L==D*H*W,"input feature has wrong size"
    assert D%2==0 and H%2==0 and W%2==0,f"x size({D}*{H}*{W}) are not even"
    x=x.view(B,D,H,W,C)
    x0=x[:,0::2,0::2,0::2,:]
    x1=x[:,1::2,0::2,0::2,:]
    x2=x[:,0::2,1::2,0::2,:]
    x3=x[:,0::2,0::2,1::2,:]
    x4=x[:,1::2, 1::2,0::2,:]
    x5=x[:,1::2,0::2,1::2,:]
    x6=x[:,0::2,1::2,1::2,:]
    x7=x[:,1::2,1::2,1::2,:]
    x=torch.cat([x0,x1,x2,x3,x4,x5,x6,x7],-1)
    x=x.view(B,-1,8*C)
    x=self.norm(x)
    x=self.reduction(x)
    return x
  
class PatchExpanding3D(nn.Module):
  def __init__(self,input_resolution,dim,norm_layer=nn.LayerNorm):
    super().__init__()
    self.input_resolution=input_resolution
    self.dim=dim
    # To output dim // 2 channels after 2x2x2 upsampling (factor of 8)
    # the linear layer should expand to 4*dim, then C//8 will be (4*dim)//8 = dim//2
    self.expand=nn.Linear(dim,4*dim,bias=False) # Changed from 2*dim to 4*dim
    self.norm=norm_layer(dim//2) # Corrected: norm_layer should operate on the final output dim after effective channel reduction
  def forward(self,x):
    D,H,W=self.input_resolution
    x=self.expand(x) # C becomes 4*dim
    B,L,C=x.shape
    assert L==D*H*W,"input feature has wrong size"
    # Reshape for 2x2x2 spatial expansion, C//8 is (4*dim)//8 = dim//2
    x=x.view(B,D,H,W,C//8,2,2,2)
    # Permute to move channel dimension next to batch, then view to expand spatial dims
    x=x.permute(0,4,1,5,2,6,3,7).contiguous().view(B,C//8,D*2,H*2,W*2)
    x=x.flatten(2).transpose(1,2) # Flatten to (B, L_new, C_new)
    x=self.norm(x)
    return x
  
class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block for 3D volumes.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution (D, H, W).
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size (window_d, window_h, window_w).
        shift_size (tuple[int]): Shift size for SW-MSA (shift_d, shift_h, shift_w).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if tuple(self.input_resolution) != tuple(self.window_size):
             # Make sure shift_size is smaller than window_size
            assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
            assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
            assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"


        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path) # Changed from DropPath
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, out_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if any(i > 0 for i in self.shift_size):
            D, H, W = self.input_resolution
            wd, wh, ww = self.window_size
            sd, sh, sw = self.shift_size

            # Calculate padding needed to ensure input_resolution is divisible by window_size for mask generation
            pad_d = (wd - D % wd) % wd
            pad_h = (wh - H % wh) % wh
            pad_w = (ww - W % ww) % ww

            # Padded dimensions for the mask
            Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w

            # Create a base mask with incremental values to identify shifted regions
            img_mask = torch.zeros((1, Dp, Hp, Wp, 1))
            cnt = 0

            # Define slices for the regions that will be unshifted and shifted
            d_slices = (slice(0, -wd), slice(-wd, -sd), slice(-sd, Dp))
            h_slices = (slice(0, -wh), slice(-wh, -sh), slice(-sh, Hp))
            w_slices = (slice(0, -ww), slice(-ww, -sw), slice(-sw, Wp))

            for d_slice in d_slices:
                for h_slice in h_slices:
                    for w_slice in w_slices:
                        img_mask[:, d_slice, h_slice, w_slice, :] = cnt
                        cnt += 1

            # Perform cyclic shift on the mask
            shifted_img_mask = torch.roll(img_mask, shifts=(-sd, -sh, -sw), dims=(1, 2, 3))

            # Partition the shifted mask into windows
            mask_windows = window_partition3d(shifted_img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, wd * wh * ww) # Flatten windows

            # Compute attention mask based on window differences
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        D, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == D * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        # pad feature maps to multiples of window size
        pad_d = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        _, Dp, Hp, Wp, _ = x.shape

        # cyclic shift
        if any(i > 0 for i in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition3d(shifted_x, self.window_size)  # nW*B, wd, wh, ww, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # nW*B, N', C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, N', C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse3d(attn_windows, self.window_size, Dp, Hp, Wp)  # B D' H' W' C

        # reverse cyclic shift
        if any(i > 0 for i in self.shift_size):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        x = x.view(B, D * H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    
class BasicLayer_up(nn.Module):
  """ A basic Swin Transformer Layer for one stage in the decoder of SwinUNETR.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (list[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): Patch expanding layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
  def __init__(self,dim,input_resolution,depth,num_heads,window_size,
                mlp_ratio=4.,qkv_bias=True,qk_scale=None,drop=0.,attn_drop=0.,
                drop_path=0.,norm_layer=nn.LayerNorm,upsample=None,use_checkpoint=False):
    super().__init__()
    self.dim=dim
    self.input_resolution=input_resolution
    self.depth=depth
    self.use_checkpoint=use_checkpoint
    self.blocks=nn.ModuleList([
        SwinTransformerBlock3D(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=(0,0,0) if (i%2==0) else tuple(w//2 for w in window_size),
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path[i] if isinstance(drop_path,list)else drop_path,
            norm_layer=norm_layer)
        for i in range(depth)
    ])
    if upsample is not None:
      self.upsample=PatchExpanding3D(input_resolution,dim=dim)
    else:
      self.upsample=None
  def forward(self,x):
    for blk in self.blocks:
      x=blk(x)
    if self.upsample is not None:
      x=self.upsample(x)
    return x

class SwinTransformerStage3D(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (list[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Patch merging layer. Default: None.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else tuple(w // 2 for w in window_size),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = PatchMerging3D(input_resolution, dim=dim)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x