import torch.nn as nn
from layers import PatchEmbed3D,SwinTransformerStage3D,PatchMerging3D,SwinTransformerBlock3D,BasicLayer_up,PatchExpanding3D
import torch

class SwinUNETR(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, feature_size=24,
                 depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24), window_size=(7, 7, 7),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = feature_size
        self.num_features = int(feature_size * 2**(self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.img_size = img_size # <--- Added this line

        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=2, # Hardcoded patch size of 2 for SwinUNETR
            in_chans=in_channels,
            embed_dim=feature_size,
            norm_layer=norm_layer)

        num_patches = self.patch_embed.num_patches
        patches_resolution = (
            img_size[0] // self.patch_embed.patch_size[0],
            img_size[1] // self.patch_embed.patch_size[1],
            img_size[2] // self.patch_embed.patch_size[2])
        self.patches_resolution = patches_resolution

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # Encoder
        self.encoder_stages = nn.ModuleList()
        for i_layer in range(self.num_layers):
            stage = SwinTransformerStage3D(
                dim=int(feature_size * 2**i_layer),
                input_resolution=(patches_resolution[0] // (2**i_layer),
                                  patches_resolution[1] // (2**i_layer),
                                  patches_resolution[2] // (2**i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging3D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.encoder_stages.append(stage)

        # Bottleneck (deepest layer)
        self.bottleneck = SwinTransformerBlock3D(
            dim=int(feature_size * 2**(self.num_layers - 1)),
            input_resolution=(patches_resolution[0] // (2**(self.num_layers - 1)),
                              patches_resolution[1] // (2**(self.num_layers - 1)),
                              patches_resolution[2] // (2**(self.num_layers - 1))),
            num_heads=num_heads[-1],
            window_size=window_size,
            shift_size=(0, 0, 0),
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[-1], # Use last drop_path rate
            norm_layer=norm_layer)

        # Decoder
        self.decoder_stages = nn.ModuleList()
        for i_layer in range(self.num_layers - 2, -1, -1): # Iterate from second to last encoder stage backwards
            stage = BasicLayer_up(
                dim=int(feature_size * 2**(i_layer + 1)), # Input to decoder is 2x previous encoder stage
                input_resolution=(patches_resolution[0] // (2**(i_layer + 1)),
                                  patches_resolution[1] // (2**(i_layer + 1)),
                                  patches_resolution[2] // (2**(i_layer + 1))),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=True)
            self.decoder_stages.append(stage)

        # Final layer to bring to original resolution and desired channels
        self.final_expanding = PatchExpanding3D(
            input_resolution=(patches_resolution[0],
                              patches_resolution[1],
                              patches_resolution[2]),
            dim=feature_size) # Corrected dim to feature_size to match input channels

        self.output_conv = nn.Conv3d(feature_size // 2, out_channels, kernel_size=1) # Corrected input channels to feature_size // 2


    def forward(self, x):
        x = self.patch_embed(x)

        # Encoder path
        encoder_features = []
        for stage in self.encoder_stages:
            x = stage(x)
            encoder_features.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        # Connect with skip connections from encoder
        for i, stage in enumerate(self.decoder_stages):
            x = encoder_features[self.num_layers - 2 - i] + x # Skip connection
            x = stage(x)

        # Final expanding layer
        x = self.final_expanding(x)
        B, L, C = x.shape
        D, H, W = self.img_size[0], self.img_size[1], self.img_size[2]
        x = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3) # B D H W C -> B C D H W
        x = self.output_conv(x)

        return x