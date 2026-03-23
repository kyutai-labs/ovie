# adapted from (https://github.com/facebookresearch/DiT & https://github.com/willisma/SiT & https://github.com/bytetriper/RAE

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import PatchEmbed, Mlp
from models.swiglu_ffn import SwiGLUFFN
from models.pos_embed import VisionRotaryEmbeddingFast
from models.rmsnorm import RMSNorm


def modulate(x, shift, scale):
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    """
    Attention module of LightningDiT.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = True,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        if use_rmsnorm:
            norm_layer = RMSNorm

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            q = rope(q)
            k = rope(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CamEmbedder(nn.Module):
    """
    Embeds camera parameters into vector representations.
    Same as DiT.
    """

    def __init__(self, in_cam_params, hidden_size):
        super().__init__()
        self.linear = nn.Linear(in_cam_params, hidden_size, bias=True)

    def forward(self, cam_params: torch.Tensor) -> torch.Tensor:
        cam_emb = self.linear(cam_params)
        return cam_emb


class ConditionalViTBlock(nn.Module):
    """
    Conditional Vision Transformer Block. We add features including:
    - ROPE
    - QKNorm
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False,
        use_rmsnorm=False,
        wo_shift=False,
        **block_kwargs,
    ):
        super().__init__()

        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)

        # Initialize attention layer
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs,
        )

        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )

        # Initialize AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 4 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift

    def forward(self, x, c, feat_rope=None):
        if c is None:
            # for unconditional ViT
            shift_msa = None
            scale_msa = None
            gate_msa = 1.0
            shift_mlp = None
            scale_mlp = None
            gate_mlp = 1.0
        else:
            if self.wo_shift:
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(
                    c
                ).chunk(4, dim=1)
                shift_msa = None
                shift_mlp = None
            else:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.adaLN_modulation(c).chunk(6, dim=1)
                )

        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of Conditional ViT.
    """

    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6
            )
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class ConditionalViT(nn.Module):
    """
    Conditional Vision Transformer backbone.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False,
        use_rope=False,
        use_rmsnorm=False,
        wo_shift=False,
        use_checkpoint=False,
        in_cam_params=7,
        out_channels=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.depth = depth
        self.hidden_size = hidden_size
        self.use_checkpoint = use_checkpoint
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.cam_embedder = CamEmbedder(in_cam_params, hidden_size)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        # use rotary position encoding, borrow from EVA
        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.feat_rope = None

        self.blocks = nn.ModuleList(
            [
                ConditionalViTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_swiglu=use_swiglu,
                    use_rmsnorm=use_rmsnorm,
                    wo_shift=wo_shift,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(
            hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize cam_embedder:
        nn.init.xavier_uniform_(self.cam_embedder.linear.weight)
        nn.init.constant_(self.cam_embedder.linear.bias, 0)

        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, cam_params):
        """
        Forward pass of LightningDiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        cam_params: (N, in_cam_params) tensor of camera parameters
        use_checkpoint: boolean to toggle checkpointing
        """

        use_checkpoint = self.use_checkpoint

        x = self.x_embedder(x)  # (N, T, D), where T = H * W / patch_size ** 2
        if not self.use_rope:
            x = x + self.pos_embed
        c = self.cam_embedder(cam_params)  # (N, D)

        for block in self.blocks:
            if use_checkpoint:
                x = checkpoint(block, x, c, self.feat_rope, use_reentrant=True)
            else:
                x = block(x, c, self.feat_rope)

        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=32):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(
                num_groups=min(num_groups, out_ch),
                num_channels=out_ch,
                eps=1e-6,
                affine=True,
            ),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch, num_res_blocks=2):
        super().__init__()
        layers = [ConvBlock(in_ch, out_ch)]
        for _ in range(num_res_blocks - 1):
            layers.append(ConvBlock(out_ch, out_ch))

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x, fetch_prepool=False):
        x = self.conv(x)
        x_pre_pool = x
        x = self.pool(x)
        if fetch_prepool:
            return x, x_pre_pool
        return x


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch, num_res_blocks=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        layers = [ConvBlock(out_ch, out_ch)]
        for _ in range(num_res_blocks - 1):
            layers.append(ConvBlock(out_ch, out_ch))

        self.conv = nn.Sequential(*layers)

    def forward(self, x, noise_scaler=None):
        x = self.up(x)
        if noise_scaler is not None:
            noise = torch.randn(
                x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype
            )
            x = x + noise * noise_scaler
        x = self.conv(x)
        return x


class ConditionalUViT(nn.Module):
    """
    U-Net without skip-connections using a ViT bottleneck.
    """

    def __init__(
        self,
        image_size,
        in_channels=3,
        out_channels=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4, 4],
        num_res_blocks=2,
        vit_hidden_size=512,
        vit_depth=12,
        vit_patch_size=1,
        vit_num_heads=8,
        vit_mlp_ratio=4.0,
        vit_use_qknorm=False,
        vit_use_swiglu=False,
        vit_use_rope=False,
        vit_use_rmsnorm=False,
        vit_wo_shift=False,
        vit_use_checkpoint=False,
        in_cam_params=7,
        final_sigmoid_activation=True,
        inject_noise_in_decoder=False,
    ):
        super().__init__()

        self.num_down = len(ch_mult)
        vit_input_size = image_size // (2**self.num_down)
        assert vit_input_size > 0

        channels = [ch * m for m in ch_mult]

        # -------- Encoder --------
        self.downs = nn.ModuleList()
        curr = in_channels
        for out_ch in channels:
            self.downs.append(Downsample(curr, out_ch, num_res_blocks))
            curr = out_ch

        bottleneck_ch = channels[-1]

        # -------- ViT bottleneck --------
        self.vit = ConditionalViT(
            input_size=vit_input_size,
            in_channels=bottleneck_ch,
            hidden_size=vit_hidden_size,
            depth=vit_depth,
            patch_size=vit_patch_size,
            num_heads=vit_num_heads,
            mlp_ratio=vit_mlp_ratio,
            use_qknorm=vit_use_qknorm,
            use_swiglu=vit_use_swiglu,
            use_rope=vit_use_rope,
            use_rmsnorm=vit_use_rmsnorm,
            wo_shift=vit_wo_shift,
            use_checkpoint=vit_use_checkpoint,
        )

        # -------- Decoder --------
        upsample_channels = []
        self.ups = nn.ModuleList()
        for out_ch in reversed(channels):
            self.ups.append(Upsample(curr, out_ch, num_res_blocks))
            upsample_channels.append(out_ch)
            curr = out_ch

        # -------- Output --------
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        self.final_sigmoid_activation = final_sigmoid_activation

        self.inject_noise_in_decoder = inject_noise_in_decoder
        if inject_noise_in_decoder:
            self.noise_scalers = nn.ParameterList()
            for ch in upsample_channels:
                # Shape is (1, C, 1, 1) to broadcast over Batch, Height, and Width
                self.noise_scalers.append(nn.Parameter(torch.zeros(1, ch, 1, 1)))

    def forward(self, x, cam_params):
        # Encoder
        for down in self.downs:
            x = down(x)

        # Bottleneck
        x = self.vit(x, cam_params)

        # Decoder
        for i, up in enumerate(self.ups):
            noise_scaler = None
            if self.inject_noise_in_decoder:
                noise_scaler = self.noise_scalers[i]
            x = up(x, noise_scaler=noise_scaler)

        x = self.final_conv(x)

        if self.final_sigmoid_activation:
            x = torch.sigmoid(x)

        return x


def OVIE_B(**kwargs):
    return ConditionalUViT(
        in_channels=3,
        out_channels=3,
        ch=128,
        ch_mult=[1, 2, 4],
        num_res_blocks=2,
        vit_hidden_size=768,
        vit_depth=12,
        vit_patch_size=1,
        vit_num_heads=12,
        final_sigmoid_activation=True,
        **kwargs,
    )


OVIE_models = {
    "OVIE_B": OVIE_B,
}
