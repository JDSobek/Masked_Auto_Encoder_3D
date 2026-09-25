"""
Code to build a 3D masked auto encoder and to extract the image encoder from a trained model.

Reference pages:
    https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6
    https://github.com/Alpsource/Visual-Representation-Learning-MAE/blob/main/models.py
    https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
    https://github.com/huggingface/transformers/blob/v5.12.0/src/transformers/models/vit_mae/modeling_vit_mae.py
"""

from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_position_encoding(sequence_length: int, num_channels: int, temperature=10000)-> torch.Tensor:
    """
    Generates sinusoidal position encoding.

    Original (https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6)
    """
    positional_encoding = torch.zeros(sequence_length, num_channels)
    for position in range(sequence_length):
        for dim in range(num_channels):
            if dim % 2 == 0:
                positional_encoding[position] = np.sin(position/(temperature**(dim/num_channels)))
            else:
                positional_encoding[position] = np.cos(position/(temperature**((dim-1)/num_channels)))
    return positional_encoding


def functional_unpatchify(patchified_pixel_values: torch.Tensor, patch_size: Tuple[int], img_size: Tuple[int], in_channels=1, class_token=False) -> torch.Tensor:
        """
        Reconstructs image input from decoder output

        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_voxel_volume)`):
                Patchified pixel values.
            patch_size: size used for patch generation
            img_size: original image size
            in_channels: original image channels
            class_token: whether the incoming pixel values include a class token that will need to be dropped for reconstruction

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_channels, depth, height, width)`:
                Pixel values.

        Original (2D version):
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/seggpt/modeling_seggpt.py#L757
        """

        patch_size_z = patch_size[0]
        patch_size_y = patch_size[1]
        patch_size_x = patch_size[2]

        num_patches_z = img_size[0] // patch_size_z
        num_patches_y = img_size[1] // patch_size_y
        num_patches_x = img_size[2] // patch_size_x

        num_patches = (img_size[0] * img_size[1] * img_size[2]) // (patch_size[0] * patch_size[1] * patch_size[2])

        if class_token:
            if num_patches + 1 != (patchified_pixel_values.shape[1]):
                raise ValueError(
                    f"The number of patches in the patchified pixel values {patchified_pixel_values.shape[1]}, does not match the number of patches on original image {num_patches} plus a class token"
                )
            # drop the class token from final image
            patchified_pixel_values = patchified_pixel_values[:,1:,:]
        else:
            if num_patches != patchified_pixel_values.shape[1]:
                raise ValueError(
                    f"The number of patches in the patchified pixel values {patchified_pixel_values.shape[1]}, does not match the number of patches on original image {num_patches}"
                )

        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size, # 0
            num_patches_z, # 1
            num_patches_y, # 2
            num_patches_x, # 3
            patch_size_z, # 4
            patch_size_y, # 5
            patch_size_x, # 6
            in_channels, # 7
        )
        # batch, channels, num patches in z, patch size z, num patches in y, patch size y, num patches in x, patch size x
        patchified_pixel_values = patchified_pixel_values.permute(0, 7, 1, 4, 2, 5, 3, 6)
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            in_channels,
            num_patches_z * patch_size_z,
            num_patches_y * patch_size_y,
            num_patches_x * patch_size_x,
        )
        return pixel_values


def functional_patchify(tensor: torch.Tensor, patch_size: Tuple[int]) -> torch.Tensor:
    """
    tensor: the input tensor that needs to be patchified
    patch_size: the 3D size of the patches to be generated

    Original (2D version):
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/seggpt/modeling_seggpt.py#L745
    """
    batch_size, num_channels, depth, height, width = tensor.shape

    patch_size_z = patch_size[0]
    patch_size_y = patch_size[1]
    patch_size_x = patch_size[2]

    if depth % patch_size_z != 0 or height % patch_size_y != 0 or width % patch_size_x != 0:
        raise ValueError("tensor dimensions must be divisible by patch_size dimensions")   

    num_patches_z = depth // patch_size_z
    num_patches_y = height // patch_size_y
    num_patches_x = width // patch_size_x

    # batch, channels, num patches in z, patch size z, num patches in y, patch size y, num patches in x, patch size x
    tensor = tensor.reshape(shape=(batch_size, num_channels, num_patches_z, patch_size_z, num_patches_y, patch_size_y, num_patches_x, patch_size_x))
    # batch, channels, num z patches, num y patches, num x patches, patch size z, patch size y, patch size x, channels
    tensor = tensor.permute(0, 2, 4, 6, 3, 5, 7, 1)
    # batch, total number of patches, patch voxel volume
    tensor = tensor.reshape(shape=(batch_size, num_patches_z*num_patches_y*num_patches_x, patch_size_z*patch_size_y*patch_size_x*num_channels))

    return tensor


def functional_random_masking(x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor]:
    """
    Perform per-sample random masking by per-sample shuffling.
    x: [N, L, D], sequence

    Original (unchanged): https://github.com/Alpsource/Visual-Representation-Learning-MAE/blob/main/models.py#L78
    """
    N, L, D = x.shape  # Batch, Length, Dim
    len_keep = int(L * (1 - mask_ratio))
    
    # Generate noise for random masking
    noise = torch.rand(N, L, device=x.device)
    
    # argsort gives us the indices that would sort the array, for selecting random patches
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    # Keep the first 'len_keep' subsets
    ids_keep = ids_shuffle[:, :len_keep]
    
    # Gather the visible patches
    # We expand indices to [N, len_keep, D] to gather along the sequence dimension
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    
    # Generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # Unshuffle to get the binary mask in original order
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return x_masked, mask, ids_restore


class PatchEmbedding3D(nn.Module):
    """
    Conv patch embedding works for the simple loss calculation of comparing 
    the full input image (both masked and unmasked patches) to the unpatchified model output.
    However, it might be preferable or more proper to only calculate loss on the masked patches,
    which isn't possible (or is harder) after the conv layer projects the input patch to the embedding dimension.

    This alternative way of patch embedding uses a callable patchify function and projects
    the patch pixels into the embedding dimension using a linear layer instead of using a convolution layer to do both.
    This means the patchify function can be called separately and be used with the mask tensor
    to only calculate loss on the masked patches.
    """
    def __init__(self, img_size: Tuple[int], patch_size: Tuple[int], embed_dim: int, in_channels: int):
        super().__init__()
        if img_size[0] % patch_size[0] != 0 or img_size[1] % patch_size[1] != 0 or img_size[2] % patch_size[2] != 0:
            raise ValueError("img_size dimensions must be divisible by patch_size dimensions")

        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.patch_voxels = (self.patch_size[0] * self.patch_size[1] * self.patch_size[2]) * self.in_channels

        self.projection_layer = nn.Linear(self.patch_voxels, self.embed_dim)
        self.norm = nn.Identity()  # replace with desired normalization function

    def patchify(self, tensor: torch.Tensor) -> torch.Tensor:
        return functional_patchify(tensor, self.patch_size)

    def forward(self, x: torch.Tensor):
        x = self.patchify(x)
        x = self.projection_layer(x)
        x = self.norm(x)
        return x
    

class ConvPatchEmbedding3D(nn.Module):
    """
    Slightly simpler method to generate and project patch embeddings using a convolutional layer.

    Originals (2D):
    https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6
    https://github.com/huggingface/transformers/blob/v5.12.0/src/transformers/models/vit_mae/modeling_vit_mae.py#L104
    """
    def __init__(self, img_size: Tuple[int], patch_size: Tuple[int], embed_dim: int, in_channels: int):
        super().__init__()
        if img_size[0] % patch_size[0] != 0 or img_size[1] % patch_size[1] != 0 or img_size[2] % patch_size[2] != 0:
            raise ValueError("img_size dimensions must be divisible by patch_size dimensions")

        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        self.projection_layer = nn.Conv3d(in_channels=self.in_channels, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = nn.Identity()  # replace with desired normalization function

    def forward(self, x: torch.Tensor):
        x = self.projection_layer(x) # (B, in_channels, D, H, W) -> (Batch, embed_dim, patch_num_z, patch_num_y, patch_num_x)
        x = self.norm(x)
        x = x.flatten(2) # (Batch, embed_dim, patch_num_z, patch_num_y, patch_num_x) -> (Batch, embed_dim, num_patches)
        x = x.transpose(1, 2) # (Batch, embed_dim, num_patches) -> (Batch, num_patches, embed_dim)
        return x
    

class SinusoidalPositionEncoding(nn.Module):
    """
    Sinusoidal positional encoding layer for pytorch models
    """
    def __init__(self, embed_dim: int, num_patches: int, temperature=10000):
        super().__init__()
        # sinusoidal position encoding
        positional_encoding = sinusoidal_position_encoding(num_patches, embed_dim, temperature)

        # buffers inherit the model's device but don't update values like parameters or nn layers
        self.register_buffer('positional_encoding', positional_encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        # Add positional encoding to embeddings
        x = x + self.positional_encoding
        return x
    

class MultiHeadSelfAttention(nn.Module):
    """
    Modified From: https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
    Computes multi-head self attention. Supports nested or padded tensors. Should be more efficient than nn.MultiheadAttention.

    Args:
        embed_dim (int): Size of embedding dim for keys, queries, and values
        num_heads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if embed_dim % num_heads != 0:
            raise("Embedding dimension must be divisible by number of heads")

        self.nheads = num_heads
        self.dropout = dropout

        self.embed_dim = embed_dim
        self.head_size = embed_dim // num_heads

        # this is a more efficient implementation of the q, k, v projection
        # further simplified from the tutorial for the self-attention case
        self.packed_proj = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias, **factory_kwargs)

        self.bias = bias

    def forward(
        self,
        x: torch.Tensor,
        is_causal=False,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for scaled dot product attention
            3. Run scaled dot product attention
            4. Apply output projection

        Args:
            x (torch.Tensor): input of shape (batch_size (N), length [patch voxel count] (L_t), embed dimension (E_q))
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        
        # Step 1. Apply input projection
        result = self.packed_proj(x)
        query, key, value = torch.chunk(result, 3, dim=-1)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.head_size]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.head_size]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.head_size]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout, is_causal=is_causal
        )
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output
    

class TransformerBlock(nn.Module):
    """
    Original: https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_expansion_ratio=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_expansion_ratio = mlp_expansion_ratio
        
        # Sub-Layer 1 Normalization
        self.norm1 = nn.LayerNorm(self.embed_dim)
        # Multi-Head Attention
        self.mha = MultiHeadSelfAttention(self.embed_dim, self.num_heads)
        # Sub-Layer 2 Normalization
        self.norm2 = nn.LayerNorm(self.embed_dim)
        # Multilayer Perception
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim*self.mlp_expansion_ratio),
            nn.GELU(),
            nn.Linear(self.embed_dim*self.mlp_expansion_ratio, self.embed_dim)
        )

    def forward(self, x: torch.Tensor):
        # Residual Connection After Sub-Layer 1
        x = x + self.mha(self.norm1(x))
        # Residual Connection After Sub-Layer 2
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    Original: https://github.com/Alpsource/Visual-Representation-Learning-MAE/blob/main/models.py#6
    """
    def __init__(self, embed_dim: int, num_heads: int, depth: int,
                img_size: Tuple[int], patch_size: Tuple[int], in_channels: int, mask_ratio: float, 
                class_token=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth

        # Input
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (self.img_size[0] * self.img_size[1] * self.img_size[2]) // (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        self.mask_ratio = mask_ratio

        # controls whether to build the encoder to handle a classification token or not
        # if the model includes one, take that into account when unpatchifying data
        if class_token:
            self.class_bool = True
            self.class_token = nn.Parameter(torch.randn(1, 1, self.embed_dim)) # Classification Token
        else:
            self.class_bool = False

        # Input Processing
        # self.patch_embedding = ConvPatchEmbedding3D(self.img_size, self.patch_size, self.embed_dim, self.in_channels)
        self.patch_embedding = PatchEmbedding3D(self.img_size, self.patch_size, self.embed_dim, self.in_channels)
        self.position_encoding = SinusoidalPositionEncoding(self.embed_dim, self.num_patches)

        # Layers
        self.transformer_layers = nn.Sequential(*[TransformerBlock(self.embed_dim, self.num_heads) for _ in range(self.depth)])
        self.norm = nn.LayerNorm(self.embed_dim)

    def _init_weights(self):
        self.apply(self._init_vit_weights)

    def _init_vit_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # MASKING LOGIC (The Core of MAE)
    def random_masking(self, x: torch.Tensor):
        return functional_random_masking(x, self.mask_ratio)

    def forward(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        x = self.position_encoding(x)
        x, mask, ids_restore = self.random_masking(x)
        if self.class_bool:
            # Expand to have class token for every image in batch
            tokens_batch = self.class_token.expand(x.size()[0], -1, -1)
            # Adding class tokens to the beginning of each embedding
            x = torch.cat((tokens_batch, x), dim=1)
        x = self.transformer_layers(x)
        return self.norm(x), mask, ids_restore
    

class ViTDecoder(nn.Module):
    """
    Original: https://github.com/Alpsource/Visual-Representation-Learning-MAE/blob/main/models.py#6
    """
    def __init__(self, embed_dim: int, num_heads: int, depth: int, num_patches: int,
                 class_token=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.num_patches = num_patches

        # Input Processing
        self.position_encoding = SinusoidalPositionEncoding(self.embed_dim, self.num_patches)

        # Mask token (Learnable placeholder for missing pixels)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        if class_token:
            self.class_bool = True
        else:
            self.class_bool = False

        # Layers
        self.transformer_layers = nn.Sequential(*[TransformerBlock(self.embed_dim, self.num_heads) for _ in range(self.depth)])
        self.norm = nn.LayerNorm(self.embed_dim)

    def _init_weights(self):
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_vit_weights)

    def _init_vit_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor):
        # append mask tokens to input sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1] + 1, 1)
        if self.class_bool:
            # need to drop class token while unshuffling patches
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # unmasked tokens
        else:
            x_ = torch.cat([x, mask_tokens], dim=1)  # unmasked tokens
        # 3. Unshuffle (Restore Order)
        # gather(input, dim, index)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]).to(x_.device))

        x_ = self.position_encoding(x_)

        if self.class_bool:
            # class token is re-added here
            x = torch.cat([x[:, :1, :], x_], dim=1)
        else:
            x = x_

        x = self.transformer_layers(x)
        return self.norm(x)
    

class MaskedAutoEncoder(nn.Module):
    """
    Original: https://github.com/Alpsource/Visual-Representation-Learning-MAE/blob/main/models.py#6
    """
    def __init__(self, encoder_num_heads=12, encoder_num_layers=12, encoder_embed_dim=768,
                decoder_num_heads=8, decoder_num_layers=12, decoder_embed_dim=512,
                img_size=(224, 224, 224), patch_size=(16, 16, 16), in_channels=1, mask_ratio = 0.75, class_token=True):
        super().__init__()

        # Input
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (self.img_size[0] * self.img_size[1] * self.img_size[2]) // (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        self.patch_voxels = (self.patch_size[0] * self.patch_size[1] * self.patch_size[2]) * self.in_channels
        self.mask_ratio = mask_ratio
        self.class_token = class_token

        # MAE Encoder
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_depth = encoder_num_layers
        self.encoder_num_heads = encoder_num_heads

        # MAE Decoder
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_num_layers
        self.decoder_num_heads = decoder_num_heads

        # Model
        self.transformer_encoder = ViTEncoder(self.encoder_embed_dim, self.encoder_num_heads, self.encoder_depth,
                                              self.img_size, self.patch_size, self.in_channels,
                                              mask_ratio=self.mask_ratio, class_token=self.class_token)
        self.decoder_embed = nn.Linear(self.encoder_embed_dim, self.decoder_embed_dim, bias=True)       
        self.transformer_decoder = ViTDecoder(self.decoder_embed_dim, self.decoder_num_heads, self.decoder_depth, self.num_patches, class_token=self.class_token)
        self.decoder_prediction_head = nn.Linear(self.decoder_embed_dim, self.patch_voxels, bias=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        self.apply(self._init_vit_weights)
        self.transformer_encoder._init_weights()
        self.transformer_decoder._init_weights()

    def _init_vit_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # FORWARD PASS
    def forward_encoder(self, x: torch.Tensor):
        x, mask, ids_restore = self.transformer_encoder(x)
        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor):
        # 1. Embed tokens to decoder dimension
        x = self.decoder_embed(x)
        # 5. Apply Transformer Decoder
        x = self.transformer_decoder(x, ids_restore)        
        # 6. Predict Pixels
        x = self.decoder_prediction_head(x)
        return x

    def forward(self, x: torch.Tensor):
        latent, mask, ids_restore = self.forward_encoder(x)
        preds = self.forward_decoder(latent, ids_restore)
        return preds, mask
    
    def unpatchify(self, patchified_pixel_values: torch.Tensor) -> torch.Tensor:
        return functional_unpatchify(patchified_pixel_values, self.patch_size, self.img_size, self.in_channels, self.class_token)
    
    def get_encoder(self):
        # Helper to extract backbone for Fine-Tuning
        return self.transformer_encoder
    

if __name__ == '__main__':
    """
    Script below is written to test changes above and to show how the model should be called and how to save/load the encoder weights after pre-training.
    """

    model = MaskedAutoEncoder(encoder_num_heads=12, encoder_num_layers=12, encoder_embed_dim=768,
                            decoder_num_heads=8, decoder_num_layers=12, decoder_embed_dim=512,
                            img_size=(224, 224, 224), patch_size=(16, 16, 16), in_channels=1, 
                            mask_ratio = 0.75, class_token=True)
    # print(model)

    # [batch, channels, depth, width, height]
    test_tensor = torch.empty([2, 1, 224, 224, 224])

    # input reading test
    print('input', test_tensor.size())

    model_output, mask_indices = model(test_tensor)
    print('output', model_output.size())
    print('masks', mask_indices.size())

    test_unpatch = model.unpatchify(model_output)
    print('unpatchified output', test_unpatch.size())

    encoder = model.get_encoder()
    print('position encoding', encoder.position_encoding.positional_encoding.size())

    # test saving and loading encoder weights with changed mask ratio
    # looks like everything is working
    test_out = encoder(test_tensor)
    print('output of first model encoder (mask ratio = 0.75)', test_out[0].size()) # embedded features (unmasked patches only)

    import os
    test_filename = os.path.abspath(r"./test_weights/mae_test.pt")
    torch.save(encoder.state_dict(), test_filename)
    
    # second model that doesn't use masking, for feature extraction for downstream tasks
    # only build the encoder since that's all we need
    # same parameters, except the mask ratio is changed
    model2 = ViTEncoder(embed_dim=768, num_heads=12, depth=12,
                        img_size=(224, 224, 224), patch_size=(16, 16, 16), in_channels=1, 
                        mask_ratio=0.0, class_token=True)
    
    model2.load_state_dict(torch.load(test_filename, weights_only=True))

    test_out2 = model2(test_tensor)
    print('output of second model (mask ratio = 0)', test_out2[0].size()) # embedded features (unmasked patches only)