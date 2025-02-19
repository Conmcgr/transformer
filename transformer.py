import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops
from multiheadattention import MultiHeadAttention

class CustomMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(CustomMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.W_q = nn.Parameter(torch.Tensor(d_model, d_model))
        self.W_k = nn.Parameter(torch.Tensor(d_model, d_model))
        self.W_v = nn.Parameter(torch.Tensor(d_model, d_model))
        self.W_o = nn.Parameter(torch.Tensor(d_model, d_model))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)
        nn.init.xavier_uniform_(self.W_o)

    def forward(self, x):
        #Self attention only so query, key and value are the same
        return MultiHeadAttention.apply(x, x, x, self.W_q, self.W_k, self.W_v, self.W_o, self.num_heads)
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_linear, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = CustomMultiHeadAttention(d_model, num_heads)
        self.linear1 = nn.Linear(d_model, d_linear)
        self.linear2 = nn.Linear(d_linear, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        attn_output = self.self_attn(src)

        src = self.norm1(src + self.dropout1(attn_output))

        linear_intermediate = self.linear1(src)

        linear_activated = F.relu(linear_intermediate)

        linear_output = self.linear2(linear_activated)

        src = self.norm2(src + self.dropout2(linear_output))

        return src
    

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x):
        x = einops.rearrange(x, 'batch channels (num_patches_vertical patch_height) (num_patches_horizontal patch_width) -> batch (num_patches_vertical num_patches_horizontal) (patch_height patch_width channels)', 
                  patch_height=self.patch_size, patch_width=self.patch_size)
        x = self.proj(x)
        return x
        
class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, num_classes,
                 d_model, num_heads, d_linear, num_layers, dropout=0.1):
        """
        image_size: Size of the input image (assumes square, e.g., 28 for MNIST).
        patch_size: Size of each patch (e.g., 2 for 2x2 patches).
        in_channels: Number of image channels (e.g., 1 for MNIST).
        num_classes: Number of output classes.
        d_model: Embedding dimension (and Transformer model dimension).
        num_heads: Number of attention heads.
        d_linear: Hidden dimension for feed-forward networks.
        num_layers: Number of Transformer encoder layers.
        dropout: Dropout probability.
        """
        super(VisionTransformer, self).__init__()
        
        #Convert images into embedded patches
        self.patch_embed = PatchEmbedding(d_model, patch_size, in_channels, d_model)
        
        num_patches = (image_size // patch_size) * (image_size // patch_size)
        
        # 2. [CLS] Token: a learnable vector to aggregate image-level information.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        #
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.pos_drop = nn.Dropout(dropout)
        
        #Stack of Transformer encoder layers.
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_linear, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
        # Map the CLS token representation to class logits.
        self.head = nn.Linear(d_model, num_classes)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Optionally, initialize the classification head.
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        #Convert image into patch embeddings.
        #Shape: (batch, num_patches, d_model)
        x = self.patch_embed(x)  
        batch_size = x.size(0)
        
        #Prepend CLS token to the patch embeddings.
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) 
        
        #Add positional embeddings.
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        #Pass the tokens through the Transformer encoder layers.
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        
        #Use the CLS token representation for classification.
        cls_output = x[:, 0]  # (batch, d_model)
        logits = self.head(cls_output)  # (batch, num_classes)
        return logits
    
if __name__ == "__main__":
    #Use dummy "MNIST-like" images
    batch_size = 8
    image_size = 28
    patch_size = 2      # 2x2 patches
    in_channels = 1
    num_classes = 10
    d_model = 64        # Embedding dimension (and Transformer model dimension)
    num_heads = 4
    d_linear = 128      # Feed-forward hidden dimension
    num_layers = 1      # Start with one encoder layer
    dropout = 0.1

    # Create dummy images to test output size
    dummy_images = torch.randn(batch_size, in_channels, image_size, image_size)

    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        d_model=d_model,
        num_heads=num_heads,
        d_linear=d_linear,
        num_layers=num_layers,
        dropout=dropout
    )

    logits = model(dummy_images)
    print("Logits shape:", logits.shape)