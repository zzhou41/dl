import numpy as np
from layers import Linear, SelfAttention

class AttentionMNIST:
    def __init__(self):
        # represent each patch with 64 learned features
        # curvature, endpoints, etc
        # attention needs rich vectors
        # small enough to train stably and efficiently
        # keep score variance well-behaved
        # for vision, 4-8x patch_dim
        self.embed = Linear(16, 64)
        self.attn = SelfAttention(64)
        self.classifier = Linear(64, 10)

    def forward(self, x):
        self.x = x
        x = self.embed.forward(x)
        attn_out = self.attn.forward(x)
        # RESIDUAL CONNECTION: pure self-attention without it almost never learns
        x = x + attn_out
        # (B, 49, 64) -> (B, 64)
        # classification expects one vector per image
        # converts patch-level features into a single image representation
        # by averaging globally-informed patch embeddings
        # minimal parameters
        self.pooled = x.mean(axis=1)
        return self.classifier.forward(self.pooled)
