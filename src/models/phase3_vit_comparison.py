import torch
import torch.nn as nn
import timm

class Phase3ExperimentModel(nn.Module):
    def __init__(self, 
                 is_hybrid=False,
                 from_scratch=True,
                 model_name="deit_tiny_patch16_224",
                 num_layers=5,
                 nhead=4,
                 embed_dim=192, # DeiT-Tiny default
                 num_classes=2,
                 q_channels=180):
        super().__init__()
        
        self.is_hybrid = is_hybrid
        self.from_scratch = from_scratch

        # 1. Adapt Input Dimensions (The "Equalizer")
        # Classical: (1, 28, 28) -> (180, 28, 28)
        # Quantum: (180, 9, 9) -> (180, 9, 9)
        if not is_hybrid:
            self.input_adapter = nn.Conv2d(1, q_channels, kernel_size=1)
        else:
            self.input_adapter = nn.Identity()

        if from_scratch:
            # Custom Scratch Transformer
            self.patch_size = 1 if is_hybrid else 3 
            self.num_patches = (9//self.patch_size)**2 if is_hybrid else (28//self.patch_size)**2
            
            self.proj = nn.Conv2d(q_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*4, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Linear(embed_dim, num_classes)
            
        else:
            # DeiT Path
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes, in_chans=q_channels)
            # Standard DeiT expects 224x224. We upscale our small maps.
            self.upscaler = nn.Upsample(size=(224, 224), mode='bilinear')

    def forward(self, x):
        x = self.input_adapter(x)
        
        if self.from_scratch:
            x = self.proj(x).flatten(2).transpose(1, 2)
            x = x + self.pos_embed
            x = self.transformer(x)
            return self.head(x.mean(dim=1))
        else:
            x = self.upscaler(x)
            return self.backbone(x)

    def print_parameter_count(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"--- Model Params: {total:,} ---")