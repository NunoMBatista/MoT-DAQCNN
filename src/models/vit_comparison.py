import torch
import torch.nn as nn
import timm

class ExperimentModel(nn.Module):
    def __init__(self, 
                 is_hybrid=False,
                 from_scratch=True,
                 # Parameters for "From Scratch" Custom Transformer
                 num_layers=5,
                 nhead=4,
                 dim_feedforward=128,
                 # Parameters for "Standard Architecture" (timm)
                 model_name="deit_tiny_patch16_224",
                 pretrained=True,
                 # Shared Parameters
                 num_classes=2,
                 num_topologies=8, 
                 qubits_per_kernel=9):
        super().__init__()
        
        self.is_hybrid = is_hybrid
        self.from_scratch = from_scratch
        self.q_channels = num_topologies * qubits_per_kernel
        
        # --- LOGIC BRANCHING ---
        
        if from_scratch:
            print(f"[Model Init] Mode: CUSTOM SCRATCH TRANSFORMER ({num_layers} Layers)")
            # 1. Custom Embedding/Projection
            # Hybrid: (B, C, 9, 9) -> (B, 81, C) | Classical: (B, 1, 28, 28) -> (B, 49, C)
            if is_hybrid:
                self.embed_dim = self.q_channels
                self.num_patches = 9 * 9 # 81
            else:
                self.embed_dim = 64 # Choose an embedding dim for pixels
                self.patch_size = 4
                self.num_patches = (28 // self.patch_size)**2 # 49
                self.pixel_proj = nn.Conv2d(1, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

            # 2. Positional Encoding
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.embed_dim))

            # 3. Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                dropout=0.1
            )
            self.transformer_engine = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Linear(self.embed_dim, num_classes)

        else:
            # Mode: Standard Architecture (DeiT, etc.)
            print(f"[Model Init] Mode: STANDARD ARCHITECTURE ({model_name})")
            print(f"[Model Init] Pretrained: {pretrained}")
            
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
            
            # Adapters to fit our data into the standard 224x224 input of DeiT
            if is_hybrid:
                self.adapter = nn.Sequential(
                    nn.ConvTranspose2d(self.q_channels, 32, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Upsample(size=(224, 224), mode='bilinear'),
                    nn.Conv2d(32, 3, kernel_size=1)
                )
            else:
                self.adapter = nn.Sequential(
                    nn.Upsample(size=(224, 224), mode='bilinear'),
                    nn.Conv2d(1, 3, kernel_size=3, padding=1)
                )

        self.print_parameter_count()

    def forward(self, x):
        if self.from_scratch:
            if self.is_hybrid:
                # x: (B, C, 9, 9) -> (B, 81, C)
                B, C, H, W = x.shape
                x = x.view(B, C, H*W).permute(0, 2, 1)
            else:
                # x: (B, 1, 28, 28) -> Patchify to (B, 49, embed_dim)
                x = self.pixel_proj(x).flatten(2).transpose(1, 2)
            
            x = x + self.pos_embedding
            x = self.transformer_engine(x)
            return self.head(x.mean(dim=1)) # Global Average Pooling
        
        else:
            # Standard Path
            x = self.adapter(x)
            return self.backbone(x)

    def print_parameter_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"--- Stats: Total {total:,} | Trainable {trainable:,} ---")

    def freeze_backbone(self, freeze=True):
        """Only relevant for Standard Architecture mode"""
        if not self.from_scratch:
            for param in self.backbone.parameters():
                param.requires_grad = not freeze
            # Classifier head always stays open
            for param in self.backbone.get_classifier().parameters():
                param.requires_grad = True