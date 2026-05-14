import torch
import torch.nn as nn
import timm


class ChestXrayModel(nn.Module):
    """
    ResNet18 fine-tuned for binary chest X-ray classification.

    Uses timm (PyTorch Image Models) — the industry standard library for
    pretrained vision models used at Google Brain, Facebook AI, etc.

    Transfer learning strategy:
      - Freeze early layers (they already know edges/textures from ImageNet)
      - Unfreeze layer4 + classifier (learn lung-specific features)
      - Replace final FC with a 2-class head + dropout for regularization
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3, freeze_layers: int = 6):
        super().__init__()

        # timm loads pretrained weights from HuggingFace Hub — more reliable than torchvision
        # num_classes=0 removes the original classifier, giving us raw 512-d features
        self.backbone = timm.create_model(
            "resnet18",
            pretrained=True,
            num_classes=0,       # strip the ImageNet head
            global_pool="avg",   # global average pooling → [batch, 512]
        )

        # --- Freeze early layers ---
        # timm ResNet18 named children: conv1, bn1, act1, maxpool,
        #                               layer1, layer2, layer3, layer4
        all_children = list(self.backbone.children())
        for layer in all_children[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        # --- New classification head ---
        in_features = self.backbone.num_features   # 512 for ResNet18
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, 3, 224, 224]
        x = self.backbone(x)      # → [batch, 512]  (timm handles pool + flatten)
        x = self.classifier(x)    # → [batch, 2]
        return x                  # raw logits (no softmax — CrossEntropyLoss handles it)

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def summary(self):
        total     = self.get_total_params()
        trainable = self.get_trainable_params()
        frozen    = total - trainable
        print(f"Model          : ResNet18 (fine-tuned)")
        print(f"Total params   : {total:,}")
        print(f"Trainable      : {trainable:,}  ({100*trainable/total:.1f}%)")
        print(f"Frozen         : {frozen:,}  ({100*frozen/total:.1f}%)")
