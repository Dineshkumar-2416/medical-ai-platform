import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for ResNet18.

    Uses forward + backward hooks to capture:
      - Activations: feature maps from the target layer (forward pass)
      - Gradients:   d(class_score) / d(feature_map)  (backward pass)

    Then computes a weighted sum of feature maps to produce a heatmap
    that highlights regions responsible for the prediction.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.activations  = None
        self.gradients    = None
        self._hooks       = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # Capture feature maps during forward pass
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # Capture gradients during backward pass
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()

    def __call__(self, image_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate GradCAM heatmap.

        Args:
            image_tensor : preprocessed image [1, 3, 224, 224] on model device
            class_idx    : class to explain (None = use predicted class)

        Returns:
            heatmap : float32 array [224, 224], values in [0, 1]
        """
        self.model.eval()

        # Forward pass — keep graph for backward
        logits = self.model(image_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward pass — compute gradients for the target class only
        self.model.zero_grad()
        class_score = logits[0, class_idx]
        class_score.backward()

        # Gradients shape: [1, C, H, W] — global average pool over H, W
        # → importance weight per channel: [C]
        weights = self.gradients.mean(dim=[2, 3])[0]         # [C]

        # Activations shape: [1, C, H, W]
        activations = self.activations[0]                    # [C, H, W]

        # Weighted sum: sum over channels (C), keeping spatial dims (H, W)
        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU — keep only regions that pushed score UP, not down
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize from 7×7 → 224×224
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)

        return cam.astype(np.float32)


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay GradCAM heatmap on the original X-ray image.

    Args:
        original_image : RGB image [H, W, 3], uint8
        heatmap        : float32 [H, W], values in [0, 1]
        alpha          : heatmap transparency (0=invisible, 1=full)
        colormap       : cv2 colormap (JET = blue→green→red)

    Returns:
        overlay : RGB image [H, W, 3], uint8
    """
    heatmap_uint8  = np.uint8(255 * heatmap)
    heatmap_color  = cv2.applyColorMap(heatmap_uint8, colormap)       # BGR
    heatmap_rgb    = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)   # RGB

    if original_image.shape[:2] != heatmap_rgb.shape[:2]:
        heatmap_rgb = cv2.resize(heatmap_rgb, (original_image.shape[1], original_image.shape[0]))

    overlay = (1 - alpha) * original_image.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def run_gradcam(
    model: nn.Module,
    image_path: str,
    device: str = "cuda",
    image_size: int = 224,
) -> dict:
    """
    Full GradCAM pipeline: load image → preprocess → predict → explain.

    Returns dict with:
        prediction     : 'NORMAL' or 'PNEUMONIA'
        confidence     : float, probability of predicted class
        heatmap        : np.ndarray [224, 224] float32
        overlay        : np.ndarray [224, 224, 3] uint8
        original_resized: np.ndarray [224, 224, 3] uint8
    """
    from src.data.transforms import get_inference_transforms

    # Load original image (keep for overlay)
    original = np.array(Image.open(image_path).convert("RGB").resize((image_size, image_size)))

    # Preprocess for model
    transform   = get_inference_transforms(image_size)
    image_tensor = transform(image=original)["image"].unsqueeze(0).to(device)  # [1,3,224,224]

    # Attach GradCAM to layer4 (last conv block — highest-level features)
    target_layer = model.backbone.layer4
    gradcam      = GradCAM(model, target_layer)

    # Forward + backward
    heatmap = gradcam(image_tensor)

    # Get prediction
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    pred_idx    = probs.argmax().item()
    pred_label  = "PNEUMONIA" if pred_idx == 1 else "NORMAL"
    confidence  = probs[pred_idx].item()

    overlay_img = overlay_heatmap(original, heatmap)
    gradcam.remove_hooks()

    return {
        "prediction":       pred_label,
        "confidence":       round(confidence, 4),
        "prob_normal":      round(probs[0].item(), 4),
        "prob_pneumonia":   round(probs[1].item(), 4),
        "heatmap":          heatmap,
        "overlay":          overlay_img,
        "original_resized": original,
    }
