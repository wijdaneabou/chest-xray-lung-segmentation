

import json
from pathlib import Path

import torch
import segmentation_models_pytorch as smp

# ==== Paths, resolved relative to this file so it works regardless of cwd ====
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

DEFAULT_CHECKPOINT_PATH = CHECKPOINTS_DIR / "unetpp_resnet34_final_robust.pth"
DEFAULT_DEPLOYMENT_CONFIG_PATH = CHECKPOINTS_DIR / "deployment_config_res.json"

# Same decoder widths as the training notebook (UNETPP_DECODER_CHANNELS).
# Must match exactly, or the saved state_dict won't load.
DECODER_CHANNELS = (256, 128, 64, 32, 16)

# Fallback if deployment_config.json is missing or doesn't have these keys.
DEFAULT_NUM_CLASSES = 3
DEFAULT_CLASS_NAMES = ["background", "right_lung", "left_lung"]
DEFAULT_IMG_SIZE = 256


def load_deployment_config(config_path=DEFAULT_DEPLOYMENT_CONFIG_PATH):
    """
    Read deployment_config.json (saved by the training notebook) if present.
    Returns an empty dict if the file is missing, so callers can fall back
    to the DEFAULT_* constants above without crashing.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


def build_model(num_classes=DEFAULT_NUM_CLASSES, pretrained=False, device="cpu"):
    """
    Instantiate U-Net++ with a ResNet34 encoder.

    Use pretrained=False when loading your own trained checkpoint (the
    ImageNet weights get overwritten by load_state_dict anyway) — only set
    pretrained=True if you intend to train from scratch.
    """
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        decoder_channels=DECODER_CHANNELS,
        in_channels=1,  # grayscale; smp averages the pretrained RGB filters into 1 channel
        classes=num_classes,
    )
    return model.to(device)


def load_model(checkpoint_path=DEFAULT_CHECKPOINT_PATH, num_classes=None, device="cpu"):
    """
    Build the model and load trained weights from a .pth checkpoint (state_dict).

    If num_classes isn't given, it's read from deployment_config.json when
    available, otherwise DEFAULT_NUM_CLASSES.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Expected it under {CHECKPOINTS_DIR}"
        )

    if num_classes is None:
        deployment_config = load_deployment_config()
        num_classes = deployment_config.get("num_classes", DEFAULT_NUM_CLASSES)

    model = build_model(num_classes=num_classes, pretrained=False, device=device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == "__main__":
    # Quick smoke test: load the real checkpoint (if present) and run a dummy
    # forward pass, or fall back to an untrained model so the architecture
    # itself can still be sanity-checked without the .pth file.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if DEFAULT_CHECKPOINT_PATH.exists():
        model = load_model(device=device)
        print(f"Loaded checkpoint: {DEFAULT_CHECKPOINT_PATH}")
    else:
        model = build_model(pretrained=False, device=device)
        print(f"No checkpoint found at {DEFAULT_CHECKPOINT_PATH} — using untrained weights.")

    dummy = torch.randn(1, 1, DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE, device=device)
    with torch.no_grad():
        out = model(dummy)
    print(f"Output shape: {tuple(out.shape)}  (expected: (1, {DEFAULT_NUM_CLASSES}, {DEFAULT_IMG_SIZE}, {DEFAULT_IMG_SIZE}))")