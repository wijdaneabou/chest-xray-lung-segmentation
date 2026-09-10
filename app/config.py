"""
Central configuration for the lung segmentation deployment app.
Mirrors the Config class used in the training notebook (unetpp-lung-segmentation.ipynb),
but only keeps what inference needs (no dataset paths, no training hyperparameters).
"""

import json
import os

import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

# JSON exported by the notebook (Section 6 "Saving the final model + deployment config").
DEPLOYMENT_CONFIG_PATH = os.path.join(CHECKPOINTS_DIR, "deployment_config_robust.json")

# Fallback only -- used if deployment_config_robust.json is missing or its weights_path
# doesn't resolve to a file actually present in checkpoints/.
DEFAULT_MODEL_WEIGHTS_PATH = os.path.join(CHECKPOINTS_DIR, "unetpp_resnet34_final_robust.pth")


class Config:
    IMG_SIZE = 256
    NUM_CLASSES = 3  # background, right lung, left lung
    CLASS_NAMES = ["Background", "Right lung", "Left lung"]

    # NOTE: deployment_config_robust.json does NOT currently export norm_mean/norm_std --
    # these hardcoded values ARE the source of truth for now (copied from the notebook's
    # Config.NORM_MEAN/NORM_STD printout, cell 55). If the notebook is re-run with a
    # different train split (e.g. a different random_state or dataset version), these
    # MUST be updated manually, since nothing will warn you of a mismatch otherwise.
    NORM_MEAN = 0.5706
    NORM_STD = 0.2804

    # Post-processing defaults -- overwritten from deployment_config.json when present.
    APPLY_POSTPROCESSING = True
    MIN_AREA_RATIO = 0.05
    FILL_HOLES = True

    BEST_BACKBONE = "unetpp_resnet34"

    # Metrics reported by the notebook at export time -- purely informational (e.g. for
    # an "About" panel in the app), never used for any inference logic.
    METRICS = {}

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_WEIGHTS_PATH = DEFAULT_MODEL_WEIGHTS_PATH


def load_deployment_metadata():
    """
    Loads img_size / num_classes / class_names / best_backbone / weights_path /
    postprocessing / metrics from deployment_config.json when present, so the app stays
    in sync with whatever the notebook last exported -- falls back to the hardcoded
    defaults above if the file hasn't been copied into checkpoints/ yet.

    IMPORTANT: this JSON schema has no norm_mean/norm_std field -- Config.NORM_MEAN/STD
    always come from the hardcoded fallback above, never from this file.
    """
    if not os.path.exists(DEPLOYMENT_CONFIG_PATH):
        print(
            f"[config] WARNING: {DEPLOYMENT_CONFIG_PATH} not found -- "
            f"using hardcoded defaults, which may be stale."
        )
        return

    with open(DEPLOYMENT_CONFIG_PATH, "r") as f:
        meta = json.load(f)

    Config.IMG_SIZE = meta.get("img_size", Config.IMG_SIZE)
    Config.NUM_CLASSES = meta.get("num_classes", Config.NUM_CLASSES)
    Config.CLASS_NAMES = meta.get("class_names", Config.CLASS_NAMES)
    Config.BEST_BACKBONE = meta.get("best_backbone", Config.BEST_BACKBONE)
    Config.METRICS = meta.get("metrics", {})

    # Only the filename is portable -- the JSON's "weights_path" is an absolute Kaggle
    # path (/kaggle/working/...) that doesn't exist on this machine.
    weights_path_in_json = meta.get("weights_path")
    if weights_path_in_json:
        weights_filename = os.path.basename(weights_path_in_json)
        candidate_path = os.path.join(CHECKPOINTS_DIR, weights_filename)
        if os.path.exists(candidate_path):
            Config.MODEL_WEIGHTS_PATH = candidate_path
        else:
            print(
                f"[config] WARNING: deployment_config names '{weights_filename}', but it "
                f"isn't in {CHECKPOINTS_DIR} -- keeping fallback {Config.MODEL_WEIGHTS_PATH}."
            )
    else:
        print(
            "[config] WARNING: 'weights_path' missing from deployment_config.json -- "
            f"keeping fallback {Config.MODEL_WEIGHTS_PATH}."
        )

    print(
        "[config] NOTE: norm_mean/norm_std are not exported in this JSON schema -- "
        f"using hardcoded values (mean={Config.NORM_MEAN}, std={Config.NORM_STD})."
    )

    postprocessing = meta.get("postprocessing", {})
    Config.APPLY_POSTPROCESSING = postprocessing.get("enabled", Config.APPLY_POSTPROCESSING)
    Config.MIN_AREA_RATIO = postprocessing.get("min_area_ratio", Config.MIN_AREA_RATIO)
    Config.FILL_HOLES = postprocessing.get("fill_holes", Config.FILL_HOLES)


load_deployment_metadata()