#!/usr/bin/env python3
"""
Run ViTPose on a single image and export keypoints + visualization.

Uses the same YOLO + ViTPose whole-body models as the WHAM preprocessor.

Example:
    python infer_vitpose_image.py --image /path/to/photo.jpg --output-dir ./vitpose_out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
WHAM_ROOT = REPO_ROOT / "WHAM"

# WHAM imports expect repo root and WHAM on sys.path (see WHAM/demo.py).
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(WHAM_ROOT))

from WHAM.lib.models.preproc.detector import (  # noqa: E402
    BBOX_CONF,
    DetectionModel,
    VIS_THRESH,
)
from mmpose.apis import inference_top_down_pose_model, vis_pose_result  # noqa: E402
from mmpose.datasets import DatasetInfo  # noqa: E402

DEFAULT_VIT_CFG = (
    "configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/"
    "ViTPose_huge_wholebody_256x192.py"
)
DEFAULT_VIT_CKPT = "checkpoints/vitpose+_huge_wholebody/wholebody.pth"
DEFAULT_DATASET = "TopDownCocoWholeBodyDataset"


def _bbox_area_xyxy(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _select_persons(
    pose_results: List[dict],
    mode: str,
) -> List[dict]:
    if not pose_results:
        return []
    if mode == "all":
        return pose_results
    if mode == "largest":
        return [max(pose_results, key=lambda r: _bbox_area_xyxy(r["bbox"]))]
    raise ValueError(f"Unknown person selection mode: {mode}")


def _keypoint_names(pose_model) -> List[str]:
    if hasattr(pose_model, "cfg") and "dataset_info" in pose_model.cfg:
        info = DatasetInfo(pose_model.cfg.dataset_info)
        return [info.keypoint_id2name[i] for i in range(info.keypoint_num)]
    num_kpts = pose_model.cfg.model["keypoint_head"]["out_channels"]
    return [str(i) for i in range(num_kpts)]


def _pose_results_to_json(
    image_path: Path,
    image_bgr: np.ndarray,
    pose_results: List[dict],
    keypoint_names: List[str],
    kpt_score_thr: float,
) -> Dict[str, Any]:
    height, width = image_bgr.shape[:2]
    persons = []
    for person_idx, pose in enumerate(pose_results):
        kpts = pose["keypoints"]
        bbox = pose.get("bbox")
        if bbox is None:
            bbox = [0.0, 0.0, 0.0, 0.0]
        keypoints = []
        for kid, (x, y, score) in enumerate(kpts):
            name = keypoint_names[kid] if kid < len(keypoint_names) else str(kid)
            keypoints.append(
                {
                    "id": kid,
                    "name": name,
                    "x": float(x),
                    "y": float(y),
                    "score": float(score),
                    "visible": bool(score >= kpt_score_thr),
                }
            )
        persons.append(
            {
                "person_index": person_idx,
                "bbox_xyxy": [float(v) for v in bbox],
                "num_keypoints": len(keypoints),
                "keypoints": keypoints,
            }
        )

    return {
        "image_path": str(image_path.resolve()),
        "image_size": {"width": int(width), "height": int(height)},
        "num_persons": len(persons),
        "num_keypoints_per_person": len(keypoint_names),
        "keypoint_names": keypoint_names,
        "kpt_score_thr": kpt_score_thr,
        "persons": persons,
    }


def infer_vitpose_on_image(
    image_path: Path,
    output_dir: Path,
    device: str = "cuda:0",
    person_mode: str = "largest",
    kpt_score_thr: float = VIS_THRESH,
    vit_cfg: str = DEFAULT_VIT_CFG,
    vit_ckpt: str = DEFAULT_VIT_CKPT,
) -> Tuple[Dict[str, Any], Path, Path]:
    """
    Run ViTPose on one image.

    Returns:
        result_json (dict), path to keypoints JSON, path to overlay image.
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    detector = DetectionModel(device=device, vit_cfg=vit_cfg, vit_ckpt=vit_ckpt)

    bboxes = detector.bbox_model.predict(
        image_bgr,
        device=detector.device,
        classes=0,
        conf=BBOX_CONF,
        save=False,
        verbose=False,
    )[0].boxes.xyxy.detach().cpu().numpy()
    person_results = [{"bbox": bbox} for bbox in bboxes]

    if not person_results:
        raise ValueError(
            "No person detected in the image. Try a clearer full-body photo."
        )

    pose_results, _ = inference_top_down_pose_model(
        detector.pose_model,
        image_bgr,
        person_results=person_results,
        format="xyxy",
        return_heatmap=False,
        outputs=None,
    )

    pose_results = _select_persons(pose_results, person_mode)
    if not pose_results:
        raise ValueError("Pose model returned no valid poses for detected persons.")

    keypoint_names = _keypoint_names(detector.pose_model)
    result_json = _pose_results_to_json(
        image_path, image_bgr, pose_results, keypoint_names, kpt_score_thr
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    json_path = output_dir / f"{stem}_keypoints.json"
    overlay_path = output_dir / f"{stem}_keypoints_overlay.jpg"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)

    vis_pose_result(
        detector.pose_model,
        image_bgr,
        pose_results,
        radius=4,
        thickness=2,
        kpt_score_thr=kpt_score_thr,
        dataset=DEFAULT_DATASET,
        show=False,
        out_file=str(overlay_path),
    )

    result_json["outputs"] = {
        "keypoints_json": str(json_path.resolve()),
        "overlay_image": str(overlay_path.resolve()),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)

    return result_json, json_path, overlay_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer ViTPose keypoints on a single image."
    )
    parser.add_argument(
        "--image",
        "-i",
        required=True,
        type=Path,
        help="Path to input image (jpg/png/...).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("vitpose_output"),
        help="Directory for JSON and overlay image (default: ./vitpose_output).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cuda:0 or cpu (default: cuda:0 if available).",
    )
    parser.add_argument(
        "--person",
        choices=("largest", "all"),
        default="largest",
        help="Return keypoints for the largest detected person or all persons.",
    )
    parser.add_argument(
        "--kpt-thr",
        type=float,
        default=VIS_THRESH,
        help=f"Keypoint score threshold for visualization (default: {VIS_THRESH}).",
    )
    parser.add_argument(
        "--vit-cfg",
        default=DEFAULT_VIT_CFG,
        help="ViTPose config path relative to WHAM/third-party/ViTPose.",
    )
    parser.add_argument(
        "--vit-ckpt",
        default=DEFAULT_VIT_CKPT,
        help="ViTPose checkpoint path relative to WHAM/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device is None:
        try:
            import torch

            args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
    elif args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            print("CUDA not available; falling back to cpu.")
            args.device = "cpu"

    result, json_path, overlay_path = infer_vitpose_on_image(
        image_path=args.image,
        output_dir=args.output_dir,
        device=args.device,
        person_mode=args.person,
        kpt_score_thr=args.kpt_thr,
        vit_cfg=args.vit_cfg,
        vit_ckpt=args.vit_ckpt,
    )

    n_kpts = result["num_keypoints_per_person"]
    n_persons = result["num_persons"]
    print(f"Detected {n_persons} person(s), {n_kpts} keypoints each.")
    print(f"Keypoints JSON: {json_path}")
    print(f"Overlay image:  {overlay_path}")


if __name__ == "__main__":
    main()
