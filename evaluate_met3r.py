#!/usr/bin/env python3
"""Evaluate multi-view consistency (MEt3R) over a folder of generated frames.

MEt3R [1] measures how well two images depict the same 3D scene: it reconstructs
their shared geometry with MASt3R and scores feature dissimilarity in the
overlapping region, so *lower* is more consistent. Following the paper, we drop
the source-pose frame (``frame_00``) and average over the 13 consecutive frame
pairs of each generated trajectory.

The input is the ``gen/`` folder written by ``evaluate.py`` (PNGs named
``<scene>_frame_NN.png``), so this runs *after* generation and needs no access to
the model -- the same split the paper uses.

MEt3R brings its own dependency set -- the ``met3r`` package with its MASt3R
submodule, FeatUp, and a source build of ``pytorch3d``, plus ``timm==0.4.12`` --
which conflicts with this repository's dependencies. Install it in a separate
environment or container and run this script from there::

    git clone https://github.com/mohammadasim98/met3r.git
    cd met3r && pip install -r requirements.txt
    python /path/to/ovie/evaluate_met3r.py \\
        --gen-dir evaluation/<run>/gen --out results/ovie_ft_re10k.json

The MASt3R and DINO weights are downloaded on first run.

Usage::

    uv run python evaluate_met3r.py --gen-dir <eval output>/gen --out met3r.json

[1] Asim et al., "MEt3R: Measuring Multi-View Consistency in Generated Images",
CVPR 2025.
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import torch
from PIL import Image

_FRAME_RE = re.compile(r"(.+)_frame_(\d+)\.png$")


def load_scene_frames(gen_dir):
    """Group ``<scene>_frame_NN.png`` files into ordered per-scene frame lists."""
    scenes = {}
    for path in glob.glob(os.path.join(gen_dir, "*.png")):
        match = _FRAME_RE.search(os.path.basename(path))
        if not match:
            continue
        scenes.setdefault(match.group(1), []).append((int(match.group(2)), path))
    for scene_id in scenes:
        scenes[scene_id].sort()
        scenes[scene_id] = [p for _, p in scenes[scene_id]]
    return scenes


def to_tensor(path, size):
    """Load an image as a (3, size, size) float tensor in [-1, 1]."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1) * 2 - 1


def build_pairs(scenes, skip_first_frame, pairing):
    pairs = []
    for frames in scenes.values():
        if skip_first_frame and len(frames) > 1:
            frames = frames[1:]
        if len(frames) < 2:
            continue
        if pairing == "consecutive":
            pairs += [(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
        else:
            pairs += [(frames[0], frames[i]) for i in range(1, len(frames))]
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="MEt3R multi-view consistency over generated frames."
    )
    parser.add_argument(
        "--gen-dir",
        required=True,
        help="Folder of generated PNGs named <scene>_frame_NN.png "
        "(the 'gen' subfolder written by evaluate.py).",
    )
    parser.add_argument("--out", required=True, help="Where to write the JSON result.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Resolution MEt3R scores at (the paper uses 256).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-scenes", type=int, default=None, help="Limit the number of scenes."
    )
    parser.add_argument(
        "--skip-first-frame",
        action="store_true",
        help="Drop frame_00 (the source-pose reconstruction) before pairing, as "
        "the paper does.",
    )
    parser.add_argument(
        "--pairing",
        choices=["consecutive", "source"],
        default="consecutive",
        help="'consecutive' pairs adjacent frames along the trajectory, 'source' "
        "pairs the first frame against every other.",
    )
    args = parser.parse_args()

    try:
        from met3r import MEt3R
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "MEt3R is not available in this environment. It needs its own "
            "environment or container (met3r + MASt3R + FeatUp + pytorch3d); "
            "see the module docstring."
        ) from exc

    metric = (
        MEt3R(
            img_size=args.image_size,
            use_norm=True,
            backbone="mast3r",
            feature_backbone="dino16",
            upsampler="featup",
            distance="cosine",
            freeze=True,
        )
        .cuda()
        .eval()
    )

    scenes = load_scene_frames(args.gen_dir)
    scene_ids = sorted(scenes)
    if args.max_scenes:
        scene_ids = scene_ids[: args.max_scenes]
        scenes = {k: scenes[k] for k in scene_ids}

    pairs = build_pairs(scenes, args.skip_first_frame, args.pairing)
    if not pairs:
        raise RuntimeError(
            f"No pairs built from {args.gen_dir} (found {len(scene_ids)} scenes)"
        )
    print(f"{len(scene_ids)} scenes, {len(pairs)} pairs", flush=True)

    scores = []
    with torch.no_grad():
        for b in range(0, len(pairs), args.batch_size):
            chunk = pairs[b : b + args.batch_size]
            images = torch.stack(
                [
                    torch.stack(
                        [to_tensor(a, args.image_size), to_tensor(c, args.image_size)]
                    )
                    for a, c in chunk
                ]
            ).cuda()
            score, *_ = metric(
                images=images,
                return_overlap_mask=False,
                return_score_map=False,
                return_projections=False,
            )
            scores.append(score.detach().float().cpu())
            if (b // args.batch_size) % 20 == 0:
                print(f"  {b + len(chunk)}/{len(pairs)} pairs", flush=True)

    scores = torch.cat(scores).numpy()
    result = {
        "gen_dir": args.gen_dir,
        "n_scenes": len(scene_ids),
        "n_pairs": len(pairs),
        "pairing": args.pairing,
        "skip_first_frame": args.skip_first_frame,
        "met3r_mean": float(scores.mean()),
        "met3r_std": float(scores.std()),
        "note": "lower = more multi-view consistent",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
