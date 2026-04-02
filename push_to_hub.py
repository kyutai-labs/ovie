"""
Upload OVIE weights to the Hugging Face Hub.

Usage:
    uv run push_to_hub.py --ckpt assets/ovie.pt --repo kyutai/ovie
    uv run push_to_hub.py --ckpt assets/ovie.pt --repo kyutai/ovie --tag v1.0

The script extracts the EMA weights from the checkpoint, loads them into the
model, and pushes the model + config to the specified Hub repository.
If --tag is provided, a git tag is created on the resulting commit so that
from_pretrained(..., revision="v1.0") pins to that exact upload.
"""

import argparse
import torch
from huggingface_hub import HfApi
from models.models import OVIEModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="assets/ovie.pt",
        help="Path to the OVIE checkpoint (.pt file containing 'ema' key)",
    )
    parser.add_argument(
        "--repo",
        default="kyutai/ovie",
        help="Hugging Face Hub repo id to push to (e.g. org/model-name)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Git tag to create after pushing (e.g. v1.0). Lets users pin "
        "from_pretrained(..., revision='v1.0') to this exact upload.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hub repository as private",
    )
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "ema" not in ckpt:
        raise KeyError(
            f"Expected an 'ema' key in the checkpoint, got: {list(ckpt.keys())}"
        )

    print("Instantiating OVIE-B model ...")
    model = OVIEModel(
        image_size=256,
        in_channels=3,
        out_channels=3,
        ch=128,
        ch_mult=[1, 2, 4],
        num_res_blocks=2,
        vit_hidden_size=768,
        vit_depth=12,
        vit_patch_size=1,
        vit_num_heads=12,
        vit_use_qknorm=False,
        vit_use_swiglu=True,
        vit_use_rope=False,
        vit_use_rmsnorm=True,
        vit_wo_shift=False,
        vit_use_checkpoint=False,
        in_cam_params=7,
        final_sigmoid_activation=True,
        inject_noise_in_decoder=False,
    )

    print("Loading EMA weights ...")
    model.load_state_dict(ckpt["ema"])
    model.eval()

    print(f"Pushing to Hub: {args.repo} ...")
    commit_url = model.push_to_hub(args.repo, private=args.private)
    print(f"Pushed: {commit_url}")

    if args.tag:
        api = HfApi()
        # Extract the commit hash from the returned URL
        commit_hash = commit_url.split("/")[-1] if commit_url else None
        api.create_tag(
            repo_id=args.repo,
            tag=args.tag,
            tag_message=f"Release {args.tag}",
            revision=commit_hash,
        )
        print(f"Tagged commit {commit_hash} as '{args.tag}'")
        print(
            f'Load with: OVIEModel.from_pretrained("{args.repo}", revision="{args.tag}")'
        )
    else:
        print(f'Load with: OVIEModel.from_pretrained("{args.repo}")')


if __name__ == "__main__":
    main()
