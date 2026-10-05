"""Blend a verified province-head correction into its source recognizer."""

import argparse

import torch

from lpr.runtime import load_checkpoint, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--corrected", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 < args.alpha <= 1:
        parser.error("alpha must be greater than zero and at most one")
    base, _ = load_checkpoint(args.base, "recognizer", "cpu")
    corrected, _ = load_checkpoint(args.corrected, "recognizer", "cpu")
    base_state = base.state_dict()
    corrected_state = corrected.state_dict()
    if any(not torch.equal(base_state[name], value) for name, value in corrected_state.items() if not name.startswith("heads.0.")):
        raise ValueError("checkpoints differ outside the province head")
    with torch.no_grad():
        for base_parameter, corrected_parameter in zip(base.heads[0].parameters(), corrected.heads[0].parameters()):
            base_parameter.lerp_(corrected_parameter, args.alpha)
    save_checkpoint(args.output, base, "recognizer", province_blend_alpha=args.alpha)
    print(f"saved province-head blend (alpha={args.alpha}) to {args.output}")


if __name__ == "__main__":
    main()
