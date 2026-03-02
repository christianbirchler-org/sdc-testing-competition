"""Export a trained TransformerEncoder checkpoint to ONNX format."""

from __future__ import annotations

import argparse

import torch

from model import TransformerEncoder
from utils import load_yaml, resolve_device


class NormalizedModel(torch.nn.Module):
    """Wraps a TransformerEncoder with z-score normalization of inputs."""

    def __init__(self, model, normalizer):
        super().__init__()
        self.model = model
        self.register_buffer("seq_mean", torch.tensor(normalizer["seq_mean"]))
        self.register_buffer("seq_std", torch.tensor(normalizer["seq_std"]))
        self.register_buffer("glob_mean", torch.tensor(normalizer["glob_mean"]))
        self.register_buffer("glob_std", torch.tensor(normalizer["glob_std"]))

    def forward(self, seq, valid_mask, glob):
        seq = (seq - self.seq_mean) / self.seq_std
        glob = (glob - self.glob_mean) / self.glob_std
        return self.model(seq, valid_mask, glob)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TransformerEncoder to ONNX.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="../rte4sdc.onnx")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = resolve_device(str(cfg["device"]))
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = TransformerEncoder(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    wrapped = NormalizedModel(model, ckpt["normalizer"]).to(device)
    wrapped.eval()

    d_seq = int(cfg["model"]["d_seq"])
    d_global = int(cfg["model"]["d_global"])

    # Dummy inputs for tracing
    seq = torch.randn(1, 16, d_seq, device=device)
    valid_mask = torch.ones(1, 16, dtype=torch.bool, device=device)
    glob = torch.randn(1, d_global, device=device)

    torch.onnx.export(
        wrapped,
        (seq, valid_mask, glob),
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["seq", "valid_mask", "glob"],
        output_names=["score"],
        dynamic_axes={
            "seq": {1: "tokens"},
            "valid_mask": {1: "tokens"},
        },
    )
    print(f"ONNX exported to: {args.output}")


if __name__ == "__main__":
    main()
