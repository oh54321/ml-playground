"""Export a clean checkpoint's policy for the browser demo."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from games.rl_2048.demo import load_policy
from games.rl_2048.environment.game_2048 import Game2048


class Policy(nn.Module):
    """The browser selects actions; it does not need the critic or optimizer."""

    def __init__(self, model):
        super().__init__()
        self.actor_trunk = model.actor_trunk
        self.action_head = model.action_head

    def forward(self, observation):
        return self.action_head(self.actor_trunk(observation))


def strip_export_metadata(message) -> None:
    # PyTorch embeds source paths in ONNX debug metadata. Keep only the graph.
    for field, value in list(message.ListFields()):
        if field.name in ("doc_string", "metadata_props"):
            message.ClearField(field.name)
        elif field.message_type is not None:
            for child in value if field.is_repeated else [value]:
                strip_export_metadata(child)


@torch.no_grad()
def export_policy(checkpoint: Path, output: Path) -> dict:
    model, encoding = load_policy(checkpoint)
    policy = Policy(model).eval()
    config = model.config
    game = Game2048(seed=12345, height=config.height, width=config.width,
                    observation_encoding=encoding)
    sample = game.state.to_tensor().unsqueeze(0)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temporary:
        path = Path(temporary) / "policy.onnx"
        with model.prepared_inference():
            torch.onnx.export(policy, (sample,), path, input_names=["observation"],
                              output_names=["logits"], opset_version=18,
                              dynamo=True, external_data=False)
        graph = onnx.load(path)
        strip_export_metadata(graph)
        onnx.checker.check_model(graph)
        onnx.save(graph, path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(12345)
        max_error = 0.0
        with model.prepared_inference():
            for _ in range(128):
                observation = game.state.to_tensor().unsqueeze(0)
                expected = policy(observation).numpy()
                actual = session.run(["logits"], {"observation": observation.numpy()})[0]
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
                max_error = max(max_error, float(np.abs(actual - expected).max()))
                legal = game.legal_actions()
                if legal:
                    assert legal[int(actual[0, legal].argmax())] == legal[int(expected[0, legal].argmax())]
                    game.step(int(rng.choice(legal)))
                if game.done:
                    game.reset()
        metadata = {"height": config.height, "width": config.width,
                    "in_channels": config.in_channels, "observation_encoding": encoding.value,
                    "architecture": model.architecture,
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        path.replace(output / "policy.onnx")
        (output / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"boards_checked": 128, "max_logit_error": max_error,
            "bytes": (output / "policy.onnx").stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("demo_web"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(export_policy(args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
