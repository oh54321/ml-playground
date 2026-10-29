"""Train a 2048 agent: python -m games.rl_2048.train_2048."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
from typing import TextIO

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from algorithms.ppo import PPO, PPOConfig
from core.models.conv_actor_critic import ConvActorCriticConfig, ConvActorCriticModel
from core.models.d4_conv_actor_critic import D4ConvActorCriticModel
from games.rl_2048.environment.game_2048 import Game2048, ObservationEncoding
from games.rl_2048.environment.symmetry_2048 import SQUARE_SYMMETRY_SPECS
from games.rl_2048.training.checkpointing import save_checkpoint, warm_start_from_checkpoint
from games.rl_2048.training.config import TrainingConfig
from games.rl_2048.training.evaluation import evaluate_games


# Experiment settings: edit these values, or override them on the command line.
TRAINING_CONFIG = TrainingConfig(
    height=4,
    width=4,
    architecture="d4",
    observation_encoding=ObservationEncoding.NUMERIC_ONEHOT,
    run_name="ppo_2048",
    learning_rate=3e-4,
    scheduler_decay=0.999,
    scheduler_decay_until=6000,  # Decay through this iteration, then hold the rate.
    seed=12345,
    outer_iterations=10000,
    warm_start_from=None,
    num_parallel_games=128,
    steps_per_game_per_rollout=1024,
    evaluation_interval=50,
    evaluation_games=500,
    evaluation_seed=67890,
    checkpoint_interval=500,
    tensorboard=False,
    compile_model=False,
)
DEVICE = None  # Automatically choose CUDA, MPS, or CPU.

# Model widths by architecture, followed by settings shared by both models.
MODEL_CHANNELS = {"cnn": 32, "d4": 64}
MODEL_SETTINGS = {
    "blocks": 4,
    "head_channels": 32,
    "kernel_size": 3,
    "block_layers": 2,
}

PPO_SETTINGS = {
    "gamma": 1.0,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "value_coefficient": 0.003,
    "entropy_coefficient": 0.03,
    "min_update_epochs": 2,
    "max_update_epochs": 8,
    "target_kl": 0.02,
    "max_episode_steps": 2**20,
    "mini_batch_size": 16384,
    "reward_multiplier": 0.01,
    "denominator_epsilon": 1e-8,
}


def read_config() -> tuple[TrainingConfig, ConvActorCriticConfig, PPOConfig, str]:
    # Optional command-line arguments override the experiment settings above.
    parser = argparse.ArgumentParser(description="Train the 2048 PPO agent")
    parser.add_argument("--warm-start-from", type=Path)
    parser.add_argument("--architecture", choices=("cnn", "d4"))
    parser.add_argument("--observation-encoding", choices=("numeric", "numeric_onehot"))
    parser.add_argument("--channels", type=int)
    parser.add_argument("--kernel-size", type=int)
    parser.add_argument("--block-layers", type=int)
    parser.add_argument("--scheduler-decay", type=float)
    parser.add_argument("--scheduler-decay-until", type=int,
                        help="last iteration to decay the learning rate, inclusive")
    for name in ("height", "width", "outer-iterations", "num-parallel-games", "steps-per-game-per-rollout",
                 "evaluation-interval", "evaluation-games", "checkpoint-interval"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--tensorboard", action="store_true", default=None)
    parser.add_argument("--compile-model", action="store_true", default=None,
                        help="compile actor and critic calls on CUDA")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    overrides = vars(parser.parse_args())

    device = overrides.pop("device") or DEVICE or (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    channels = overrides.pop("channels")
    model_settings = MODEL_SETTINGS.copy()
    for name in ("kernel_size", "block_layers"):
        value = overrides.pop(name)
        if value is not None:
            model_settings[name] = value
    config = replace(TRAINING_CONFIG, **{key: value for key, value in overrides.items() if value is not None})
    if channels is None:
        channels = MODEL_CHANNELS[config.architecture]
    model_config = ConvActorCriticConfig(
        in_channels=config.observation_encoding.in_channels(config.height, config.width),
        height=config.height, width=config.width, output_dim=4, n_channels=channels,
        **model_settings,
    )
    if config.compile_model and device != "cuda":
        raise RuntimeError("--compile-model requires a CUDA device")
    ppo_config = PPOConfig(
        **PPO_SETTINGS,
        num_parallel_games=config.num_parallel_games,
        steps_per_game_per_rollout=config.steps_per_game_per_rollout,
        seed=config.seed,
    )
    return config, model_config, ppo_config, device


def create_run_directory(config: TrainingConfig, model_config: ConvActorCriticConfig,
                         ppo_config: PPOConfig, device: str) -> Path:
    output = Path("artifacts/ppo_2048") / f"{config.run_name}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps({
        "training": asdict(config), "model": asdict(model_config), "ppo": asdict(ppo_config),
        "device": device,
    }, indent=2, default=str) + "\n")
    print(f"Training on {device}; saving to {output}", flush=True)
    print(f"Model: {config.architecture}, {config.observation_encoding.value}, {model_config.n_channels} channels", flush=True)
    return output


def log_iteration(metrics: dict[str, float | int], metrics_file: TextIO, writer) -> None:
    iteration = metrics["iteration"]
    metrics_file.write(json.dumps(metrics) + "\n")
    metrics_file.flush()
    if writer is not None:
        for key, value in metrics.items():
            if key != "iteration":
                writer.add_scalar(key, value, iteration)
        writer.flush()
    message = f"Iteration {iteration}: {metrics['ppo/update_epochs']} epochs, policy loss {metrics['ppo/policy_loss']:.4f}"
    if "evaluation/mean_score" in metrics:
        message += f", score {metrics['evaluation/mean_score']:.0f}, win rate {metrics['evaluation/win_rate']:.1%}"
    print(message, flush=True)


def main() -> None:
    config, model_config, ppo_config, device = read_config()
    torch.manual_seed(config.seed)
    if config.architecture == "cnn":
        model = ConvActorCriticModel(model_config)
    else:
        action_permutations = {
            spec.transform: spec.action_permutation for spec in SQUARE_SYMMETRY_SPECS
        }
        model = D4ConvActorCriticModel(model_config, action_permutations)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=config.scheduler_decay)
    first_iteration = 1
    if config.warm_start_from is not None:
        first_iteration = warm_start_from_checkpoint(
            config.warm_start_from, model, optimizer, scheduler, ppo_config,
            observation_encoding=config.observation_encoding,
        )
    if first_iteration > config.outer_iterations:
        raise ValueError("Checkpoint has already completed the requested iterations")
    if config.compile_model:
        model.compile_trunks()
        print("Actor/critic compilation enabled; kernels compile on first use.", flush=True)
    ppo = PPO(model, optimizer, ppo_config, lambda seed: Game2048(
        seed=seed, height=config.height, width=config.width,
        observation_encoding=config.observation_encoding,
    ))
    output = create_run_directory(config, model_config, ppo_config, device)
    writer = None
    if config.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=output / "tensorboard", flush_secs=5)
    best_score = float("-inf")
    try:
        with (output / "metrics.jsonl").open("a") as metrics_file:
            for iteration in range(first_iteration, config.outer_iterations + 1):
                started = perf_counter()
                epochs = ppo.run_outer_iteration()
                if config.scheduler_decay_until is None or iteration <= config.scheduler_decay_until:
                    scheduler.step()
                metrics = {
                    "iteration": iteration,
                    "training/learning_rate": optimizer.param_groups[0]["lr"],
                    "ppo/update_epochs": len(epochs),
                    **{f"ppo/{key}": value for key, value in asdict(epochs[-1]).items()},
                }
                if iteration % config.evaluation_interval == 0:
                    results = evaluate_games(ppo, [
                        config.evaluation_seed + index for index in range(1, config.evaluation_games + 1)
                    ], height=config.height, width=config.width,
                       observation_encoding=config.observation_encoding)
                    metrics.update({f"evaluation/{key}": value for key, value in results.metrics().items()})
                    (output / f"evaluation_{iteration:04d}.json").write_text(json.dumps(asdict(results)) + "\n")
                    save_checkpoint(output / "latest.pt", model, optimizer, scheduler, ppo_config, iteration,
                                    observation_encoding=config.observation_encoding)
                    if metrics["evaluation/mean_score"] > best_score:
                        best_score = metrics["evaluation/mean_score"]
                        save_checkpoint(output / "best.pt", model, optimizer, scheduler, ppo_config, iteration,
                                        observation_encoding=config.observation_encoding)
                if iteration % config.checkpoint_interval == 0:
                    save_checkpoint(output / f"checkpoint_{iteration:04d}.pt", model, optimizer, scheduler, ppo_config, iteration,
                                    observation_encoding=config.observation_encoding)
                metrics["training/iteration_seconds"] = perf_counter() - started
                log_iteration(metrics, metrics_file, writer)
        save_checkpoint(output / "final.pt", model, optimizer, scheduler, ppo_config, config.outer_iterations,
                        observation_encoding=config.observation_encoding)
    except KeyboardInterrupt:
        print("Interrupted; warm-start from the most recent saved checkpoint.")
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
