"""Train a 2048 agent with PPO."""

from dataclasses import asdict
from statistics import fmean
from time import perf_counter

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from algorithms.ppo import PPOConfig
from core.models.conv_actor_critic import ConvActorCriticConfig
from core.models.d4_conv_actor_critic import D4ConvActorCriticModel
from games.rl_2048.environment.symmetry_2048 import SQUARE_SYMMETRY_SPECS
from games.rl_2048.training.checkpointing import save_checkpoint
from games.rl_2048.training.config import LinearExplorationSchedule, TrainingConfig
from games.rl_2048.training.evaluation_reporting import evaluate_and_report
from games.rl_2048.training.options import read_training_options
from games.rl_2048.training.setup import prepare_training_run, select_device
from games.rl_2048.utils.plot_training_diagnostics import save_training_diagnostic_plots
from games.rl_2048.utils.reporting import (
    report_evaluation_skipped,
    report_run_start,
    report_training_iteration,
    report_training_iteration_start,
    tee_stdout,
)


# Experiment settings. Edit these defaults for a new run; command-line options
# in training/options.py can override run length, rollout size, evaluation,
# and schedule settings.
MODEL_CONFIG = ConvActorCriticConfig(
    in_channels=1,
    n_channels=64,
    blocks=4,
    height=4,
    width=4,
    output_dim=4,
    head_channels=32,
    kernel_size=3,
    block_layers=2,
    include_groupnorm=True,
    shared_trunk=False,
)

TRAINING_CONFIG = TrainingConfig(
    run_name="d4_group_convolutions_140k",
    learning_rate=3e-4,
    seed=12345,
    outer_iterations=1000,  # Last outer iteration to run, inclusive.
    warm_start_from=None,
    num_parallel_games=32,
    steps_per_game_per_rollout=1024,
    symmetry_augmentation="step",
    exploration_schedule=LinearExplorationSchedule(
        start_probability=0.05,
        end_probability=0.0,
        end_iteration=1000,
    ),
    evaluation_interval=10,
    evaluation_games=100,
    evaluation_seed=67890,
    save_traces=False,
    scheduler_decay=0.999,
    # Keep iteration 1000 when extending the run; the current final iteration
    # is already written as final.pt.
    saved_checkpoints=(1000,),
)

PPO_SETTINGS = {
    "gamma": 1.0,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "value_coefficient": 0.003,
    "entropy_coefficient": 0.01,
    "min_update_epochs": 1,
    "max_update_epochs": 8,
    "target_kl": 0.02,
    "max_episode_steps": 2**20,
    "mini_batch_size": 4096,
    "reward_multiplier": 0.01,
    "value_transform": "identity",
    "denominator_epsilon": 1e-8,
    "gradient_clip_norm": None,
    "print_moves": False,
    "print_timestamps": True,
}


def main() -> None:
    config = read_training_options(TRAINING_CONFIG)
    ppo_config = PPOConfig(
        **PPO_SETTINGS,
        num_parallel_games=config.num_parallel_games,
        steps_per_game_per_rollout=config.steps_per_game_per_rollout,
        seed=config.seed,
    )
    device = select_device()
    torch.manual_seed(config.seed)
    action_permutations = {
        spec.transform: spec.action_permutation
        for spec in SQUARE_SYMMETRY_SPECS
    }
    model = D4ConvActorCriticModel(MODEL_CONFIG, action_permutations).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=config.scheduler_decay)
    run = prepare_training_run(
        config,
        ppo_config,
        device,
        model,
        MODEL_CONFIG,
        optimizer,
        scheduler,
    )
    saved_checkpoints = set(config.saved_checkpoints) - {run.last_iteration}

    writer = None
    if config.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError(
                "TensorBoard logging was requested but its dependency is missing. "
                "Install it with: python -m pip install tensorboard"
            ) from error
        writer = SummaryWriter(log_dir=run.output_dir / "tensorboard", flush_secs=5)

    interrupted = False
    try:
        with tee_stdout(run.raw_summary_path):
            report_run_start(
                str(run.device),
                type(run.model).__name__,
                asdict(run.model_config),
                config.symmetry_augmentation,
                run.exploration_data,
                run.parameter_count,
                config.warm_start_from,
                run.first_iteration,
            )

            # PPO outer loop: collect a rollout, optimize the policy/value model,
            # advance the learning-rate schedule, then evaluate on schedule.
            for iteration in range(run.first_iteration, run.last_iteration + 1):
                iteration_started = perf_counter()
                probability = run.exploration.probability(
                    iteration,
                    int(run.exploration_data["start_iteration"]),
                    run.exploration.end_iteration or run.last_iteration,
                )
                run.ppo.set_exploration_probability(probability)
                report_training_iteration_start(iteration, probability)

                epoch_stats = run.ppo.run_outer_iteration()
                run.scheduler.step()
                report_training_iteration(
                    run.optimizer.param_groups[0]["lr"],
                    epoch_stats,
                    run.ppo.last_gradient_statistics,
                    run.ppo.last_outer_iteration_policy_kl,
                )

                if iteration % config.evaluation_interval == 0:
                    evaluate_and_report(
                        run.ppo,
                        run.model,
                        run.optimizer,
                        ppo_config,
                        run.model_config,
                        iteration,
                        run.evaluation,
                    )
                else:
                    report_evaluation_skipped(iteration, config.evaluation_interval)

                # Preserve selected model/optimizer states even when no
                # evaluation is scheduled for that iteration.
                if iteration in saved_checkpoints:
                    evaluation_index = next(
                        (
                            index
                            for index, evaluated_iteration in enumerate(
                                run.evaluation.iterations
                            )
                            if evaluated_iteration == iteration
                        ),
                        None,
                    )
                    evaluation_metrics = (
                        (
                            run.evaluation.history["mean"][evaluation_index],
                            run.evaluation.history["median"][evaluation_index],
                            run.evaluation.history["mean_moves"][evaluation_index],
                        )
                        if evaluation_index is not None
                        else (None, None, None)
                    )
                    save_checkpoint(
                        run.output_dir / f"checkpoint_{iteration:04d}.pt",
                        run.model,
                        run.optimizer,
                        ppo_config,
                        run.model_config,
                        iteration,
                        *evaluation_metrics,
                        config.symmetry_augmentation,
                        run.exploration_data,
                    )

                if writer is not None:
                    elapsed = perf_counter() - iteration_started
                    writer.add_scalar("training/iteration_seconds", elapsed, iteration)
                    writer.add_scalar(
                        "training/learning_rate",
                        run.optimizer.param_groups[0]["lr"],
                        iteration,
                    )
                    writer.add_scalar(
                        "training/exploration_probability", probability, iteration
                    )
                    if epoch_stats:
                        for field in epoch_stats[0].__dataclass_fields__:
                            writer.add_scalar(
                                f"ppo/{field}",
                                fmean(getattr(stats, field) for stats in epoch_stats),
                                iteration,
                            )
                    gradients = run.ppo.last_gradient_statistics
                    if gradients is not None:
                        for field in gradients.__dataclass_fields__:
                            value = getattr(gradients, field)
                            if isinstance(value, (float, int)) and not isinstance(value, bool):
                                writer.add_scalar(
                                    f"gradients/{field}", value, iteration
                                )
                    policy_kl = run.ppo.last_outer_iteration_policy_kl
                    if policy_kl is not None:
                        writer.add_scalar(
                            "ppo/policy_kl_between_iterations", policy_kl, iteration
                        )
                    if run.evaluation.iterations and run.evaluation.iterations[-1] == iteration:
                        for metric, values in run.evaluation.history.items():
                            writer.add_scalar(
                                f"evaluation/{metric}", values[-1], iteration
                            )
                    writer.flush()
    except KeyboardInterrupt:
        interrupted = True
        print(
            "Training interrupted. Any completed evaluations are saved in latest.pt; "
            "iterations since that checkpoint will need to be repeated."
        )
    finally:
        if writer is not None:
            writer.close()

    if not interrupted:
        final_iteration = run.last_iteration
        if (
            run.evaluation.iterations
            and run.evaluation.iterations[-1] == final_iteration
        ):
            final_index = run.evaluation.iterations.index(final_iteration)
            final_evaluation_metrics = (
                run.evaluation.history["mean"][final_index],
                run.evaluation.history["median"][final_index],
                run.evaluation.history["mean_moves"][final_index],
            )
        else:
            final_evaluation_metrics = (None, None, None)
        save_checkpoint(
            run.output_dir / "final.pt",
            run.model,
            run.optimizer,
            ppo_config,
            run.model_config,
            final_iteration,
            *final_evaluation_metrics,
            config.symmetry_augmentation,
            run.exploration_data,
        )
    save_training_diagnostic_plots(
        run.raw_summary_path,
        run.output_dir,
        run.run_summary_path,
    )


if __name__ == "__main__":
    main()
