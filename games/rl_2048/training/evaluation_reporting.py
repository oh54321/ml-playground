"""Summarize evaluation runs and persist their reports and checkpoints."""

from statistics import median

import torch

from algorithms.ppo import PPO, PPOConfig
from core.models.actor_critic import ActorCriticModel
from core.models.conv_actor_critic import ConvActorCriticConfig
from games.rl_2048.training.checkpointing import save_checkpoint
from games.rl_2048.training.evaluation import EvaluationRun, evaluate_games
from games.rl_2048.utils.plot_training_diagnostics import save_evaluation_plots
from games.rl_2048.utils.reporting import print_timestamp


def write_evaluation_scalars(writer, evaluation: EvaluationRun, iteration: int) -> None:
    """Write the completed evaluation's metrics to a TensorBoard writer."""
    for metric, values in evaluation.history.items():
        if values:
            writer.add_scalar(f"evaluation/{metric}", values[-1], iteration)


def evaluate_and_report(
    ppo: PPO,
    model: ActorCriticModel,
    optimizer: torch.optim.Optimizer,
    ppo_config: PPOConfig,
    model_config: ConvActorCriticConfig,
    outer_iteration: int,
    evaluation: EvaluationRun,
) -> None:
    """Run evaluation, record its summary, update plots, and save the best model."""
    print_timestamp("Starting evaluation games.", ppo_config.print_timestamps)
    was_training = model.training
    model.eval()
    try:
        evaluation_dir = evaluation.output_dir / f"outer_iteration_{outer_iteration:04d}"
        seeds = [
            evaluation.seed_base + game_number
            for game_number in range(1, evaluation.num_games + 1)
        ]
        result = evaluate_games(
            ppo,
            seeds,
            save_traces=evaluation.save_traces,
            trace_output_dir=evaluation_dir if evaluation.save_traces else None,
        )
        scores = sorted(result.scores)
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        if evaluation.save_traces:
            assert result.worst_game.trace is not None
            assert result.best_game.trace is not None
            (evaluation_dir / "worst_game.txt").write_text(
                result.worst_game.trace,
                encoding="utf-8",
            )
            (evaluation_dir / "best_game.txt").write_text(
                result.best_game.trace,
                encoding="utf-8",
            )

        mean_score = sum(scores) / len(scores)
        median_score = median(scores)
        mean_moves = sum(result.moves) / len(result.moves)
        win_rate = result.win_rate
        evaluation.history["mean"].append(mean_score)
        evaluation.history["median"].append(median_score)
        evaluation.history["best"].append(result.best_game.score)
        evaluation.history["worst"].append(result.worst_game.score)
        evaluation.history["mean_moves"].append(mean_moves)
        evaluation.history["win_rate"].append(win_rate)
        evaluation.iterations.append(outer_iteration)
        save_evaluation_plots(
            evaluation.history,
            evaluation.iterations,
            evaluation.output_dir,
        )

        if mean_score > evaluation.best_mean_score:
            evaluation.best_mean_score = mean_score
            save_checkpoint(
                evaluation.output_dir / "best.pt",
                model,
                optimizer,
                ppo_config,
                model_config,
                outer_iteration,
                mean_score,
                median_score,
                mean_moves,
                evaluation.symmetry_augmentation,
                evaluation.exploration_schedule,
                win_rate_2048=win_rate,
            )
            print(f"  Saved new best checkpoint: {evaluation.output_dir / 'best.pt'}")

        save_checkpoint(
            evaluation.output_dir / "latest.pt",
            model,
            optimizer,
            ppo_config,
            model_config,
            outer_iteration,
            mean_score,
            median_score,
            mean_moves,
            evaluation.symmetry_augmentation,
            evaluation.exploration_schedule,
            win_rate_2048=win_rate,
        )

        traces_location = str(evaluation_dir) if evaluation.save_traces else "disabled"
        summary = (
            f"Outer iteration {outer_iteration}\n"
            f"Mean score: {mean_score:.1f}\n"
            f"Median score: {median_score:.1f}\n"
            f"Mean moves: {mean_moves:.1f}\n"
            f"Win rate (2048+): {win_rate:.1%} "
            f"({sum(tile >= 2048 for tile in result.max_tiles)}/{len(result.max_tiles)} games)\n"
            f"Raw scores: {scores}\n"
            f"Best game: score {result.best_game.score}, steps {result.best_game.steps}, "
            f"seed {result.best_game.seed}\n"
            f"Worst game: score {result.worst_game.score}, steps {result.worst_game.steps}, "
            f"seed {result.worst_game.seed}\n"
            f"Traces: {traces_location}\n\n"
        )
        with (evaluation.output_dir / "summary.txt").open("a", encoding="utf-8") as summary_file:
            summary_file.write(summary)
        print(
            f"  Evaluation mean score: {mean_score:.1f}; "
            f"median score: {median_score:.1f}; mean moves: {mean_moves:.1f}; "
            f"win rate (2048+): {win_rate:.1%}"
        )
        print(f"  Evaluation raw scores: {scores}")
        print(
            f"  Best game: {result.best_game.score} points; "
            f"worst game: {result.worst_game.score} points ({evaluation_dir})"
        )
        print(
            f"  Evaluation traces: {traces_location}; "
            f"run summary: {evaluation.output_dir / 'summary.txt'}"
        )
        print_timestamp("Evaluation games complete.", ppo_config.print_timestamps)
    finally:
        model.train(was_training)
