import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import categorical

from core.game import Game, State, StepResult
from games.rl_2048.agent import ActorCriticOutput, ResNet


@dataclass(frozen=True)
class PolicyRolloutInformation:
    state: State
    action: int
    next_state: State
    log_prob: float
    value: float
    reward: float
    done: bool # environment ended naturally
    truncated: bool # rollout was truncated due to max_episode_steps
    legal_actions_mask: torch.Tensor
    advantage: float = None


@dataclass(frozen=True)
class ActionDecision:
    action: int
    log_prob: float
    value: float
    illegal_actions_mask: torch.Tensor


@dataclass(frozen=True)
class PPOConfig:
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    value_coefficient: float
    entropy_coefficient: float
    update_epochs: int
    num_parallel_games: int
    steps_per_game_per_rollout: int
    max_episode_steps: int
    mini_batch_size: int
    reward_multiplier: float
    value_transform: Literal["identity", "signed_log1p"]
    denominator_epsilon: float
    gradient_clip_norm: float | None
    seed: int
    print_moves: bool
    print_timestamps: bool = True


@dataclass(frozen=True)
class BatchObjectives:
    policy: torch.Tensor
    value: torch.Tensor
    negative_entropy: torch.Tensor
    approximate_kl: torch.Tensor
    outside_clip_range: torch.Tensor
    value_error: torch.Tensor
    value_target: torch.Tensor


@dataclass(frozen=True)
class EpochStatistics:
    policy_loss: float
    value_loss: float
    negative_entropy: float
    approximate_kl: float
    clip_fraction: float
    explained_variance: float


class PPO():
    def __init__(
        self,
        model: ResNet,
        optimizer: torch.optim.Optimizer,
        config: PPOConfig,
        game_factory: Callable[[int], Game],
    ):
        if config.value_transform not in ("identity", "signed_log1p"):
            raise ValueError(f"Unsupported value transform: {config.value_transform}")
        if config.num_parallel_games <= 0:
            raise ValueError("num_parallel_games must be positive")
        if config.steps_per_game_per_rollout <= 0:
            raise ValueError("steps_per_game_per_rollout must be positive")
        if config.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if config.gradient_clip_norm is not None and config.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when provided")

        torch.manual_seed(config.seed)

        self.config = config
        self.model: ResNet = model
        self.optimizer = optimizer
        self.rng = np.random.default_rng(config.seed)

        seeds = self.rng.integers(
            0,
            2**32,
            size=config.num_parallel_games,
            dtype=np.uint32,
        )
        self.games: List[Game] = [
            game_factory(int(seed))
            for seed in seeds
        ]
        self.game_steps = [0 for _ in self.games]

    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def _state_batch_tensor(self, states: List[State]) -> torch.Tensor:
        """Stack CPU state samples, then transfer the completed batch once."""
        state_batch = torch.stack([state.to_tensor() for state in states], dim=0)
        return state_batch.to(device=self.device(), dtype=self.dtype())

    def _predict_value(self, state: State) -> float:
        actor_critic_output: ActorCriticOutput = self.model(
            self._state_batch_tensor([state])
        )
        return self._untransform_predicted_value(actor_critic_output.value).item()

    def _transform_value_target(self, value: torch.Tensor) -> torch.Tensor:
        """Map raw values to the space predicted by the value head."""
        if self.config.value_transform == "identity":
            return value
        return torch.sign(value) * torch.log1p(torch.abs(value))

    def _untransform_predicted_value(self, transformed_value: torch.Tensor) -> torch.Tensor:
        """Map a value-head output back to raw reward units for GAE."""
        if self.config.value_transform == "identity":
            return transformed_value
        return torch.sign(transformed_value) * torch.expm1(torch.abs(transformed_value))

    def _steps_to_tensor(
        self,
        steps: List[PolicyRolloutInformation],
        field_name: str,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Convert one rollout field into a tensor on the model's device."""
        if len(steps) == 0:
            raise ValueError("Cannot create a tensor from an empty step list")
        values = [getattr(step, field_name) for step in steps]
        if isinstance(values[0], torch.Tensor):
            tensor = torch.stack(values, dim=0)
            if dtype is None:
                return tensor.to(device=self.device())
            return tensor.to(device=self.device(), dtype=dtype)
        return torch.tensor(
            values,
            device=self.device(),
            dtype=dtype if dtype is not None else self.dtype(),
        )

    def _print_timestamp(self, message: str) -> None:
        if self.config.print_timestamps:
            timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            print(f"[{timestamp}] {message}")

    # This requires and validates that all games have at least one legal action.
    # The function will throw if at least one game does not satisfy this
    def generate_next_actions(
        self,
        games: List[Game],
        deterministic: bool = False,
    ) -> List[ActionDecision]:
        if not games:
            raise ValueError("Must evaluate at least one game")
        legal_actions_by_game = [game.legal_actions() for game in games]
        legal_action_counts = torch.tensor(
            [len(legal_actions) for legal_actions in legal_actions_by_game],
            device=self.device(),
            dtype=torch.long,
        )
        if any(len(legal_actions) == 0 for legal_actions in legal_actions_by_game):
            raise ValueError(
                "Every game passed to generate_next_actions must have at least one legal action"
            )
        tensor_state = self._state_batch_tensor(
            [game.current_state() for game in games]
        )
        actor_critic_output: ActorCriticOutput = self.model(tensor_state)
        action_logits = actor_critic_output.action_logits
        values = self._untransform_predicted_value(actor_critic_output.value)
        illegal_actions_mask = torch.ones_like(action_logits, dtype=torch.bool)
        game_rows = torch.repeat_interleave(
            torch.arange(
                len(games),
                device=self.device(),
                dtype=torch.long,
            ),
            legal_action_counts,
        )
        legal_action_columns = torch.tensor(
            [action for actions in legal_actions_by_game for action in actions],
            device=self.device(),
            dtype=torch.long,
        )
        illegal_actions_mask[game_rows, legal_action_columns] = 0
        action_logits = action_logits.masked_fill(illegal_actions_mask, -torch.inf)
        categorical_distribution = categorical.Categorical(logits=action_logits)
        action = (
            action_logits.argmax(dim=-1)
            if deterministic
            else categorical_distribution.sample()
        )
        log_probability = categorical_distribution.log_prob(action)
        action_values = action.tolist()
        log_probability_values = log_probability.tolist()
        value_values = values.tolist()
        return [
            ActionDecision(
                action=int(action_values[i]),
                log_prob=float(log_probability_values[i]),
                value=float(value_values[i]),
                illegal_actions_mask=illegal_actions_mask[i].clone(),
            )
            for i in range(len(games))
        ]

    # Runs games in parallel until we have the desired amount of policy rollouts.
    # Starts games from their stored state from the last episode
    # Continually generates the next state along with its information
    # If there are no legal moves, resets the game under a different seed
    @torch.no_grad()
    def collect_batched_information(self) -> Tuple[List[PolicyRolloutInformation], float, float]:
        rollouts_by_game: List[List[PolicyRolloutInformation]] = [
            [] for _ in range(self.config.num_parallel_games)
        ]
        for _ in range(self.config.steps_per_game_per_rollout):
            states = [game.current_state().copy() for game in self.games]
            decisions: List[ActionDecision] = self.generate_next_actions(self.games)
            if self.config.print_moves:
                for game_index, (game, state, decision) in enumerate(
                    zip(self.games, states, decisions)
                ):
                    legal_action_names = ", ".join(
                        game.action_name(legal_action)
                        for legal_action in game.legal_actions()
                    )
                    print(f"Game {game_index} board:\n{state}")
                    print(
                        f"Legal actions: {legal_action_names}; selected action: "
                        f"{game.action_name(decision.action)}"
                    )
            results: List[StepResult] = [
                game.step(decision.action)
                for game, decision in zip(self.games, decisions)
            ]
            assert all(result.action_applied for result in results), (
                "If we selected an action from legal_actions, it should be applied"
            )
            for game_index, (rollout, state, decision, result) in enumerate(
                zip(rollouts_by_game, states, decisions, results)
            ):
                game_steps = self.game_steps[game_index] + 1
                truncated = (
                    not result.terminated
                    and game_steps >= self.config.max_episode_steps
                )
                rollout.append(
                    PolicyRolloutInformation(
                        state=state,
                        action=decision.action,
                        next_state=result.next_state.copy(),
                        log_prob=decision.log_prob,
                        value=decision.value,
                        reward=result.reward * self.config.reward_multiplier,
                        done=result.terminated,
                        truncated=truncated,
                        legal_actions_mask=decision.illegal_actions_mask,
                    )
                )
                self.game_steps[game_index] = game_steps
            for game_index, (game, result) in enumerate(zip(self.games, results)):
                if result.terminated or rollouts_by_game[game_index][-1].truncated:
                    game.reset(seed=int(self.rng.integers(0, 2**32)))
                    self.game_steps[game_index] = 0
        # TODO: could maybe parallelize?
        for rollout in rollouts_by_game:
            gae = 0
            for t in reversed(range(len(rollout))):
                step: PolicyRolloutInformation = rollout[t]
                if step.done:
                    next_value = 0
                    gae = 0
                elif step.truncated:
                    next_value = self._predict_value(step.next_state)
                    gae = 0
                elif t + 1 < len(rollout):
                    next_value = rollout[t+1].value
                else:
                    next_value = self._predict_value(step.next_state)
                delta = step.reward + self.config.gamma * next_value - step.value
                gae = delta + self.config.gamma * self.config.gae_lambda * gae
                rollout[t] = dataclasses.replace(step, advantage=gae)
        all_rollouts = [step for rollout in rollouts_by_game for step in rollout]
        advantages = [step.advantage for step in all_rollouts]
        advantage_mean = float(np.mean(advantages))
        advantage_std = float(np.std(advantages))
        print(
            "Rollout advantages: "
            f"mean={advantage_mean:.4f}; std={advantage_std:.4f}"
        )
        return all_rollouts, advantage_mean, advantage_std

    # Uses policy rollout information on minibatches 
    def run_epoch(
        self,
        policy_rollout_information_list: List[PolicyRolloutInformation],
        advantage_mean: float,
        advantage_std: float,
    ) -> EpochStatistics:
        shuffled_rollouts = self.rng.permutation(policy_rollout_information_list)
        policy_objectives = []
        value_objectives = []
        entropy_objectives = []
        kl_divergences = []
        outside_ranges = []
        value_errors = []
        value_targets = []
        for i in range(0, len(shuffled_rollouts), self.config.mini_batch_size):
            minibatch = shuffled_rollouts[i : i + self.config.mini_batch_size]
            self.optimizer.zero_grad()
            objectives = self._compute_objective(minibatch, advantage_mean, advantage_std)
            policy_objectives.append(objectives.policy.detach())
            value_objectives.append(objectives.value.detach())
            entropy_objectives.append(objectives.negative_entropy.detach())
            kl_divergences.append(objectives.approximate_kl.detach())
            outside_ranges.append(objectives.outside_clip_range.detach())
            value_errors.append(objectives.value_error.detach())
            value_targets.append(objectives.value_target.detach())
            objective = (
                objectives.policy
                + self.config.value_coefficient * objectives.value
                + self.config.entropy_coefficient * objectives.negative_entropy
            ).mean()
            objective.backward()
            if self.config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )
            self.optimizer.step()
        return EpochStatistics(
            policy_loss=torch.cat(policy_objectives).mean().item(),
            value_loss=torch.cat(value_objectives).mean().item(),
            negative_entropy=torch.cat(entropy_objectives).mean().item(),
            approximate_kl=torch.cat(kl_divergences).mean().item(),
            clip_fraction=torch.cat(outside_ranges).float().mean().item(),
            explained_variance=1 - torch.cat(value_errors).var().item() /(torch.cat(value_targets).var().item() + self.config.denominator_epsilon)
        )

    def _compute_objective(
        self,
        steps: List[PolicyRolloutInformation],
        advantage_mean: float,
        advantage_std: float,
    ) -> BatchObjectives:
        advantages = self._steps_to_tensor(steps, "advantage")
        normalized_advantages = (advantages - advantage_mean) / (advantage_std + self.config.denominator_epsilon)
        old_policy_log_probs = self._steps_to_tensor(steps, "log_prob")
        # dimension 0 is the batch dimension
        actor_critic_output: ActorCriticOutput = self.model(
            self._state_batch_tensor([step.state for step in steps])
        )
        action_logits = actor_critic_output.action_logits
        transformed_predicted_values = actor_critic_output.value
        predicted_values = self._untransform_predicted_value(transformed_predicted_values)
        legal_actions_mask = self._steps_to_tensor(
            steps,
            "legal_actions_mask",
            dtype=torch.bool,
        )
        action_logits = action_logits.masked_fill(legal_actions_mask, -torch.inf)
        log_probs = F.log_softmax(action_logits, dim=-1)
        sampled_actions = self._steps_to_tensor(steps, "action", dtype=torch.long)
        current_log_prob = log_probs[torch.arange(log_probs.shape[0], device=self.device()), sampled_actions]
        clip_multiplier = torch.where(normalized_advantages > 0, 1 + self.config.clip_epsilon, 1 - self.config.clip_epsilon)
        advantage_clip = clip_multiplier * normalized_advantages
        approximate_kl = old_policy_log_probs - current_log_prob
        probability_ratio = torch.exp(-approximate_kl)
        policy_objective = -torch.minimum(probability_ratio * normalized_advantages, advantage_clip)
        outside_clip_range = torch.abs(probability_ratio - 1) > self.config.clip_epsilon
        values = self._steps_to_tensor(steps, "value")
        value_target = advantages + values
        transformed_value_target = self._transform_value_target(value_target)
        value_error = value_target - predicted_values
        value_objective = (transformed_value_target - transformed_predicted_values) ** 2
        dist = categorical.Categorical(logits=action_logits)
        entropy_objective = -dist.entropy()
        return BatchObjectives(
            policy=policy_objective,
            value=value_objective,
            negative_entropy=entropy_objective,
            approximate_kl=approximate_kl,
            outside_clip_range=outside_clip_range,
            value_error=value_error,
            value_target=value_target
        )

    # Outer iteration loop of PPO
    # Collects the policy rollout, then runs epochs reusing the rollout information
    def run_outer_iteration(self) -> List[EpochStatistics]:
        self._print_timestamp("Outer iteration starting; collecting rollouts.")
        policy_rollout_information_list, advantage_mean, advantage_std = self.collect_batched_information()
        self._print_timestamp("Rollouts collected; starting PPO updates.")
        epoch_stats = []
        self._print_timestamp(f"PPO update epoch 0/{self.config.update_epochs}.")
        for epoch in range(1, self.config.update_epochs + 1):
            epoch_stats.append(self.run_epoch(policy_rollout_information_list, advantage_mean, advantage_std))
            if epoch % 10 == 0:
                self._print_timestamp(f"PPO update epoch {epoch}/{self.config.update_epochs} complete.")
        self._print_timestamp("Outer iteration complete.")
        return epoch_stats
