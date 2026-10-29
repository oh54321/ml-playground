"""Game interaction and rollout collection for PPO."""

import dataclasses
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.distributions import categorical

from core.environments.game import Game, StepResult
from core.environments.state import State
from core.models.actor_critic import ActorCriticModel
from .ppo_types import ActionDecision, PolicyRolloutInformation, PPOConfig


@dataclass
class _GameSlot:
    game: Game
    steps_in_episode: int = 0


class RolloutCollector:
    """Own parallel game state and collect PPO training rollouts."""

    def __init__(
        self,
        model: ActorCriticModel,
        config: PPOConfig,
        game_factory: Callable[[int], Game],
        rng: np.random.Generator,
    ) -> None:
        self.model = model
        self.config = config
        self.rng = rng
        seeds = self.rng.integers(
            0,
            2**32,
            size=config.num_parallel_games,
            dtype=np.uint32,
        )
        self._game_slots = [
            _GameSlot(game=game_factory(int(seed)))
            for seed in seeds
        ]
        self._games = [slot.game for slot in self._game_slots]

    @property
    def games(self) -> list[Game]:
        return self._games

    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def state_batch_tensor(self, states: list[State]) -> torch.Tensor:
        """Encode a homogeneous CPU batch, then transfer it to the device once."""
        if not states:
            raise ValueError("Cannot encode an empty state batch")
        state_batch = type(states[0]).batch_to_tensor(states)
        return state_batch.to(device=self.device(), dtype=self.dtype())

    def predict_value(self, state: State) -> float:
        return self.model(self.state_batch_tensor([state])).value.item()

    def steps_to_tensor(
        self,
        steps: list[PolicyRolloutInformation],
        field_name: str,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Convert one rollout field into a tensor on the model's device."""
        if not steps:
            raise ValueError("Cannot create a tensor from an empty step list")
        values = [getattr(step, field_name) for step in steps]
        if isinstance(values[0], torch.Tensor):
            return torch.stack(values).to(device=self.device(), dtype=dtype)
        return torch.tensor(
            values,
            device=self.device(),
            dtype=dtype if dtype is not None else self.dtype(),
        )

    def generate_next_actions(
        self,
        games: list[Game],
        deterministic: bool = False,
    ) -> list[ActionDecision]:
        """Choose actions and return the state snapshots used to predict them."""
        if not games:
            raise ValueError("Must evaluate at least one game")
        states = [game.current_state().copy() for game in games]
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
        output = self.model(self.state_batch_tensor(states))
        action_logits = output.action_logits
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
        distribution_logits = torch.log_softmax(action_logits, dim=-1)
        categorical_distribution = categorical.Categorical(logits=distribution_logits)
        action = (
            action_logits.argmax(dim=-1)
            if deterministic
            else categorical_distribution.sample()
        )
        log_probability = categorical_distribution.log_prob(action)
        action_values = action.tolist()
        log_probability_values = log_probability.tolist()
        value_values = output.value.tolist()
        return [
            ActionDecision(
                state=states[i],
                action=int(action_values[i]),
                log_prob=float(log_probability_values[i]),
                value=float(value_values[i]),
                illegal_actions_mask=illegal_actions_mask[i].clone(),
            )
            for i in range(len(games))
        ]

    def _step_games(self, decisions: list[ActionDecision]) -> list[StepResult]:
        results = [
            game.step(decision.action)
            for game, decision in zip(self.games, decisions, strict=True)
        ]
        assert all(result.action_applied for result in results), (
            "If we selected an action from legal_actions, it should be applied"
        )
        return results

    def _record_transitions(
        self,
        rollouts_by_game: list[list[PolicyRolloutInformation]],
        decisions: list[ActionDecision],
        results: list[StepResult],
    ) -> None:
        """Record transitions, then reset games that ended or reached their step limit."""
        for slot, rollout, decision, result in zip(
            self._game_slots, rollouts_by_game, decisions, results, strict=True,
        ):
            slot.steps_in_episode += 1
            truncated = (
                not result.terminated
                and slot.steps_in_episode >= self.config.max_episode_steps
            )
            rollout.append(
                PolicyRolloutInformation(
                    state=decision.state,
                    action=decision.action,
                    next_state=result.next_state.copy(),
                    log_prob=decision.log_prob,
                    value=decision.value,
                    reward=result.reward * self.config.reward_multiplier,
                    done=result.terminated,
                    truncated=truncated,
                    illegal_actions_mask=decision.illegal_actions_mask,
                )
            )

            if result.terminated or truncated:
                slot.game.reset(seed=int(self.rng.integers(0, 2**32)))
                slot.steps_in_episode = 0

    def _add_advantages(
        self,
        rollouts_by_game: list[list[PolicyRolloutInformation]],
    ) -> list[PolicyRolloutInformation]:
        for rollout in rollouts_by_game:
            gae = 0
            for t in reversed(range(len(rollout))):
                step = rollout[t]
                if step.done:
                    next_value = 0
                    gae = 0
                elif step.truncated:
                    next_value = self.predict_value(step.next_state)
                    gae = 0
                elif t + 1 < len(rollout):
                    next_value = rollout[t + 1].value
                else:
                    next_value = self.predict_value(step.next_state)
                delta = step.reward + self.config.gamma * next_value - step.value
                gae = delta + self.config.gamma * self.config.gae_lambda * gae
                rollout[t] = dataclasses.replace(step, advantage=gae)
        return [step for rollout in rollouts_by_game for step in rollout]

    @torch.no_grad()
    def collect_batched_information(self) -> tuple[list[PolicyRolloutInformation], float, float]:
        with self.model.prepared_inference():
            rollouts_by_game: list[list[PolicyRolloutInformation]] = [[] for _ in self.games]
            for _ in range(self.config.steps_per_game_per_rollout):
                decisions = self.generate_next_actions(self.games)
                results = self._step_games(decisions)
                self._record_transitions(rollouts_by_game, decisions, results)
            all_rollouts = self._add_advantages(rollouts_by_game)
        advantages = [step.advantage for step in all_rollouts]
        advantage_mean = float(np.mean(advantages))
        advantage_std = float(np.std(advantages))
        return all_rollouts, advantage_mean, advantage_std
