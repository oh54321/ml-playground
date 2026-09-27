"""Game interaction and rollout collection for PPO."""

import dataclasses
import math
from dataclasses import dataclass
from typing import Callable, List, Literal, Tuple

import numpy as np
import torch
from torch.distributions import categorical

from core.environments.game import Game, StepResult
from core.environments.game_symmetry import GameSymmetry
from core.environments.state import State
from core.models.actor_critic import ActorCriticModel, ActorCriticOutput
from .ppo_types import ActionDecision, PolicyRolloutInformation, PPOConfig
from .ppo_utils import action_log_probabilities as mixed_action_log_probabilities
from .ppo_utils import untransform_predicted_value as reverse_value_transform


@dataclass
class _GameSlot:
    game: Game
    steps_in_episode: int = 0
    symmetry: GameSymmetry | None = None


class RolloutCollector:
    """Own parallel game state and collect PPO training rollouts."""

    def __init__(
        self,
        model: ActorCriticModel,
        config: PPOConfig,
        game_factory: Callable[[int], Game],
        rng: np.random.Generator,
        symmetry_augmentation: Literal["step", "episode"] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.rng = rng
        self.symmetry_rng = np.random.default_rng(
            np.random.SeedSequence(config.seed).spawn(1)[0]
        )
        self.symmetry_augmentation = symmetry_augmentation
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
        if symmetry_augmentation is not None:
            for slot in self._game_slots:
                slot.symmetry = self._sample_symmetry(slot.game)
        self.exploration_probability = 0.0

    @property
    def games(self) -> list[Game]:
        return self._games

    def _sample_symmetry(self, game: Game) -> GameSymmetry | None:
        symmetries = game.symmetries()
        if not symmetries:
            return None
        return symmetries[int(self.symmetry_rng.integers(len(symmetries)))]

    def set_exploration_probability(self, probability: float) -> None:
        if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
            raise ValueError("exploration probability must be finite and in [0, 1)")
        self.exploration_probability = float(probability)

    def action_log_probabilities(
        self,
        masked_logits: torch.Tensor,
        illegal_actions_mask: torch.Tensor,
    ) -> torch.Tensor:
        return mixed_action_log_probabilities(
            masked_logits,
            illegal_actions_mask,
            self.exploration_probability,
        )

    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def state_batch_tensor(self, states: List[State]) -> torch.Tensor:
        """Encode a homogeneous CPU batch, then transfer it to the device once."""
        if not states:
            raise ValueError("Cannot encode an empty state batch")
        state_batch = type(states[0]).batch_to_tensor(states)
        return state_batch.to(device=self.device(), dtype=self.dtype())

    def predict_value(self, state: State) -> float:
        actor_critic_output: ActorCriticOutput = self.model(
            self.state_batch_tensor([state])
        )
        return self.untransform_predicted_value(actor_critic_output.value).item()

    def untransform_predicted_value(self, value: torch.Tensor) -> torch.Tensor:
        return reverse_value_transform(value, self.config.value_transform)

    def steps_to_tensor(
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

    def generate_next_actions(
        self,
        games: List[Game],
        deterministic: bool = False,
        states: List[State] | None = None,
        legal_actions: List[list[int]] | None = None,
    ) -> List[ActionDecision]:
        if not games:
            raise ValueError("Must evaluate at least one game")
        if states is None:
            states = [game.current_state() for game in games]
        elif len(states) != len(games):
            raise ValueError("A supplied state batch must match the game batch")
        if legal_actions is None:
            legal_actions_by_game = [game.legal_actions() for game in games]
        else:
            if len(legal_actions) != len(games):
                raise ValueError("A supplied legal-action batch must match the game batch")
            legal_actions_by_game = legal_actions
        legal_action_counts = torch.tensor(
            [len(legal_actions) for legal_actions in legal_actions_by_game],
            device=self.device(),
            dtype=torch.long,
        )
        if any(len(legal_actions) == 0 for legal_actions in legal_actions_by_game):
            raise ValueError(
                "Every game passed to generate_next_actions must have at least one legal action"
            )
        tensor_state = self.state_batch_tensor(states)
        actor_critic_output: ActorCriticOutput = self.model(tensor_state)
        action_logits = actor_critic_output.action_logits
        values = self.untransform_predicted_value(actor_critic_output.value)
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
        # Deterministic evaluation uses the unmodified network policy. Training
        # samples from the same mixture whose log probabilities PPO will update.
        distribution_logits = (
            action_logits
            if deterministic
            else self.action_log_probabilities(action_logits, illegal_actions_mask)
        )
        categorical_distribution = categorical.Categorical(logits=distribution_logits)
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

    def _policy_views(
        self,
    ) -> tuple[list[GameSymmetry | None], list[State], list[list[int]]]:
        """Return each game's state and legal actions in its policy view."""
        symmetries = [slot.symmetry for slot in self._game_slots]
        states: list[State] = []
        legal_actions_by_game: list[list[int]] = []
        for slot, symmetry in zip(self._game_slots, symmetries, strict=True):
            state = slot.game.current_state()
            legal_actions = slot.game.legal_actions()
            if symmetry is not None:
                state = symmetry.transform_state(state)
                legal_actions = sorted(
                    symmetry.to_transformed_action(action)
                    for action in legal_actions
                )
            states.append(state.copy())
            legal_actions_by_game.append(legal_actions)
        return symmetries, states, legal_actions_by_game

    def _print_moves(
        self,
        states: list[State],
        legal_actions_by_game: list[list[int]],
        decisions: list[ActionDecision],
        symmetries: list[GameSymmetry | None],
    ) -> None:
        if not self.config.print_moves:
            return
        for game_index, (game, state, decision) in enumerate(zip(
            self.games, states, decisions, strict=True
        )):
            symmetry = symmetries[game_index]
            canonical_legal_actions = (
                [symmetry.to_canonical_action(action)
                 for action in legal_actions_by_game[game_index]]
                if symmetry is not None
                else legal_actions_by_game[game_index]
            )
            selected_action = (
                symmetry.to_canonical_action(decision.action)
                if symmetry is not None
                else decision.action
            )
            legal_action_names = ", ".join(
                game.action_name(action) for action in canonical_legal_actions
            )
            print(f"Game {game_index} board:\n{state}")
            print(
                f"Legal actions: {legal_action_names}; selected action: "
                f"{game.action_name(selected_action)}"
            )

    def _step_games(
        self,
        decisions: list[ActionDecision],
        symmetries: list[GameSymmetry | None],
    ) -> list[StepResult]:
        results = [
            game.step(
                symmetry.to_canonical_action(decision.action)
                if symmetry is not None else decision.action
            )
            for game, decision, symmetry in zip(
                self.games, decisions, symmetries, strict=True
            )
        ]
        assert all(result.action_applied for result in results), (
            "If we selected an action from legal_actions, it should be applied"
        )
        return results

    def _record_transitions(
        self,
        rollouts_by_game: list[list[PolicyRolloutInformation]],
        states: list[State],
        decisions: list[ActionDecision],
        results: list[StepResult],
    ) -> None:
        for game_index, (rollout, state, decision, result) in enumerate(
            zip(rollouts_by_game, states, decisions, results, strict=True)
        ):
            slot = self._game_slots[game_index]
            if self.symmetry_augmentation == "step":
                slot.symmetry = self._sample_symmetry(slot.game)
            next_state = (
                slot.symmetry.transform_state(result.next_state).copy()
                if slot.symmetry is not None
                else result.next_state.copy()
            )
            slot.steps_in_episode += 1
            truncated = (
                not result.terminated
                and slot.steps_in_episode >= self.config.max_episode_steps
            )
            rollout.append(
                PolicyRolloutInformation(
                    state=state,
                    action=decision.action,
                    next_state=next_state,
                    log_prob=decision.log_prob,
                    value=decision.value,
                    reward=result.reward * self.config.reward_multiplier,
                    done=result.terminated,
                    truncated=truncated,
                    legal_actions_mask=decision.illegal_actions_mask,
                )
            )

    def _reset_finished_games(
        self,
        rollouts_by_game: list[list[PolicyRolloutInformation]],
        results: list[StepResult],
    ) -> None:
        for slot, rollout, result in zip(
            self._game_slots,
            rollouts_by_game,
            results,
            strict=True,
        ):
            if result.terminated or rollout[-1].truncated:
                slot.game.reset(seed=int(self.rng.integers(0, 2**32)))
                slot.steps_in_episode = 0
                if self.symmetry_augmentation is not None:
                    slot.symmetry = self._sample_symmetry(slot.game)

    def _add_advantages(
        self,
        rollouts_by_game: list[list[PolicyRolloutInformation]],
    ) -> list[PolicyRolloutInformation]:
        # TODO: could maybe parallelize?
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
    def collect_batched_information(
        self,
    ) -> Tuple[List[PolicyRolloutInformation], float, float]:
        rollouts_by_game: list[list[PolicyRolloutInformation]] = [
            [] for _ in self.games
        ]
        for _ in range(self.config.steps_per_game_per_rollout):
            symmetries, states, legal_actions = self._policy_views()
            decisions = self.generate_next_actions(
                self.games,
                states=states,
                legal_actions=legal_actions,
            )
            self._print_moves(states, legal_actions, decisions, symmetries)
            results = self._step_games(decisions, symmetries)
            self._record_transitions(
                rollouts_by_game,
                states,
                decisions,
                results,
            )
            self._reset_finished_games(rollouts_by_game, results)

        all_rollouts = self._add_advantages(rollouts_by_game)
        advantages = [step.advantage for step in all_rollouts]
        advantage_mean = float(np.mean(advantages))
        advantage_std = float(np.std(advantages))
        print(
            "Rollout advantages: "
            f"mean={advantage_mean:.4f}; std={advantage_std:.4f}"
        )
        return all_rollouts, advantage_mean, advantage_std
