import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import categorical

from core.game import Game, State
from games.rl_2048.agent import ActorCriticOutput, ResNet


@dataclass(frozen=True)
class PolicyRolloutInformation:
    state: State
    action: int
    next_state: State
    log_prob: float
    value: float
    reward: float
    done: bool
    legal_actions_mask: torch.Tensor
    advantage: float = None


@dataclass(frozen=True)
class PPOConfig:
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    value_coefficient: float
    entropy_coefficient: float
    update_epochs: int
    num_policy_rollout: int
    max_game_steps: int
    mini_batch_size: int
    seed: int
    print_moves: bool
    print_timestamps: bool = True


class PPO():
    def __init__(self, 
        model: ResNet, 
        optimizer: torch.optim.Optimizer, 
        config: PPOConfig
    ):
        torch.manual_seed(config.seed)
        self.model: ResNet = model
        self.optimizer = optimizer
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.clip_epsilon = config.clip_epsilon
        self.update_epochs = config.update_epochs
        self.num_policy_rollout = config.num_policy_rollout
        self.mini_batch_size = config.mini_batch_size
        self.rng = np.random.default_rng(config.seed)
        self.max_game_steps = config.max_game_steps
        self.value_coefficient = config.value_coefficient
        self.entropy_coefficient = config.entropy_coefficient
        self.print_moves = config.print_moves
        self.print_timestamps = config.print_timestamps

    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def _state_tensor(self, state: State) -> torch.Tensor:
        return state.to_tensor().to(device=self.device(), dtype=self.dtype())

    def _print_timestamp(self, message: str) -> None:
        if self.print_timestamps:
            timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            print(f"[{timestamp}] {message}")

    # From the current state and a list of legal actions, return information for the next action
    # Returns the action itself, the log probability of that action,
    # the value function from the current state, and a mask for the list of legal actions
    def select_action(
        self,
        state: State,
        legal_actions: List[int],
        deterministic: bool = False,
    ) -> Tuple[int, float, float, torch.Tensor]:
        tensor_state = self._state_tensor(state)
        actor_critic_output: ActorCriticOutput = self.model(tensor_state)
        action_logits = actor_critic_output.action_logits[0]
        value = actor_critic_output.value.item()
        legal_actions_mask = torch.ones(len(action_logits), dtype=bool, device=action_logits.device)
        legal_actions_mask[legal_actions] = 0
        action_logits = action_logits.masked_fill(legal_actions_mask, -torch.inf)
        categorical_distribution = categorical.Categorical(logits=action_logits)
        action = action_logits.argmax() if deterministic else categorical_distribution.sample()
        log_probability = categorical_distribution.log_prob(action)
        return action.item(), log_probability.item(), value, legal_actions_mask

    # Runs games until we have the desired amount of policy rollouts.
    # Starts a game from scratch, and continually calls select_action to generate the next state along with its information
    # If there are no legal moves, resets the game under a different seed
    @torch.no_grad()
    def collect_information(self, game: Game) -> Tuple[List[PolicyRolloutInformation], float, float]:
        policy_rollout_information_list: List[PolicyRolloutInformation] = []
        while len(policy_rollout_information_list) < self.num_policy_rollout:
            current_rollout: List[PolicyRolloutInformation] = []
            game_not_over = True
            while len(current_rollout) < self.max_game_steps and game_not_over:
                state = game.current_state()
                legal_actions = game.legal_actions()
                action, log_prob, value, legal_actions_mask = self.select_action(state, legal_actions)
                if self.print_moves:
                    legal_action_names = ", ".join(game.action_name(legal_action) for legal_action in legal_actions)
                    print(f"Board:\n{state}")
                    print(f"Legal actions: {legal_action_names}; selected action: {game.action_name(action)}")
                next_state, reward, done, moved = game.step(action)
                assert moved, "If we selected an action from legal_actions, we should have moved"
                policy_rollout_information = PolicyRolloutInformation(
                    state, action, next_state, log_prob, value, reward, done, legal_actions_mask)
                current_rollout.append(policy_rollout_information)
                game_not_over = game_not_over and (not done)
            game.reset(seed=self.rng.integers(0, 100000000))
            gae = 0
            for t in reversed(range(len(current_rollout))):
                step: PolicyRolloutInformation = current_rollout[t]
                if step.done:
                    next_value = 0
                elif t + 1 < len(current_rollout):
                    next_value = current_rollout[t+1].value
                else:
                    actor_critic_output: ActorCriticOutput = self.model(self._state_tensor(step.next_state))
                    next_value = actor_critic_output.value.item()
                delta = step.reward + self.gamma * next_value - step.value
                gae = delta + self.gamma * self.gae_lambda * gae
                current_rollout[t] = dataclasses.replace(step, advantage=gae)
            policy_rollout_information_list.extend(current_rollout)
        advantages = [step.advantage for step in policy_rollout_information_list]
        advantage_mean = float(np.mean(advantages))
        advantage_std = float(np.std(advantages))
        print(
            "Rollout advantages: "
            f"mean={advantage_mean:.4f}; std={advantage_std:.4f}"
        )
        return policy_rollout_information_list, advantage_mean, advantage_std
    
    # Uses policy rollout information on minibatches 
    def run_epoch(self, policy_rollout_information_list: List[PolicyRolloutInformation], advantage_mean: float, advantage_std: float) -> Tuple[float, float, float]:
        shuffled_rollouts = self.rng.permutation(policy_rollout_information_list)
        objective_history = []
        for i in range(0, len(shuffled_rollouts), self.mini_batch_size):
            minibatch = shuffled_rollouts[i : i + self.mini_batch_size]
            self.optimizer.zero_grad()
            objective_tensor = self._compute_objective(minibatch, advantage_mean, advantage_std)
            assert len(objective_tensor.shape) == 1 and objective_tensor.shape[0] == 3
            if len(minibatch) == self.mini_batch_size:
                objective_history.append(objective_tensor.detach())
            objective = objective_tensor[0] + self.value_coefficient * objective_tensor[1] + self.entropy_coefficient * objective_tensor[2]
            objective.backward()
            # TODO: consider gradient clipping
            self.optimizer.step()
        return tuple(torch.stack(objective_history).mean(dim=0).detach().cpu().tolist())

    def _compute_objective(self, steps: List[PolicyRolloutInformation], advantage_mean: float, advantage_std: float) -> torch.Tensor:
        # TODO: refactor these list comprehensions into somewhere nice
        advantages = torch.tensor([step.advantage for step in steps], device=self.device(), dtype=self.dtype())
        # TODO: pass in epsilon to PPO instead of hardcoding here
        normalized_advantages = (advantages - advantage_mean) / (advantage_std + 1e-8)
        old_policy_log_probs = torch.tensor([step.log_prob for step in steps], device = self.device(), dtype=self.dtype())
        # dimension 0 is the batch dimension
        actor_critic_output: ActorCriticOutput = self.model(torch.cat([self._state_tensor(step.state) for step in steps], dim=0))
        action_logits = actor_critic_output.action_logits
        predicted_values = actor_critic_output.value
        legal_actions_mask = torch.stack([step.legal_actions_mask for step in steps], dim=0)
        action_logits = action_logits.masked_fill(legal_actions_mask, -torch.inf)
        log_probs = F.log_softmax(action_logits, dim=-1)
        sampled_actions = torch.tensor([step.action for step in steps], device=self.device())
        current_log_prob = log_probs[torch.arange(log_probs.shape[0], device=self.device()), sampled_actions]
        clip_multiplier = torch.where(normalized_advantages > 0, 1 + self.clip_epsilon, 1 - self.clip_epsilon)
        advantage_clip = clip_multiplier * normalized_advantages
        policy_objective = -torch.minimum(torch.exp(current_log_prob - old_policy_log_probs) * normalized_advantages, advantage_clip)
        values = torch.tensor([step.value for step in steps], device=self.device(), dtype=self.dtype())
        reward_target = advantages + values
        value_objective = ((reward_target - predicted_values) ** 2)
        dist = categorical.Categorical(logits=action_logits)
        entropy_objective = -dist.entropy()
        return torch.stack([policy_objective, value_objective, entropy_objective]).mean(dim=-1)

    # Outer iteration loop of PPO
    # Collects the policy rollout, then runs epochs reusing the rollout information
    def run_outer_iteration(self, game: Game) -> List[Tuple[float, float, float]]:
        self._print_timestamp("Outer iteration starting; collecting rollouts.")
        policy_rollout_information_list, advantage_mean, advantage_std = self.collect_information(game)
        self._print_timestamp("Rollouts collected; starting PPO updates.")
        epoch_stats = []
        self._print_timestamp(f"PPO update epoch 0/{self.update_epochs}.")
        for epoch in range(1, self.update_epochs + 1):
            epoch_stats.append(self.run_epoch(policy_rollout_information_list, advantage_mean, advantage_std))
            if epoch % 10 == 0:
                self._print_timestamp(f"PPO update epoch {epoch}/{self.update_epochs} complete.")
        self._print_timestamp("Outer iteration complete.")
        return epoch_stats
