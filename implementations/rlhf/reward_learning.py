import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class Trajectory(ABC):
    # General class for a trajectory.
    # For a given game, implement a subclass of this class to represent the trajectory.
    def __init__(self):
        pass

    @abstractmethod
    def display(self):
        # Display the trajectory to something the human can process
        pass

    @abstractmethod
    def to_tensor(self) -> torch.Tensor:
        # Convert the trajectory into a process-able tensor
        pass


class HumanComparison:
    def __init__(self, trajectory_1: Trajectory, trajectory_2: Trajectory, preference: int):
        self.trajectory_1 = trajectory_1
        self.trajectory_2 = trajectory_2
        assert preference in [-1, 0, 1], "Preference must be -1, 0, or 1"
        self.preference = preference


class RewardModel(nn.Module):
    def __init__(self, model: nn.Module, human_error_probability=0.1):
        super(RewardModel, self).__init__()
        self.model = model
        self.human_error_probability = human_error_probability
        self.human_comparisons = []

    def forward(self, trajectories: list[Trajectory]):
        trajectory_tensor = torch.stack([trajectory.to_tensor() for trajectory in trajectories])
        return self.model.forward(trajectory_tensor)

    def evaluate_trajectory(self, trajectory: Trajectory):
        """
        Evaluate a trajectory using a model.
        """
        return self.forward(trajectory).item()

    def calculate_probability_from_two_trajectories(self, human_comparison: HumanComparison):
        """
        Calculate the probability from two trajectories.
        """
        trajectory_1 = human_comparison.trajectory_1
        trajectory_2 = human_comparison.trajectory_2
        reward_1 = self.evaluate_trajectory(trajectory_1)
        reward_2 = self.evaluate_trajectory(trajectory_2)
        softmax = torch.exp(reward_1) / (torch.exp(reward_1) + torch.exp(reward_2))
        return self.human_error_probability * 0.5 + (1 - self.human_error_probability) * softmax

    def calculate_loss_from_two_trajectories(self, human_comparison: HumanComparison):
        prob_prefer_1 = self.calculate_probability_from_two_trajectories(human_comparison)
        # preference = 1 means mu_1 = 1, mu_2 = 0
        # preference = -1 means mu_1 = 0, mu_2 = 1
        # preference = 0 means mu_1 = mu_2 = 0.5
        mu_1 = (human_comparison.preference + 1) / 2
        mu_2 = (1 - human_comparison.preference) / 2
        return -(mu_1 * prob_prefer_1 + mu_2 * (1 - prob_prefer_1))

    def get_human_preference(self, trajectory_1: Trajectory, trajectory_2: Trajectory):
        """
        Get the human preference between two trajectories.
        """
        # Return 1 if trajectory_1 is preferred, -1 if trajectory_2 is preferred, 0 if they are tied.
        # Play the trajectories by calling the display function
        print("Trajectory 1:")
        trajectory_1.display()

        print("\nTrajectory 2:")
        trajectory_2.display()

        print("\nWhich trajectory do you prefer?")
        print("Enter 1 for Trajectory 1, 2 for Trajectory 2, or 0 for Tie/No preference.")
        response = self.get_human_response()
        self.human_comparisons.append(HumanComparison(trajectory_1, trajectory_2, response))

    def get_human_response(self):
        while True:
            user_input = input("Your choice (1/2/0): ").strip()
            if user_input == "1":
                return 1
            elif user_input == "2":
                return -1
            elif user_input == "0":
                return 0
            else:
                print("Invalid input. Please enter 1, 2, or 0.")

