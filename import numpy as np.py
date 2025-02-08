import numpy as np
import random

class CascadingBanditMultiAgent:
    def __init__(self, num_players, num_arms, probabilities):
        """
        Multi-agent cascading bandit environment.

        :param num_players: Number of players (agents).
        :param num_arms: Total number of arms (items) available.
        :param probabilities: List of click probabilities for each arm (used to compute joint reward).
        """
        assert len(probabilities) == num_arms, "Probabilities must match the number of arms."
        self.num_players = num_players
        self.num_arms = num_arms
        self.probabilities = probabilities
        self.reset()

    def reset(self):
        """Reset the environment."""
        self.history = []  # Stores history of joint arm selections and rewards

    def recommend(self, joint_arm):
        """
        Simulate a recommendation of a joint arm (tuple of arms).

        :param joint_arm: Tuple of arms selected by all agents.
        :return: Shared reward (0 or 1, based on the cascading model).
        """
        assert len(joint_arm) == self.num_players, "Joint arm must have one arm per player."

        # Compute reward based on cascading model (first click in the joint arms)
        for arm in joint_arm:
            if np.random.rand() < self.probabilities[arm]:
                self.history.append((joint_arm, 1))
                return 1  # Click occurred
        self.history.append((joint_arm, 0))
        return 0  # No click


def simulate_cascading_bandit_multi_agent():
    num_players = 2
    num_arms = 5
    total_rounds = 1000
    click_probabilities = [random.uniform(0, 1) for _ in range(num_arms)]

    # Initialize environment
    bandit = CascadingBanditMultiAgent(num_players, num_arms, click_probabilities)

    # Initialize player-specific parameters
    empirical_means = [np.zeros(num_arms) for _ in range(num_players)]
    observations = [np.zeros(num_arms) for _ in range(num_players)]
    scores = [0] * num_players  # Track individual scores

    for t in range(total_rounds):
        joint_arm = []

        # Each player selects an arm independently based on UCB
        for p in range(num_players):
            UCB = np.zeros(num_arms)
            for arm in range(num_arms):
                if observations[p][arm] == 0:
                    UCB[arm] = np.inf
                else:
                    UCB[arm] = empirical_means[p][arm] + np.sqrt(1.5 * np.log(total_rounds) / observations[p][arm])
            selected_arm = np.argmax(UCB)
            joint_arm.append(selected_arm)

        # Simulate the joint arm and receive the shared reward
        reward = bandit.recommend(tuple(joint_arm))

        # Update parameters for each player based on the shared reward
        for p, arm in enumerate(joint_arm):
            empirical_means[p][arm] = (empirical_means[p][arm] * observations[p][arm] + reward) / (observations[p][arm] + 1)
            observations[p][arm] += 1
            scores[p] += reward

        # Optional: print progress
        if (t + 1) % 100 == 0:
            print(f"Round {t + 1}: Joint arm {tuple(joint_arm)}, Reward {reward}, Scores: {scores}")

    return scores


# Run the simulation
final_scores = simulate_cascading_bandit_multi_agent()
print("Final Scores:", final_scores)