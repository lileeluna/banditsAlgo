import numpy as np
import random
import math

class MultiplayerCascadingBandit:
    def __init__(self, player1_arms, player2_arms, probabilities, num_positions):
        """
        Initialize the cascading bandit environment for two players.

        :param player1_arms: Total number of arms available to player 1.
        :param player2_arms: Total number of arms available to player 2.
        :param probabilities: Matrix of click probabilities for each joint arm (tuple).
        :param num_positions: Number of positions to recommend.
        """
        assert len(probabilities) == player1_arms and len(probabilities[0]) == player2_arms, \
            "Probabilities must match the number of arms for both players."
        
        self.player1_arms = player1_arms
        self.player2_arms = player2_arms
        self.probabilities = probabilities
        self.num_positions = num_positions
        self.reset()

    def reset(self):
        """Reset the environment (e.g., for a new simulation run)."""
        self.history = []  # Stores history of joint arm selections and clicks

    def recommend(self, selected_joint_arms):
        """
        Simulate a recommendation to the user.

        :param selected_joint_arms: List of tuples representing joint arms to recommend.
        :return: Clicks vector (1 if clicked, 0 otherwise) for each position.
        """
        assert len(selected_joint_arms) == self.num_positions, \
            "Number of selected joint arms must match num_positions."

        isClick = False

        for i, (arm1, arm2) in enumerate(selected_joint_arms):
            if np.random.rand() < self.probabilities[arm1][arm2]:  # Simulate click
                click = i
                isClick = True
                break  # Stop after the first click (cascading model)

        if not isClick:
            click = self.num_positions

        self.history.append((selected_joint_arms, click))
        return click

def simulate_multiplayer_cascading_bandit():
    player1_arms = 6
    player2_arms = 6

    # Generate a random matrix of click probabilities for joint arms
    probabilities = [[random.uniform(0, 1) for _ in range(player2_arms)] for _ in range(player1_arms)]
    num_positions = 8  # Number of joint arms to recommend at a time

    # Initialize environment
    bandit = MultiplayerCascadingBandit(
        player1_arms=player1_arms, 
        player2_arms=player2_arms, 
        probabilities=probabilities, 
        num_positions=num_positions
    )

    # Initialize UCB parameters for both players
    empirical_means = np.zeros((player1_arms, player2_arms))
    observations = np.zeros((player1_arms, player2_arms))

    total_rounds = 1000
    score = 0

    for t in range(total_rounds):
        # Calculate UCB intervals
        UCB = np.zeros((player1_arms, player2_arms))

        for i in range(player1_arms):
            for j in range(player2_arms):
                if observations[i][j] == 0:
                    UCB[i][j] = np.inf
                else:
                    UCB[i][j] = empirical_means[i][j] + ((1.5 * np.log(total_rounds) / observations[i][j]) ** 0.5)

        # Select top joint arms based on UCB
        joint_arm_indices = np.dstack(np.unravel_index(np.argsort(UCB.ravel())[::-1], UCB.shape))[0]
        selected_joint_arms = [tuple(joint_arm_indices[i]) for i in range(num_positions)]

        # Recommend and observe clicks
        click = bandit.recommend(selected_joint_arms)

        if click != num_positions:
            score += 1

        # Update means and observation counts
        for i, (arm1, arm2) in enumerate(selected_joint_arms[:click + 1]):
            if i == click:
                inc = 1
            else:
                inc = 0

            empirical_means[arm1][arm2] = (
                empirical_means[arm1][arm2] * observations[arm1][arm2] + inc
            ) / (observations[arm1][arm2] + 1)
            observations[arm1][arm2] += 1

        # Display round information
        # print(f"Round {t + 1}: Recommended joint arms {selected_joint_arms}")
        # Convert np.int64 values to native Python int values
        selected_joint_arms_converted = [(int(a), int(b)) for a, b in selected_joint_arms]

        # Print without np.int64 in the output
        print(f"Round {t + 1}: Recommended joint arms {selected_joint_arms_converted}")

    return score

score = simulate_multiplayer_cascading_bandit()
print(f"Final Score: {score}")