import numpy as np
import random
import math
from matplotlib import pyplot as plt

class CascadingBandit:
    def __init__(self, num_arms, probabilities, num_positions):
        """
        Initialize the cascading bandit environment.

        :param num_arms: Total number of arms (items) available.
        :param probabilities: List of click probabilities for each arm.
        :param num_positions: Number of positions to recommend.
        """
        assert len(probabilities) == num_arms, "Probabilities must match the number of arms."
        assert num_positions <= num_arms, "Number of positions cannot exceed number of arms."

        self.num_arms = num_arms
        self.probabilities = probabilities
        self.num_positions = num_positions
        self.reset()

    def reset(self):
        """Reset the environment (e.g., for a new simulation run)."""
        self.history = []  # Stores history of arm selections and clicks

    def recommend(self, selected_arms):
        """
        Simulate a recommendation to the user.

        :param selected_arms: List of indices of arms to recommend (length should match num_positions).
        :return: Clicks vector (1 if clicked, 0 otherwise) for each position.
        """
        assert len(selected_arms) == self.num_positions, "Number of selected arms must match num_positions."



        isClick = False

        for i, arm in enumerate(selected_arms):
            if np.random.rand() < self.probabilities[arm]:  # Simulate click
                click = i
                isClick = True
                break  # Stop after the first click (cascading model)

        if isClick == False:
            click = self.num_positions

        self.history.append((selected_arms, click))
        return click

def optimize(click_probabilities, num_positions):
    click_probabilities.sort(reverse = True)
    # top_positions = click_probabilities[:num_positions]
    # for arm in range(num_positions, len(click_probabilities)):
    #     for i, top_arm in enumerate(top_positions):
    #         if arm > top_arm:
    #             top_positions[i] = arm
    #             break
    
    return calc_score(list(range(num_positions)), click_probabilities, num_positions)

def calc_score(positions, click_probabilities, num_positions):
    prob = 1
    for arm in positions:
        arm = int(arm)
        prob = prob * (1 - click_probabilities[arm])
    prob = 1 - prob
    return  prob

# Example Simulation
def simulate_cascading_bandit(total_rounds):
    num_positions = 5  # Number of items to recommend at a time
    num_players = 2
    indiv_arms = 3
    num_arms = indiv_arms ** num_players
    click_probabilities = []
    for i in range(num_arms):
        click_probabilities.append(random.uniform(0, 1))
    
    # for i in range(len(click_probabilities)):
    #     print(click_probabilities[i])

    # Initialize environment
    bandit = CascadingBandit(num_arms = num_arms, probabilities = click_probabilities, num_positions = num_positions)

    # UCB Intervals Algorithm Problem B Parameters to Update
    empirical_means = np.zeros((num_players, num_arms))
    observations = np.zeros(num_arms)
    desired_set = list(range(num_arms))
    current_order = np.arange(num_positions)
    UCB = np.zeros((num_players, num_arms))
    LCB = np.zeros((num_players, num_arms))

    # for regret
    optimal_score = optimize(click_probabilities, num_positions)
    current_regret = 0
    regret = []
    score = 0

    for t in range(total_rounds):
        num_popped = 0

        # Calculate UCB Intervals
        for p in range(num_players):
            for arm in range(num_arms):
                if(observations[arm] == 0):
                    UCB[p][arm] = np.inf
                    LCB[p][arm] = -np.inf
                else:
                    UCB[p][arm] = empirical_means[p][arm] + ((1.5) * np.log(total_rounds) / observations[arm]) ** (0.5)
                    LCB[p][arm] = empirical_means[p][arm] - ((1.5) * np.log(total_rounds) / observations[arm]) ** (0.5)

        # print(str(UCB) + " " + str(LCB))

        # Initialize recommendations from the current_order
        recommendations = [desired_set[i] for i in current_order]

        # Check if desired_set is already right size, if not, check for disjoint arms
        popped = False
        for p in range(num_players):
            if len(desired_set) > num_positions:
                for i, rec in enumerate(recommendations):
                    counter = 0
                    for arm in desired_set:
                        if UCB[p][rec] < LCB[p][arm]:
                            counter += 1
                    if(counter >= num_positions):
                        # arm is disjoint, replace it in the recommendation with another arm in the desired set
                        recommendations[i] = desired_set[(current_order[-1] + (i + 1)) % num_positions]
                        #remove arm from desired set
                        # print(desired_set[(current_order[0] + i) % num_arms])
                        desired_set.pop((current_order[0] + i) % len(desired_set))
                        num_popped += 1
                        # print(f"New desired set {desired_set}. popped {rec}")
                        popped = True
                        break
            if popped:
                break
            # Recommend and observe clicks
            
        click = [bandit.recommend(recommendations) for p in range(num_players)]

        score = calc_score(recommendations, click_probabilities, num_positions)

        # Update means and error terms
        for p in range(num_players):
            for i, arm in enumerate(recommendations[:click[p] + 1]):
                if i == click[p]:
                    inc = 1
                else:
                    inc = 0
                empirical_means[p][arm] = (empirical_means[p][arm] * observations[arm] + inc) / (observations[arm] + 1)
            observations[arm] += (1 / num_players)

        # Update current_order recommendation
        if num_popped == 0:
            current_order = (current_order + 1) % (len(desired_set))
        else:
            current_order = np.arange(num_positions) % len(desired_set)

        # Toggle comment if don't want to display rounds
        # print(f"Round {t + 1}: Recommended arms {recommendations}")
        # print(", Click Index {click}")
        # print("Observations: ", end="")
        # for i in observations:
        #     print(str(int(i)), end="")
        #     print()
        # print("Desired set: ", end="")
        # print(*desired_set)
        # print(str(num_positions) + " " + str(num_arms))
        current_regret += optimal_score - score
        # print(optimal_score, score)
        regret.append(current_regret)

    # print("Length of regret: ", len(regret))
    return regret

    # regret = optimal_score - score
    # return regret

T = 10000
regret = simulate_cascading_bandit(T)
plt.scatter(list(range(T)), regret)
plt.show()
print("Final Regret: " + str(regret))