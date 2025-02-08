import numpy as np
import random
import math
import heapq
from matplotlib import pyplot as plt

class CascadingBandit:
    def __init__(self, num_arms, num_positions, num_players):
        """
        Initialize the cascading bandit environment.

        :param num_arms: Total number of arms (items) available.
        :param probabilities: List of click probabilities for each arm.
        :param num_positions: Number of positions to recommend.
        """
        assert num_positions <= num_arms, "Number of positions cannot exceed number of arms."

        self.num_arms = num_arms
        self.probabilities = []
        for i in range(num_arms):
            self.probabilities.append(random.uniform(0, 1))
        self.num_positions = num_positions
        self.empirical_means = np.zeros(num_arms)
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
            if np.random.rand() < self.probabilities[i]:  # Simulate click
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

def convert_to_int(arm, M, L):
    result = 0
    for index, a in enumerate(arm):
        result += a * (L ** (M - index - 1))
    return result
    
def convert_to_arm(num, M, L):
    result = []
    while(len(result) < M):
        result.insert(0, num % L)
        num = num // L
    return result

def k_largest_indices(lst, k):
    return [i for _, i in heapq.nlargest(k, enumerate(lst), key = lambda x: x[1])]

# Example Simulation
def simulate_cascading_bandit(total_rounds):
    explore_phase = True
    num_players = 4
    num_positions = 3
    indiv_arms = 5
    num_arms = indiv_arms ** num_players
    players = [CascadingBandit(num_arms, num_positions, num_players) for i in range(num_players)]
    click_probabilities = []
    for i in range(num_arms):
        click_probabilities.append(random.uniform(0, 1))
    optimal_score = optimize(click_probabilities, num_positions)
    current_regret = 0
    regret = []
    score = 0
    observations = np.zeros(num_arms)
    desired_set = list(range(num_arms))
    current_order = np.arange(num_positions)
    
    # click_probabilities = []
    # for i in range(num_players):
    #     for i in range(num_arms):
    #         click_probabilities.append(random.uniform(0, 1))
    #     print(str(num_arms), str(len(click_probabilities)))
    #     bandit = CascadingBandit(num_arms = num_arms, probabilities = click_probabilities, num_positions = num_positions)
    #     click_probabilities.clear()
    #     print("Print ", *click_probabilities)
    #     players.append(bandit)
    
    # for i in range(len(click_probabilities)):
    #     print(click_probabilities[i])
      # Number of items to recommend at a time

    # Initialize environment
    # bandit = CascadingBandit(num_arms = num_arms, probabilities = click_probabilities, num_positions = num_positions)

    # UCB Intervals Algorithm Problem B Parameters to Update
    # empirical_means = np.zeros(num_arms)
    # observations = np.zeros(num_arms)
    # UCB = np.zeros(num_arms)
    # LCB = np.zeros(num_arms)

    # # for regret
    # optimal_score = optimize(click_probabilities, num_positions)
    # current_regret = 0
    # regret = []
    # score = 0

    phase = 1
    t = 1
    while(True):
        # Explore phase
        for j in range(num_arms * phase):
            current_order = (current_order + 1) % (len(desired_set))
            recommendations = [desired_set[i] for i in current_order]       # fix later

            score = calc_score(recommendations, click_probabilities, num_positions)
            current_regret += optimal_score - score
            regret.append(current_regret)
            
            click = [players[p].recommend(recommendations) for p in range(num_players)]

            for p in range(num_players):
                for i, arm in enumerate(recommendations[:click[p] + 1]):
                    if i == click[p]:
                        inc = 1
                    else:
                        inc = 0
                    players[p].empirical_means[arm] = (players[p].empirical_means[arm] * observations[arm] + inc) / (observations[arm] + 1)
                observations[arm] += (1 / num_players)
            t += 1
            if(t > T):
                break
        if(t > T):
            break
        
        arms_p_indiv = [[] for p in range(num_players)]
        for p in range(num_players):
            arms_p = (k_largest_indices(players[p].empirical_means, num_positions))
            for a in arms_p:
                arms_p_indiv[p].append(convert_to_arm(a, num_players, indiv_arms)[p])
        
        arms = []
        for n in range(num_positions):
            arms.append([arms_p_indiv[p][n] for p in range(num_players)])

        recommendations = [convert_to_int(arms[n], num_players, indiv_arms) for n in range(num_positions)]
        
        score = calc_score(recommendations, click_probabilities, num_positions)

        while(math.log2(t) % 1 != 0):
            current_regret += optimal_score - score
            # print(optimal_score, "-", score, "=", current_regret)
            regret.append(current_regret)
            t += 1
            # print(t, ":", math.log2(t) % 1 != 0)

        phase += 1

        if(t > T):
            break
        


    # for t in range(total_rounds):
    #     if explore_phase:
    #         for i in range(num_players):
    #             print("explore phase")
    #         explore_phase = False
    #     else:
    #         while(math.log2(t) % 1 != 0):
    #             print("commit phase")
    #         explore_phase = True

        # num_popped = 0
        # Calculate UCB Intervals
        # for arm in range(num_arms):
        #     if(observations[arm] == 0):
        #         UCB[arm] = np.inf
        #         LCB[arm] = -np.inf
        #     else:
        #         UCB[arm] = empirical_means[arm] + ((1.5) * np.log(total_rounds) / observations[arm]) ** (0.5)
        #         LCB[arm] = empirical_means[arm] - ((1.5) * np.log(total_rounds) / observations[arm]) ** (0.5)

        # print(str(UCB) + " " + str(LCB))

        # Initialize recommendations from the current_order
        # recommendations = [desired_set[i] for i in current_order]

        # # Check if desired_set is already right size, if not, check for disjoint arms
        # if len(desired_set) > num_positions:
        #     for i, rec in enumerate(recommendations):
        #         counter = 0
        #         for arm in desired_set:
        #             if UCB[rec] < LCB[arm]:
        #                 counter += 1
        #         if(counter >= num_positions):
        #             # arm is disjoint, replace it in the recommendation with another arm in the desired set
        #             recommendations[i] = desired_set[(current_order[-1] + (i + 1)) % num_positions]
        #             #remove arm from desired set
        #             # print(desired_set[(current_order[0] + i) % num_arms])
        #             desired_set.pop((current_order[0] + i) % len(desired_set))
        #             num_popped += 1
        #             # print(f"New desired set {desired_set}. popped {rec}")
        #             break

        #     # Recommend and observe clicks
        # click = bandit.recommend(recommendations)
        # click = ""
        # for i in range(num_players):
        #     click += str(players[i].recommend(recommendations))
        
        # print("Click: ", click)
        # convert = [int(char) for char in click]
        # print(convert)
        # print("Int: ", str(convert_to_int(convert, num_players, num_arms)))

        # # score = calc_score(recommendations, click_probabilities, num_positions)

        # # # Update means and error terms
        # for p in players:
        #     for i, arm in enumerate(p.empirical_means[:int(click) + 1]):
        #         if i == click:
        #             inc = 1
        #         else:
        #             inc = 0
        #         p.empirical_means[i] = (p.empirical_means[i] * p.observations[i] + inc) / (p.observations[i] + 1)
        #         p.observations[i] += 1

        # for p in players:
        #     print(p.empirical_means)

        # # for i, arm in enumerate(recommendations[:click + 1]):
        # #     if i == click:
        # #         inc = 1
        # #     else:
        # #         inc = 0
        # #     empirical_means[arm] = (empirical_means[arm] * observations[arm] + inc) / (observations[arm] + 1)
        # #     observations[arm] += 1

        # # Update current_order recommendation
        # # if num_popped == 0:
        # #     current_order = (current_order + 1) % (len(desired_set))
        # # else:
        # current_order = np.arange(num_positions) % len(desired_set)

        # # Toggle comment if don't want to display rounds
        # print(f"Round {t + 1}: Recommended arms {recommendations}")
        # print(", Click Index:", click)
        # print("Observations: ", end="")
        # for i in observations:
        #     print(str(int(i)), end="")
        #     print()
        # print("Desired set: ", end="")
        # print(*desired_set)
        # print(str(num_positions) + " " + str(num_arms))

    # print("Length of regret: ", len(regret))
    return regret

    # regret = optimal_score - score
    # return regret

T = 1000000
regret = simulate_cascading_bandit(T)
regret = regret[:T]
plt.scatter(list(range(T)), regret)
plt.show()
# print("Final Regret: " + str(regret))
# print(str(len(regret)))