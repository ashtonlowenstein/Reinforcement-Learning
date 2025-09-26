import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from Boltzmann import boltzBandit

def boltzmannBanditTestbed(num_bandits: int, num_arms: int, num_steps: int, learning_rate: float, backend = 'numpy', **kwargs):
    
    ensemble = []
    for i in range(num_bandits):
        ensemble.append(boltzBandit(num_arms, total_steps=num_steps, learning_rate=learning_rate, backend=backend))
        
    for bandit in ensemble:
        bandit.fullGame()
    
    rewardHistory = np.sum(np.stack(np.array([bandit.history for bandit in ensemble])), axis=-1)
    
    avgReward = np.average(rewardHistory, axis=0)

    return avgReward

def rewardPlots(num_bandits: int, num_arms: int, num_steps: int, learning_rate: float, backend = 'numpy', **kwargs):
    rewardContainer = []
    
    for sz in learning_rate:
        reward = boltzmannBanditTestbed(num_bandits, num_arms, num_steps, sz, backend)
        rewardContainer.append(reward)
        
    for i in range(len(learning_rate)):
        plt.plot(rewardContainer[i][:-1], label='l_rate = ' + f'{learning_rate[i]}')
        plt.legend()
        plt.xlabel('Steps')
        plt.ylabel('Average Reward')
        plt.show