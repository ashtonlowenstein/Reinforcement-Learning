import matplotlib.pyplot as plt
from nBandit import nBandit
import numpy as np

def banditTestbed(num_bandits: int, num_arms: int, num_steps: int, epsilon: list[float], step_mode: str, choice_mode: str, **kwargs):
    
    ensemble = []
    for i in range(num_bandits):
        ensemble.append(nBandit(index=i, n=num_arms, total_steps=num_steps, epsilon=epsilon, step_mode=step_mode, choice_mode=choice_mode))
        
    if 'exploration' in kwargs.keys():
        for bandit in ensemble:
            bandit.fullGame(step_size = kwargs['step_size'], exploration = kwargs['exploration'])
    else:
        for bandit in ensemble:
            bandit.fullGame()
    rewardHistory = np.sum(np.stack(np.array([bandit.history for bandit in ensemble])), axis=-1)
    
    avgReward = np.average(rewardHistory, axis=0)
    
    return avgReward

def rewardPlots(num_bandits: int, num_arms: int, num_steps: int, epsilon: list[float], step_mode: str, choice_mode: str, **kwargs):
    rewardContainer = []
    
    for ep in epsilon:
        if 'exploration' in kwargs.keys():
            reward = banditTestbed(num_bandits, num_arms, num_steps, ep, step_mode=step_mode, choice_mode=choice_mode,
                                   step_size = kwargs['step_size'], exploration = kwargs['exploration'])
            rewardContainer.append(reward)
        else:
            reward = banditTestbed(num_bandits, num_arms, num_steps, ep, step_mode=step_mode, choice_mode=choice_mode)
            rewardContainer.append(reward)
        
    for i in range(len(epsilon)):
        plt.plot(rewardContainer[i][:-1], label='epsilon = ' + f'{epsilon[i]}')
        plt.legend()
        plt.xlabel('Steps')
        plt.ylabel('Average Reward')
        plt.show