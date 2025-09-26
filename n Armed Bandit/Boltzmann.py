import numpy as np
from numpy.random import default_rng
import torch as tc

class boltzBandit:
    def __init__(self, n_arms: int, total_steps: int, learning_rate: float, backend = 'numpy'):
        '''
        Simulates the n-arm bandit using numerical preference, where each state is assigned a probability of being chosen based on
        some preference function. The joint probability distribution is the Boltzmann or softmax distribution, making the
        preference function like the states energy. The inputs are:
         - n_arms: number of arms
         - total_steps: length of simulation or number of decisions
         - learning_rate: parameter in recursive update used to learn the preference function
         - backend: framework for running the simulation, options are 'numpy' and 'torch'
        '''
        self.backend = backend
        
        self.arms = n_arms
        self.time = 1
        self.ending = total_steps+1
        self.rng = default_rng()
        self.values = self.rng.standard_normal((self.arms,))
        self.baseline = 0.
        if self.backend == 'numpy':
            self.preference = np.zeros((n_arms,))
        elif self.backend == 'torch':
            self.preference = tc.zeros((n_arms))
            self.probability = tc.nn.Softmax(0)
        self.rate = learning_rate
        self.history = np.zeros((self.ending, self.arms))
        
    def Boltzmann(self, arm: int):
        '''
        Implements the Boltzmann or softmax distribution. Input is the number of arms.
        '''
        if self.backend == 'numpy':
            pFun = np.sum(np.exp(self.preference))
            return np.exp(self.preference[arm])/pFun
        elif self.backend == 'torch':
            return self.probability(self.preference)[arm]
    
    def updateBaseline(self, outcome: float):
        '''
        Updates the average total reward.
        '''
        self.baseline = ((self.time-1)/self.time)*self.baseline + outcome/self.time
    
    def updatePreference(self, arm: int, outcome: float):
        '''
        Implements a stoachastic gradient ascent recursion relation to update the preference function estimate.
        '''
        for i in range(self.arms):
            if i == arm:
                self.preference[arm] = self.preference[arm] + self.rate*(outcome-self.baseline)*(1-self.Boltzmann(arm))
            else:
                self.preference[i] = self.preference[i] - self.rate*(outcome-self.baseline)*self.Boltzmann(i)
                
    def act(self):
        '''
        Implements the policy where the maximally likely action is chosen.
        '''
        if np.allclose(self.preference, self.preference[0]):
            action = self.rng.integers(0,self.arms)
            reward = self.values[action] + self.rng.standard_normal()
        else:
            if self.backend == 'numpy':
                action = np.argmax(self.preference)
            elif self.backend == 'torch':
                action = tc.argmax(self.preference)
                action = action.numpy()
            reward = self.values[action] + self.rng.standard_normal()

        self.updateBaseline(reward)
        self.time += 1
        self.updatePreference(action, reward)
        self.history[self.time-1, action] = reward
        
    def fullGame(self):
        for _ in range(self.ending-1):
            self.act()