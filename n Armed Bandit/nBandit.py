import numpy as np
from numpy.random import default_rng

class nBandit:
    def __init__(self, index: int, n: int, total_steps: int, epsilon: float, plotting=True, step_mode = 'sample_average', choice_mode = 'ep_greedy'):
        '''
        Simulates the n-arm bandit decision system using action-value estimation. The inputs are:
         - index (int): numerical name of the bandit
         - n (int): number of arms
         - total_steps (int): length of simulation, or number of decisions made
         - epsilon (float): parameter for epsilon-greedy policy
         - plotting (bool) : determines whether or not to store the rewards from each decision for the purpose of plotting
         - step_mode (str) : the value function used by the agent, options are 'sample_average' and 'constant'
         - choice_mode (str) : policy used by agent, options are 'ep_greedy' (epsilon-greedy) and 'UCB' ()
        '''
        self.index = index
        self.arms = n
        self.time = 1
        self.ending = total_steps+1
        self.nongreed = epsilon
        self.rng = default_rng()
        self.values = self.rng.standard_normal((self.arms,))
        self.estimates = np.zeros((self.arms,))
        self.Nk = np.zeros((self.arms,))
        self.plotting = plotting
        if self.plotting:
            self.history = np.zeros((self.ending, self.arms))
        self.optimalCounter = 0
        self.mode = step_mode
        self.choice_mode = choice_mode
    
    def update_estimates(self, arm: int, reward: float, **kwargs):
        '''
        Updates the state value function using a recursive formula. The learning rate is either 1/k for the kth appearance of state
        or a constant.
        '''
        if self.mode == 'sample_average':
            self.estimates[arm] = self.estimates[arm] + (reward - self.estimates[arm])/self.Nk[arm]
        elif self.mode == 'constant':
            alpha = kwargs['step_size']
            self.estimates[arm] = self.estimates[arm] + alpha*(reward - self.estimates[arm])

    def act(self, **kwargs):
        '''
        Implements the policy chosen by the user. Though the real rewards are fixed when the class is initialized, the reward delivered to the
        agent is affected by a noise term to make learning the rewards harder to learn.
        '''
        if self.choice_mode == 'ep_greedy':
            if self.time <= self.ending:
                if np.allclose(self.estimates, self.estimates[0]):
                    action = self.rng.integers(0,self.arms)
                    reward = self.values[action] + self.rng.standard_normal()
                else:
                    r = self.rng.random()
                    if r >= self.nongreed:
                        action = np.argmax(self.estimates)
                        reward = self.values[action] + self.rng.standard_normal()
                        self.optimalCounter += 1
                    else:
                        gChoice = np.argmax(self.estimates)
                        action = self.rng.choice(np.delete(np.arange(0,self.arms,1,dtype=int),gChoice))
                        reward = self.values[action] + self.rng.standard_normal()
                self.Nk[action] += 1.
                self.update_estimates(action, reward)
        elif self.choice_mode == 'UCB':
            if self.time <= self.ending:
                if np.allclose(self.estimates, self.estimates[0]):
                    action = self.rng.integers(0,self.arms)
                    reward = self.values[action] + self.rng.standard_normal()
                else:
                    action = np.argmax(self.estimates + kwargs['exploration'] * np.sqrt(np.log(self.time)/self.Nk))
                    reward = self.values[action] + self.rng.standard_normal()

            self.Nk[action] += 1.
            self.update_estimates(action, reward, step_size = kwargs['step_size'])
        self.time += 1
        if self.plotting:
            self.history[self.time-1, action] = reward
        else:
            return

    def fullGame(self, **kwargs):
        '''
        kwargs: step_size, exploration
        '''
        if 'exploration' in kwargs.keys():
            for _ in range(self.ending-1):
                self.act(step_size = kwargs['step_size'], exploration = kwargs['exploration'])
        else:
            for _ in range(self.ending-1):
                self.act()