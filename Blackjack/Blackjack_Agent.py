import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from Blackjack import Blackjack
from collections import namedtuple, deque
import math
from itertools import count

class BlackNet(nn.Module):
    def __init__(self, out_mode = 'tanh'):
        """
        Creates a neural network with three hidden linear layers and three hidden normalization layers. There are two output modes whose interpretations are different
        but which nevertheless allow the agent to play blackjack.

        Args:
            out_mode (str, optional): Chooses whether or not to apply the _tanh_ function to the two output values. When 'linear' is chosen, there is 
            no change made to the output linear layer. Only the 'tanh' option allows for an interpretation in terms of state-action value. Defaults to 'tanh'.
        """
        super(BlackNet, self).__init__()
        self.out_mode = out_mode
        #Linear layers
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4_output = nn.Linear(16,2)
    
    def forward(self, input):
        f1 = F.tanh(self.fc1(input))
        f2 = F.tanh(self.fc2(f1))
        f3 = F.tanh(self.fc3(f2))
        
        if self.out_mode == 'tanh':
            #output = F.tanh(self.fc4_output(l3))
            output = F.tanh(self.fc4_output(f3))
        elif self.out_mode == 'linear':
            #output = self.fc4_output(l3)
            output = self.fc4_output(f3)

        return output
    
class ReplayDataset(Dataset):
    def __init__(self, capacity:int, transform = None):
        """
        Creates a buffer into which we can load SASR tuples for the reinforcement learning problem. In this case, each tuple also includes a 'terminal' flag
        to keep track of the last state in an episode. States can be pushed into the buffer endlessly, but once capacity is reached the buffer gets rid of the earliest
        elements (it's a queue).

        Args:
            capacity (int): The length of the buffer, or the maximum number of tuples stored.
            transform (_type_, optional): _description_. Defaults to None.
        """
        self.capacity = capacity
        self.memory = deque([], maxlen = self.capacity)
        self.sasr_tuple = namedtuple('SASR',
                        ('state', 'action', 'next_state', 'reward', 'terminal'))
        self.transform = transform
        
    def push(self, *args):
        """
        Add a new SASRT tuple to the buffer.
        """
        self.memory.append(self.sasr_tuple(*args))
        
    def clearMemory(self):
        """
        Reset the buffer to be empty.
        """
        self.memory = deque([], maxlen = self.capacity)
        
    def netWinnings(self) -> float:
        """
        Sums the rewards in all of the element tuples.

        Returns:
            float: The net reward contained in the buffer.
        """
        winnings = 0.
        for sasr in self.memory:
            winnings += sasr.reward
        return winnings
    
    def __len__(self):
        return len(self.memory)
    
    def __getitem__(self, idx):
        if tc.is_tensor(idx):
            idx = idx.tolist()
        sample = self.memory[idx]    
        if self.transform:
            self.transform(sample)
        
        return sample
    
class AgentGame(Blackjack):
    """
    This multi-purpose class both simulates the rules of blackjack and incorporates the automatic actions of an agent using an epsilon-greedy policy to choose whether
    to hit or stay.
    """
    def __init__(self, policy_net: BlackNet):
        """
        Stores the input neural network to be used as a value function, defines a state as a _namedtuple_ called **State**, initializes an empty state history list, 
        and initializes two internal clocks

        Args:
            policy_net (BlackNet): _description_
        """
        super(AgentGame, self).__init__(player_ace=False, policy=None)
        self.policy_net = policy_net
        self.state_tuple = namedtuple('State', ['hand_sum', 'usable_ace', 'dealer_card'])
        self.state_history = []
        self.time = 0.
        self.episode_timer = 0
        self.episode_terminated = False
        #self.current_state
        
    def updateCurrentState(self):
        """
        Sets a property called **current_state** based on the most recent element of **state_history** using a **State** _namedtuple_.
        """
        self.current_state = self.state_tuple((self.playerSums[-1]-4.)/27., float(self.playerAces[-1]), (self.dealerCard-1.)/9.)
        
    def evalPolicy(self) -> tc.Tensor:
        """
        Converts **current_state** so that the agent's hand value lies in the range [0., 1.], the usable ace status is either 0 or 1, and the dealer's top card
        is in the range [0., 1.]. Then evaluates the agent's **policy_net** on the current state and returns the index of the maximum output.

        Returns:
            tc.Tensor: The index of the larger of the two network outputs.
        """
        cur_stat = tc.tensor([self.current_state.hand_sum, self.current_state.usable_ace, self.current_state.dealer_card]).to(tc.float32)
        with tc.no_grad():
            return self.policy_net(cur_stat).max(-1).indices
        
    def agentDecision(self, epsilon: float):
        """
        Implements the agent's epsilon-greedy policy. When the agent uses it's own value function it acquires the state-action values from the **evalPolicy** method.
        Updates the current state based on the action chosen.

        Args:
            epsilon (float): The parameter determining whether the agent uses its value function or chooses randomly.
        """
        r = self.deck.rng.random()
        if r <= epsilon:
            p = self.deck.rng.choice([0,1])
            if p == 1:
                self.playerActions.append(1)
                self.player.dealToPlayer(self.deck.deal())
                self.player.updateHandSum()
                
                self.playerSums.append(self.player.handSum)
                self.playerAces.append(self.player.usableAce)
            elif p == 0:
                self.playerActions.append(0)
                self.playerPlaying = False
                self.episode_terminated = True
        
        else:
            a = self.evalPolicy()
            if a == 1:
                self.playerActions.append(1)
                self.player.dealToPlayer(self.deck.deal())
                self.player.updateHandSum()
                
                self.playerSums.append(self.player.handSum)
                self.playerAces.append(self.player.usableAce)
            elif a == 0:
                self.playerActions.append(0)
                self.playerPlaying = False
                self.episode_terminated = True
        self.updateCurrentState()
    
    def agentStep(self, epsilon:float):
        """
        Simulates the process of an agent interacting with the environment (receiving cards, hitting/staying, getting a reward). Used as an alternative to the _playerHand_
        method in the _Blackjack_ module.

        Args:
            epsilon (float): The epsilon in epsilon-greedy.

        Returns:
            sasr (tuple[namedtuple, int, namedtuple, float]): state, action, resulting state, reward
        """
        #Only want to initialize these things when the episode begins. Having it like this removes the need to have a separate initialization function.
        if self.episode_timer == 0:
            self.playerPlaing = True
            self.player.dealToPlayer(self.deck.deal())
            self.player.dealToPlayer(self.deck.deal())
            self.player.updateHandSum()
            self.playerSums.append(self.player.handSum)
            self.playerAces.append(self.player.usableAce)
                
            self.dealerHand()
            self.dealerCard = self.dealer.hand[0]
            
            if self.player.handSum == 21:
                self.playerNatural = True
                self.playerPlaying = False
                self.episode_terminated = True
                self.playerActions.append(0)
                self.reward = 1.5
                #self.state_history.append(self.current_state)
        
            self.updateCurrentState()
            self.state_history.append(self.current_state)
            #self.episode_timer += 1
        
        if self.playerPlaying:
            self.agentDecision(epsilon)
            self.state_history.append(self.current_state)
            self.episode_timer += 1
            if self.playerPlaying:
                if self.playerSums[-1] > 21:
                    #Player busts
                    self.playerBust = True
                    self.playerPlaying = False
                    self.episode_terminated = True
                    self.reward = -1.

            else:
                if (not self.playerBust) and self.dealerBust:
                    #Dealer busts
                    self.reward = 1.
                elif (not self.playerBust) and (not self.dealerBust):
                    if self.player.handSum > self.dealer.handSum:
                        #Player wins
                        self.reward = 1.
                    elif self.player.handSum < self.dealer.handSum:
                        #Dealer wins
                        self.reward = -1.
                    else:
                        #Draw
                        self.reward = 0.

        self.time += 1
        
        if self.episode_terminated and self.playerNatural:
            return self.state_history[-1], self.playerActions[-1], self.state_history[-1], self.reward
        elif self.episode_terminated:
            return self.state_history[-2], self.playerActions[-1], self.state_history[-1], self.reward

        else:
            return self.state_history[-2], self.playerActions[-1], self.state_history[-1], 0.
        
    def reset(self):
        self.player.clearHand()
        self.dealer.clearHand()
        self.playerNatural = False
        self.playerSums = []
        self.playerAces = []
        self.playerActions = []
        self.playerPlaying = True
        self.dealerPlaying = True
        self.playerBust = False
        self.episode_timer = 0
        self.state_history = []
        self.episode_terminated = False
        
class Model:
    def __init__(self, policy_net):
        """
        Creates an object containing an instance of the _AgentGame_ class and an instance of the _BlackNet_ class. In order to train the neural network contained
        inside the agent, the methods **create_memory** and **set_params** must be run with their corresponding arguments. 
        """
        self.policy_net = policy_net
        self.target_net = BlackNet()
        self.agent = AgentGame(policy_net=self.policy_net)
        #self.memory
        #self.batch_size
        #self.num_episodes
        #self.lr = lr
        #self.gamma = gamma
        #self.tau = tau
        #self.ep_start, self.ep_end, self.ep_rate
        
    def create_memory(self, capacity:int):
        """
        Generates a replay memory in the form of a _ReplayDataset_ object.

        Args:
            capacity (_int_): The length of the memory buffer. Once the number of SASR tuples inside the memory exceeds this amount, the earlier elements at the front of the list\
            get popped to make way for new ones.
        """
        self.memory = ReplayDataset(capacity=capacity)
        
    def set_params(self, batch_size:int, num_episodes:int, lr:float, gamma:float, tau:float, epsilon_data:tuple[float]):
        """
        Sets the parameters necessary to train the agent's neural network, as well as the optimizer for the training process.

        Args:
            batch_size (_int_): The number of SASR tuples to be included in a training mini-batch
            num_episodes (_int_): The number of hands of blackjack to train the agent on
            lr (_float_): The learning rate of the training process, which controls how large updates are to the neural networks' parameters
            gamma (_float_): The discount rate for future rewards in the reinforcement learning set up
            tau (_float_): A parameter controlling the soft update of the target network's parameters. Each update is a convex combination of the form\
                *(target's new parameters)* = *(tau)(target's old parameters) + (1-tau)(policy's new parameters)*
            epsilon_data (_tuple[float]_): Parameters which control the **epsilon_decay** method, consisting of (*ep_start*, *ep_end*, *ep_rate*)
        """
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.ep_start, self.ep_end, self.ep_rate = epsilon_data
        self.optimizer = tc.optim.Adam(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        
    def epsilon_decay(self, t:float) -> float:
        """
        Takes in a time step and returns the scheduled value of the epsilon-greedy policy's parameter.

        Args:
            t (float): Global time step, meant to be retrieved from the running timer built into the agent

        Returns:
            float: Scheduled value of epsilon
        """
        return self.ep_end + (self.ep_start - self.ep_end)*math.exp(-1.0 * self.ep_rate * t)
    
    def State_to_tensor(self, state) -> tc.Tensor:
        """
        Converts a **State** _namedtuple_ from the _AgentGame_ class into a _torch.Tensor_, taking **batch_size** into account.
        Args:
            state (_type_): _description_

        Returns:
            tc.Tensor: A tensor with shape (**batch_size**, 3) representing a mini-batch of game states.
        """
        return tc.cat([state.hand_sum.view(self.batch_size,1), state.usable_ace.view(self.batch_size,1), state.dealer_card.view(self.batch_size,1)], dim=1).to(tc.float32)
    
    def optimize_model(self):
        """
        Performs an on-line training step if the memory buffer has enough SASR tuples in it.
        
        A new _DataLoader_ object is created each time this is run because the memory dataset is altered each time as well. A single sample of size *batch_size* is taken from memory
        and a filter is created to indiciate which states are terminating states (i.e. states where the player stays or busts). The expected value of each non-terminating
        next_state is calculated using the target net, and terminating ones are assigned an expected value of 0. The target value is calculated by adding the next_state-expected value to
        the reward for the action taken. We use the _SmoothL1Loss_ (Huber) function to compute the loss, then perform back-prop, then update the parameters.
        """

        if len(self.memory) < self.batch_size:
            return

        loader = DataLoader(self.memory, batch_size=self.batch_size, shuffle=True)
        sample = next(iter(loader))
        
        non_final_mask = tc.tensor(tuple(map(lambda s: not s, sample.terminal)), dtype=tc.bool)
        non_final_next_states = self.State_to_tensor(sample.next_state)[non_final_mask,:]
        
        state_batch = self.State_to_tensor(sample.state)
        action_batch = sample.action
        reward_batch = sample.reward
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).to(tc.float32)
        
        next_state_values = tc.zeros(self.batch_size)
        with tc.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values.to(tc.float32)
            
        expected_state_action_values = (self.gamma * next_state_values) + reward_batch
        #criterion = nn.SmoothL1Loss()
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1).to(tc.float32))
        
        
        self.optimizer.zero_grad()
        loss.backward()
        
        tc.nn.utils.clip_grad_value_(self.agent.policy_net.parameters(), 50)
        self.optimizer.step()
        
    def train_agent(self):
        """
        Completes the training loop *num_episodes* times. Each hand, the agent is presented with the choice of hitting or staying, a decision which is made based on an epsilon-greedy
        policy where the state-action values are determined using the agent's policy network. After each decision the SASR tuple is loaded into memory and the model is optimized.
        """
        for ep in range(self.num_episodes):
            for t in count():
                state, action, next_state, reward = self.agent.agentStep(self.epsilon_decay(self.agent.time))
                terminated = self.agent.episode_terminated
                
                self.memory.push(state, action, next_state, reward, terminated)
                
                self.optimize_model()
                
                #Soft update the target network using the policy network's parameters
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.agent.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)
                
                if terminated:
                    self.agent.reset()
                    break
                
def epsilon_decay(initial:float, final:float, rate:float, t:float|int) -> float:
    """
    Calculates the scheduled value of the epsilon-greedy parameter at time **t** based on exponential decay.

    Args:
        initial (float): Starting scheduled value
        final (float): Final (asymptotic) scheduled value
        rate (float): Exponential decay parameter, or time constant
        t (float | int): Current time

    Returns:
        epsilon (float): Scheduled value of epsilon
    """
    return final + (initial - final)*math.exp(-1.0 * rate * t)

def agent_plays_episode(agent:AgentGame, mem:ReplayDataset, epsilon_data:tuple[float, float, float]):
    """
    Simulates **agent** playing a full hand of blackjack, storing each state, action, resulting state, and reward in a memory buffer.

    Args:
        agent (AgentGame):
        mem (ReplayDataset):
        epsilon_data (tuple[float, float, float]):
    """
    agent.reset()
    
    for t in count():
        state, action, next_state, reward = agent.agentStep(epsilon_decay(epsilon_data[0], epsilon_data[1], epsilon_data[2], agent.time))
        
        mem.push(state, action, next_state, reward, agent.episode_terminated)
        
        if agent.episode_terminated:
            break
        
def expected_winnings(agent:AgentGame, num_episodes:int, mode='win') -> float:
    """
    Simulates **num_episodes** games of blackjack played by **agent** and stores results in a buffer. Then, returns an accumulation
    of the rewards. Evaluates the agent's policy. 

    Args:
        agent (AgentGame): Meant to be the agent stored in a _Model_ object, presumably after training.
        num_episodes (int): Number of hands of blackjack simulated.
        mode (str, optional): Determines the form of the output. The options are:
        >'win': outputs the sum of the rewards
        >'loss': outputs the negative of the sum of the rewards
        >'loss percentage': outputs the negative of the sum of the rewards divided by the total number of episodes, then turned into a percentage
        Defaults to 'win'.
    """
    pholder_memory = ReplayDataset(20*num_episodes)
    for ep in range(num_episodes):
        agent_plays_episode(agent, pholder_memory, (0.0, 0.0, 0.))
    
    if mode == 'win':
        return pholder_memory.netWinnings()
    elif mode == 'loss':
        return -pholder_memory.netWinnings()
    elif mode == 'loss percentage':
        return -(pholder_memory.netWinnings() * 100)/num_episodes