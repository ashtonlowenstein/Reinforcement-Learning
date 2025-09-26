This repository includes original code I'm writing while reading "Reinforcement Learning: An Introduction" by Sutton and Barto. So far it includes:
 - Simulations of the n Armed bandit problem exploring different agent policies (epsilon-greedy, UCB) and different value functions and update rules. It's broken into two general approaches:
 action-value estimation and numerical preference approximation.
 - State value estimation in Blackjack using a specific policy (agent hits on any sum less than 20). The states are specified by the agent's hand sum in the range 12-21 (no real decisions are
   needed if the sum is less than 12 because hitting has no risk, and no decision can be made once the agent busts), the dealer's face-up card (which allows the agent to estimate the dealer's
   hand in the more complete reinforcement learning problem), and whether or not the agent has an ace whose value is 11.
