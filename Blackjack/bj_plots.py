import numpy as np
import matplotlib.pyplot as plt
from Blackjack import ValueFunction

def valuePlotsMC(num_eps, usable_ace):
    '''
    Creates an instance of the ValueFunction class and runs through the training for num_eps (int) episodes. Outputs a plot of the simulated state-value function for each choice of
    usable_ace (bool).
    '''
    t = ValueFunction(num_eps,player_ace=False, testing=False)
    t.run()
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')

    x = np.arange(1., 11.)
    y = np.arange(12., 22.)
    X,Y = np.meshgrid(x,y)
    Z = t.averageValue[:,:,int(usable_ace)]

    ax.plot_wireframe(X, Y, Z)
    
    ax.set_xlabel('Dealer card')
    ax.set_ylabel('Player hand sum')
    ax.set_zlabel('State value')
    
    ax.set_box_aspect(None, zoom=0.95)
    
    ax.set_title('Usable ace: ' + f'{usable_ace}')
    plt.tight_layout()
    plt.show()