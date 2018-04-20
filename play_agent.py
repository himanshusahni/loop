import numpy as np
import gym
import fourrooms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

env = gym.make('Fourrooms-v0')
nactions = env.action_space.n

# load trained q values
q = np.load('q.npy')
fig, ax = plt.subplots(1)
ax.set_xlim([0,13])
ax.set_ylim([0,13])
walls = []
for i in range(13):
    ax.axvline(x=i)
    ax.axhline(y=i)
walls = np.where(env.env.occupancy == 1)
wall_patches = []
for (y,x) in zip(*walls):
    wall_patches.append(patches.Rectangle((x,12-y), 1, 1))

pc = PatchCollection(wall_patches, facecolor='gray')
ax.add_collection(pc)
print(env.env.tocell[env.env.goal])

while(1):
    s = env.reset()
    done = False
    while not done:
        # render the environment
        (y,x) = env.env.tocell[s]
        plt.plot(x+0.5, 12-y+0.5, 'ro')
        plt.savefig('ex.png')
        exit()
