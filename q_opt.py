import numpy as np
import gym
import fourrooms
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', help='Discount factor', type=float, default=0.99)
parser.add_argument('--lr', help="Learning rate", type=float, default=1e-2)
parser.add_argument('--inter_eps', help="Epsilon-greedy for policy over options", type=float, default=5e-2)
parser.add_argument('--intra_eps', help="Epsilon-greedy for intra option policy", type=float, default=5e-2)
parser.add_argument('--beta_eps', help="Epsilon-greedy for termination", type=float, default=5e-2)
parser.add_argument('--zeta', help="Regularization on termination value", type=float, default=1e-2)
parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=250)
parser.add_argument('--nruns', help="Number of runs", type=int, default=100)
parser.add_argument('--nsteps', help="Maximum number of steps per episode", type=int, default=1000)
parser.add_argument('--noptions', help='Number of options', type=int, default=4)
parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=1e-2)

args = parser.parse_args()

env = gym.make('Fourrooms-v0')
nactions = env.action_space.n
nstates = env.observation_space.n

def eps_greedy_action(q, s, w, eps=0.):
    '''
    epsilon greedy selection of actions
    '''
    if np.random.random() < eps:
        return np.random.randint(0, nactions)
    else:
        return int(np.argmax([q[s, w, a] for a in range(nactions)]))

def eps_greedy_option(q, s, eps=0.):
    '''
    epsilon greedy selection of option
    '''
    if np.random.random() < eps:
        return np.random.randint(0, args.noptions)
    else:
        q_ws = [np.max([q[s, w, a] for a in range(nactions)]) for w in range(args.noptions)]
        return int(np.argmax(q_ws))

def eps_greedy_beta(q_beta0, q_beta1, eps=0.):
    if np.random.random() < eps:
        opt_done = True if np.random.random() < 0.5 else False
        return opt_done
    else:
        return (q_beta1 > q_beta0)

history = np.zeros((args.nruns, args.nepisodes, 4))
for run in range(args.nruns):
    q = np.zeros((nstates, args.noptions, nactions))
    for episode in range(args.nepisodes):
        s = env.reset()
        opt_done = True
        cumreward = 0.
        option_switches = 0
        for step in range(args.nsteps):
            if opt_done:
                # pick an option eps-greedy
                w = eps_greedy_option(q, s, eps=1.-float(episode)/args.nepisodes)
                option_switches += 1
            # pick an action eps-greedy
            a = eps_greedy_action(q, s, w, eps=1.-float(episode)/args.nepisodes)
            s_p, reward, done, _ = env.step(a)
            cumreward += reward

            # value for continuing current option
            q_beta0 = max([q[s_p, w, a_p] for a_p in range(nactions)]) + args.zeta
            # value for termination and selecting a new option
            q_beta1 = max([q[s_p, w_p, a_p] for a_p in range(nactions) for w_p in range(args.noptions)])
            # off-policy update
            q[s, w, a] = q[s, w, a] + args.lr*(reward + (1. - done)*args.gamma*np.max([q_beta0, q_beta1]) - q[s, w, a])

            if done:
                break

            # off-policy option termination
            opt_done = eps_greedy_beta(q_beta0, q_beta1, eps=1.-float(episode)/args.nepisodes)
            # advance state
            s = s_p
        optionduration = step/option_switches
        history[run, episode, 0] = cumreward
        history[run, episode, 1] = step
        history[run, episode, 2] = optionduration
        history[run, episode, 3] = np.mean(q)
        # save history and q values at every episode
        np.save('history.npy', history)
        np.save('q.npy', q)
        np.save('episode.npy', np.array([episode]))
