import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
if "../" not in sys.path:
  sys.path.append("../")

# from lib import plotting
from collections import deque, namedtuple

EpisodeStats = namedtuple("Stats",["episode_lengths", "option_lengths", "episode_rewards", "avg_q_value", "avg_discounted_return"])

env = gym.envs.make("Breakout-v0")
env.frameskip=4

# Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
# VALID_ACTIONS = [0, 1, 2, 3]
VALID_ACTIONS = range(env.action_space.n)

class StateProcessor():
    """
    Processes a raw Atari images. Resizes it and converts it to grayscale.
    """
    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State

        Returns:
            A processed [84, 84] state representing grayscale values.
        """
        return sess.run(self.output, { self.input_state: state })

class Estimator():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, num_options, scope="estimator", summaries_dir=None):
        self.scope = scope
        self.num_options = num_options
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our input
        # Our input are 4 RGB frames of shape 160, 160 each
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        self.options_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="options")

        X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        # Three convolutional layers
        conv1 = tf.contrib.layers.conv2d(
            X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, self.num_options*len(VALID_ACTIONS))
        self.predictions_reshaped = tf.reshape(self.predictions, (batch_size, self.num_options, len(VALID_ACTIONS)))

        # Get the predictions for the chosen option and actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + len(VALID_ACTIONS)*self.options_pl + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.losses.huber_loss(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        max_option_qs = tf.reduce_max(self.predictions_reshaped, axis=-1)
        option_q_summaries = [tf.summary.scalar(
            "max_q_value_option_{}".format(option),
            tf.reduce_max(max_option_qs[:,option])
            ) for option in range(self.num_options)]
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ] + option_q_summaries)


    def predict(self, sess, s):
        """
        Predicts action values.

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 4, 160, 160, 3]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated
          action values.
        """
        return sess.run(self.predictions_reshaped, { self.X_pl: s })

    def update(self, sess, s, a, o, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 4, 160, 160, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a , self.options_pl: o}
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss

def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


class EpsGreedyPolicy():
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.
        nO: Number of options

    Returns:

    """

    def __init__(self, nA, nO, zeta):
        self.nA = nA
        self.nO = nO
        self.zeta = zeta

    def beta_prob(self, q_values, option, epsilon):
        B = np.ones(2, dtype=float) * epsilon / 2
        q_beta0 = np.max(q_values[option]) * self.zeta
        q_beta1 = np.max(q_values)
        best_beta = np.argmax([q_beta0, q_beta1])
        B[best_beta] += (1.0-epsilon)
        return B


    def option_prob(self, q_values, epsilon):
        """
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
        """
        O = np.ones(self.nO, dtype=float) * epsilon / self.nO
        opt_q_values = np.max(q_values, axis=1)
        best_option = np.argmax(opt_q_values)
        O[best_option] += (1.0 - epsilon)
        return O

    def action_prob(self, q_values, option, epsilon):
        """
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
        """
        A = np.ones(self.nA, dtype=float) * epsilon / self.nA
        best_action = np.argmax(q_values[option])
        A[best_action] += (1.0 - epsilon)
        return A


def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    state_processor,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=500000,
                    replay_memory_init_size=50000,
                    update_target_estimator_every=10000,
                    discount_factor=0.99,
                    zeta=1.05,
                    num_options=8,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    batch_size=32,
                    record_video_every=200,
                    save_every=200,
                    test_every=20):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        sess: Tensorflow Session object
        env: OpenAI environment
        q_estimator: Estimator object used for the q values
        target_estimator: Estimator object used for the targets
        state_processor: A StateProcessor object
        num_episodes: Number of episodes to run for
        experiment_dir: Directory to save Tensorflow summaries in
        replay_memory_size: Size of the replay memory
        replay_memory_init_size: Number of random experiences to sampel when initializing
          the reply memory.
        update_target_estimator_every: Copy parameters from the Q estimator to the
          target estimator every N steps
        discount_factor: Gamma discount factor
        epsilon_start: Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value
        epsilon_end: The final minimum value of epsilon after decaying is done
        epsilon_decay_steps: Number of steps to decay epsilon over
        batch_size: Size of batches to sample from the replay memory
        record_video_every: Record a video every N episodes

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    Transition = namedtuple("Transition", ["state", "option", "action", "reward", "next_state", "done"])

    # The replay memory
    replay_memory = []

    # Keeps track of useful statistics
    training_stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        option_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        avg_q_value=np.zeros(num_episodes),
        avg_discounted_return=np.zeros(num_episodes))
    testing_stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes//test_every),
        option_lengths=np.zeros(num_episodes//test_every),
        episode_rewards=np.zeros(num_episodes//test_every),
        avg_q_value=np.zeros(num_episodes//test_every),
        avg_discounted_return=np.zeros(num_episodes//test_every))

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)

    saver = tf.train.Saver()
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    total_t = sess.run(tf.contrib.framework.get_global_step())

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = EpsGreedyPolicy(
        len(VALID_ACTIONS),
        num_options,
        zeta)

    # Populate the replay memory with initial experience
    print("Populating replay memory...")
    state = env.reset()
    state = state_processor.process(sess, state)
    state = np.stack([state] * 4, axis=2)
    # run the estimator and get all values
    q_values = q_estimator.predict(sess, np.expand_dims(state, 0))[0]
    option_done = True
    option = None
    for i in range(replay_memory_init_size):
        epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]
        # select a new option
        if option_done:
            option_probs = policy.option_prob(q_values, epsilon)
            option = np.random.choice(np.arange(len(option_probs)), p=option_probs)
        action_probs = policy.action_prob(q_values, option, epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
        next_state = state_processor.process(sess, next_state)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
        replay_memory.append(Transition(state, option, action, reward, next_state, done))
        if done:
            state = env.reset()
            state = state_processor.process(sess, state)
            state = np.stack([state] * 4, axis=2)
            option_done = True
            option = None
        else:
            # decide whether to terminate option
            beta_probs = policy.beta_prob(q_values, option, epsilon)
            option_done = np.random.choice(np.arange(2), p=beta_probs)
            state = next_state
        # get the next set of q values
        q_values = q_estimator.predict(sess, np.expand_dims(state, 0))[0]

    # Record videos
    # Use the gym env Monitor wrapper
    # env = Monitor(env,
                  # directory=monitor_path,
                  # resume=True,
                  # video_callable=lambda count: count % record_video_every ==0)

    for i_episode in range(num_episodes):
        start_time = time.time()

        # Save the current checkpoint
        if i_episode % save_every == 0:
            saver.save(tf.get_default_session(), checkpoint_path, global_step=global_step)

        if i_episode % test_every == 0:
            # run a testing episode
            state = env.reset()
            state = state_processor.process(sess, state)
            state = np.stack([state] * 4, axis=2)
            # run the estimator and get all values
            q_values = q_estimator.predict(sess, np.expand_dims(state, 0))[0]
            avg_q_value = np.mean(q_values)
            rewards = []

            option_done = True
            option = None
            option_switches = 0
            loss = None

            epsilon = 0.05
            # One step in the environment
            for t in itertools.count():
                if option_done:
                    # select a new option
                    option_probs = policy.option_prob(q_values, epsilon)
                    option = np.random.choice(np.arange(len(option_probs)), p=option_probs)
                    option_switches += 1
                # Take a step
                action_probs = policy.action_prob(q_values, option, epsilon)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
                next_state = state_processor.process(sess, next_state)
                next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
                # Update statistics
                testing_stats.episode_rewards[i_episode//test_every] += reward
                testing_stats.episode_lengths[i_episode//test_every] = t
                rewards.append(reward)
                if done:
                    break
                # get the next set of q values
                q_values = q_estimator.predict(sess, np.expand_dims(next_state, 0))[0]
                avg_q_value += np.mean(q_values)
                # decide whether to terminate option
                beta_probs = policy.beta_prob(q_values, option, epsilon)
                option_done = np.random.choice(np.arange(2), p=beta_probs)
                state = next_state

            testing_stats.option_lengths[i_episode//test_every] = float(t)/option_switches
            testing_stats.avg_q_value[i_episode//test_every] = avg_q_value/t
            avg_return = 0
            gamma_geometric_sum = 1
            for (k, r) in enumerate(rewards):
                avg_return += r * gamma_geometric_sum
                gamma_geometric_sum += np.power(discount_factor, k+1)
            testing_stats.avg_discounted_return[i_episode//test_every] = avg_return/t
            # Print out which step we're on, useful for debugging.
            print("\rTESTING @ Episode {}/{}, Steps {}, Reward: {}, Avg. Option Length: {:.1f}, Episode Length: {}".format(
                i_episode + 1, num_episodes, total_t,
                testing_stats.episode_rewards[i_episode//test_every],
                testing_stats.option_lengths[i_episode//test_every],
                testing_stats.episode_lengths[i_episode//test_every]),
                end="")
            sys.stdout.flush()

            # Add summaries to tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=testing_stats.episode_rewards[i_episode//test_every], node_name="testing/episode_reward", tag="testing/episode_reward")
            episode_summary.value.add(simple_value=testing_stats.episode_lengths[i_episode//test_every], node_name="testing/episode_length", tag="testing/episode_length")
            episode_summary.value.add(simple_value=testing_stats.option_lengths[i_episode//test_every], node_name="testing/option_length", tag="testing/option_length")
            episode_summary.value.add(simple_value=testing_stats.avg_discounted_return[i_episode//test_every], node_name="testing/avg_discounted_return", tag="testing/avg_discounted_return")
            episode_summary.value.add(simple_value=testing_stats.avg_q_value[i_episode//test_every], node_name="testing/avg_q_value", tag="testing/avg_q_value")
            q_estimator.summary_writer.add_summary(episode_summary, total_t)
            q_estimator.summary_writer.flush()


        # Reset the environment
        state = env.reset()
        state = state_processor.process(sess, state)
        state = np.stack([state] * 4, axis=2)
        # run the estimator and get all values
        q_values = q_estimator.predict(sess, np.expand_dims(state, 0))[0]
        avg_q_value = np.mean(q_values)
        rewards = []

        option_done = True
        option = None
        option_switches = 0
        loss = None

        # One step in the environment
        for t in itertools.count():

            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

            # Add epsilon to Tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=epsilon, tag="epsilon")
            q_estimator.summary_writer.add_summary(episode_summary, total_t)

            # Maybe update the target estimator
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, q_estimator, target_estimator)
                print("\nCopied model parameters to target network.")

            if option_done:
                # select a new option
                option_probs = policy.option_prob(q_values, epsilon)
                option = np.random.choice(np.arange(len(option_probs)), p=option_probs)
                option_switches += 1
            # Take a step
            action_probs = policy.action_prob(q_values, option, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
            next_state = state_processor.process(sess, next_state)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # Save transition to replay memory
            replay_memory.append(Transition(state, option, action, reward, next_state, done))

            # Update statistics
            training_stats.episode_rewards[i_episode] += reward
            training_stats.episode_lengths[i_episode] = t
            rewards.append(reward)

            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch_size)
            states_batch, option_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
            # clip rewards to (-1,1)
            reward_batch = np.clip(reward_batch, -1, 1)

            # print("option batch shape : ", option_batch.shape)
            # print("option batch values : ", option_batch)
            # Calculate q values and targets (Double DQN)
            q_values_batch = q_estimator.predict(sess, next_states_batch) # batch_size x noptions x nactions
            # print("Next q values shape: ", q_values_batch.shape)
            # print("Next q values : ", q_values_batch[0])
            # # best action for all options
            best_actions = np.argmax(q_values_batch, axis=2) # batch_size x noptions
            best_options = np.argmax(np.max(q_values_batch, axis=2), axis=1) # batch_size x 1
            # print("best actions shape: ", best_actions.shape)
            # print("best actions : ", best_actions[0])
            # print("best option shape: ", best_options.shape)
            # print("best option : ", best_options)
            # # best action per option
            best_action_option = best_actions[range(batch_size), option_batch] # batch_size x 1
            # print("best actions option shape: ", best_action_option.shape)
            # print("best actions option : ", best_action_option)

            q_beta0_batch = q_values_batch[range(batch_size), option_batch, best_action_option] # batch_size x 1
            # print("q_beta0_batch shape: ", q_beta0_batch.shape)
            # print("q_beta0_batch values : ", q_beta0_batch)
            q_beta1_batch = q_values_batch[range(batch_size), best_options, best_actions[range(batch_size), best_options]] # batch_size x 1
            # print("q_beta1_batch shape: ", q_beta1_batch.shape)
            # print("q_beta1_batch values : ", q_beta1_batch)
            beta_max_batch = np.where(q_beta1_batch > zeta*q_beta0_batch)
            # print("beta_max_batch shape: ", beta_max_batch.shape)
            # print("beta_max_batch values : ", beta_max_batch)
            # get target q values
            target_q_values_batch = target_estimator.predict(sess, next_states_batch)
            # print("target_q_values_batch shape: ", target_q_values_batch.shape)
            # print("target_q_values_batch values : ", target_q_values_batch[0])
            target_q_beta0_batch = target_q_values_batch[range(batch_size), option_batch, best_action_option]
            # print("target_q_beta0_batch shape: ", target_q_beta0_batch.shape)
            # print("target_q_beta0_batch values : ", target_q_beta0_batch)
            target_q_beta1_batch = target_q_values_batch[range(batch_size), best_options, best_actions[range(batch_size), best_options]]
            # print("target_q_beta1_batch shape: ", target_q_beta1_batch.shape)
            # print("target_q_beta1_batch values : ", target_q_beta1_batch)
            q_values_next_target = target_q_beta0_batch.copy()
            q_values_next_target[beta_max_batch] = target_q_beta1_batch[beta_max_batch]
            # print("q_values_next_target values : ", q_values_next_target)


            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                discount_factor * q_values_next_target
            # print("targets_batch shape: ", targets_batch.shape)
            # print("targets_batch values : ", targets_batch)

            # input()

            # Perform gradient descent update
            states_batch = np.array(states_batch)
            loss = q_estimator.update(sess, states_batch, action_batch, option_batch, targets_batch)

            if done:
                break
            # get the next set of q values
            q_values = q_estimator.predict(sess, np.expand_dims(next_state, 0))[0]
            avg_q_value += np.mean(q_values)
            # decide whether to terminate option
            beta_probs = policy.beta_prob(q_values, option, epsilon)
            option_done = np.random.choice(np.arange(2), p=beta_probs)

            state = next_state
            total_t += 1
        training_stats.option_lengths[i_episode] = float(t)/option_switches
        training_stats.avg_q_value[i_episode] = avg_q_value/t
        avg_return = 0
        gamma_geometric_sum = 1
        for (k, r) in enumerate(rewards):
            avg_return += r * gamma_geometric_sum
            gamma_geometric_sum += np.power(discount_factor, k+1)
        training_stats.avg_discounted_return[i_episode] = avg_return/t
        # Print out which step we're on, useful for debugging.
        print("\nEpisode {}/{}, Steps {}, Reward: {}, Avg. Option Length: {:.1f}, Episode Length: {}, Avg. time per step (ms) {:.0f}".format(
            i_episode + 1, num_episodes, total_t,
            training_stats.episode_rewards[i_episode],
            training_stats.option_lengths[i_episode],
            training_stats.episode_lengths[i_episode],
	    1000*(time.time() - start_time)/t),
            end="")
        sys.stdout.flush()

        # Add summaries to tensorboard
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=training_stats.episode_rewards[i_episode], node_name="training/episode_reward", tag="training/episode_reward")
        episode_summary.value.add(simple_value=training_stats.episode_lengths[i_episode], node_name="training/episode_length", tag="training/episode_length")
        episode_summary.value.add(simple_value=training_stats.option_lengths[i_episode], node_name="training/option_length", tag="training/option_length")
        episode_summary.value.add(simple_value=training_stats.avg_discounted_return[i_episode], node_name="training/avg_discounted_return", tag="training/avg_discounted_return")
        episode_summary.value.add(simple_value=training_stats.avg_q_value[i_episode], node_name="training/avg_q_value", tag="training/avg_q_value")
        q_estimator.summary_writer.add_summary(episode_summary, total_t)
        q_estimator.summary_writer.flush()

    # env.monitor.close()
    return training_stats


tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create estimators
NUM_OPTIONS = 8
q_estimator = Estimator(NUM_OPTIONS, scope="q", summaries_dir=experiment_dir)
target_estimator = Estimator(NUM_OPTIONS, scope="target_q")

# State processor
state_processor = StateProcessor()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    deep_q_learning(sess,
        env,
        q_estimator=q_estimator,
        target_estimator=target_estimator,
        state_processor=state_processor,
        experiment_dir=experiment_dir,
        num_episodes=1000000,
        replay_memory_size=1000000,
        replay_memory_init_size=50000,
        update_target_estimator_every=10000,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=1000000,
        discount_factor=0.99,
        zeta=1.05,
        num_options=NUM_OPTIONS,
        batch_size=32)
