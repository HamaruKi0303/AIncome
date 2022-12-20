import os
import numpy as np
import tensorflow as tf


def total_episode_reward_logger2(rew_acc, rewards, masks, writer, steps, n_steps=128, train_num=None):
    """
    calculates the cumulated episode reward, and prints to tensorflow log the output

    :param rew_acc: (np.array float) the total running reward
    :param rewards: (np.array float) the rewards
    :param masks: (np.array bool) the end of episodes
    :param writer: (TensorFlow Session.writer) the writer to log to
    :param steps: (int) the current timestep
    :return: (np.array float) the updated total running reward
    :return: (np.array float) the updated total running reward
    """

    #print(">>>>> step : {}".format(steps))
    #print("rewards: {}".format(rewards))

    with tf.variable_scope("environment_info", reuse=True):
        for env_idx in range(rewards.shape[0]):
            dones_idx = np.sort(np.argwhere(masks[env_idx]))

            """
            print("masks    : {}".format(masks))
            print("dones_idx: {}".format(dones_idx))
            print("dones_idx: {}".format(len(dones_idx)))
            """

            if len(dones_idx) == 0:
                rew_acc[env_idx] += sum(rewards[env_idx])
            else:
                rew_acc[env_idx] += sum(rewards[env_idx, :dones_idx[0, 0]])
                summary = tf.Summary(value=[tf.Summary.Value(tag="episode_reward", simple_value=rew_acc[env_idx])])
               
                stepA = int((int(steps/(train_num)) -1)*train_num + (env_idx+1)*(train_num/rewards.shape[0]))

                """
                print("---=== step+   : {}".format(steps + dones_idx[0, 0]))
                print("--->>> step    : {}".format(steps))
                print("------ env_idx : {}".format(env_idx))
                print("---*** stepA   : {}".format(stepA))
                """

                writer.add_summary(summary, stepA)

                for k in range(1, len(dones_idx[:, 0])):
                    rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[k - 1, 0]:dones_idx[k, 0]])
                    summary = tf.Summary(value=[tf.Summary.Value(tag="episode_reward", simple_value=rew_acc[env_idx])])
                    #writer.add_summary(summary, steps + dones_idx[k, 0])
                    writer.add_summary(summary, steps + dones_idx[k, 0])
                    print("---*** step : {}".format(steps + dones_idx[k, 0]))
                rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[-1, 0]:])

    return rew_acc