import os
import numpy as np

from gym_utils import make_vec_envs


class EvogymControllerEvaluatorNS:
    def __init__(self, env_id, robot, num_eval=1):
        self.env_id = env_id
        self.robot = robot
        self.num_eval = num_eval

    def evaluate_controller(self, key, controller, generation):
        env = make_vec_envs(self.env_id, self.robot, 0, 1)

        obs = env.reset()

        obs_data = []
        act_data = []

        episode_rewards = []
        episode_data = []
        while len(episode_rewards) < self.num_eval:
            action = np.array(controller.activate(obs[0]))*2 - 1
            obs_data.append(obs)
            act_data.append(action)
            obs, _, done, infos = env.step([np.array(action)])

            if 'episode' in infos[0]:
                obs_data = np.vstack(obs_data)
                obs_cov = self.calc_covar(obs_data)

                act_data = np.clip(np.vstack(act_data),-1,1)
                act_cov = self.calc_covar(act_data, align=False)

                data = np.hstack([obs_cov,act_cov])
                episode_data.append(data)

                obs_data = []
                act_data = []

                reward = infos[0]['episode']['r']
                episode_rewards.append(reward)

        env.close()

        results = {
            'reward': np.mean(episode_rewards),
            'data': np.mean(np.vstack(episode_data),axis=0)
        }
        return results

    @staticmethod
    def calc_covar(vec, align=True):
        ave = np.mean(vec,axis=0)
        if align:
            vec_align = (vec-ave).T
        else:
            vec_align = vec.T
        comb_indices = np.tril_indices(vec.shape[1],k=0)
        covar = np.mean(vec_align[comb_indices[0]]*vec_align[comb_indices[1]],axis=1)
        return covar
