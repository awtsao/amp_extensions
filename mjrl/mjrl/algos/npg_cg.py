import logging
#logging.disable(logging.CRITICAL)
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spLA
import copy
import time as timer
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy

# samplers
import mjrl.samplers.core as trajectory_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog
from mjrl.utils.cg_solve import cg_solve
from mjrl.algos.batch_reinforce import BatchREINFORCE

import pdb


class NPG(BatchREINFORCE):
    def __init__(self, env, policy, baseline,
                 normalized_step_size=0.01,
                 const_learn_rate=None,
                 FIM_invert_args={'iters': 10, 'damping': 1e-4},
                 hvp_sample_frac=1.0,
                 seed=123,
                 save_logs=False,
                 kl_dist=None,
                 input_normalization=None,
                 bc_args={'flag': False, 'expert_paths': None, 'reg_coeff': None},
                 **kwargs
                 ):
        """
        All inputs are expected in mjrl's format unless specified
        :param normalized_step_size: Normalized step size (under the KL metric). Twice the desired KL distance
        :param kl_dist: desired KL distance between steps. Overrides normalized_step_size.
        :param const_learn_rate: A constant learn rate under the L2 metric (won't work very well)
        :param FIM_invert_args: {'iters': # cg iters, 'damping': regularization amount when solving with CG
        :param hvp_sample_frac: fraction of samples (>0 and <=1) to use for the Fisher metric (start with 1 and reduce if code too slow)
        :param seed: random seed
        """

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.alpha = const_learn_rate
        self.n_step_size = normalized_step_size if kl_dist is None else 2.0 * kl_dist
        self.seed = seed
        self.save_logs = save_logs
        self.FIM_invert_args = FIM_invert_args
        self.hvp_subsample = hvp_sample_frac
        self.running_score = None
        if save_logs: self.logger = DataLog()
        # input normalization (running average)
        self.input_normalization = input_normalization
        if self.input_normalization is not None:
            if self.input_normalization > 1 or self.input_normalization <= 0:
                self.input_normalization = None

        # BC regularization args
        self.bc_args = bc_args

    def CPI_surrogate(self, observations, actions, advantages, infos=None):
        # Jonathan: Overwritten to incorporate BC reg. For now keeping it to just TRPO and not PPO
        adv_var = Variable(torch.from_numpy(advantages).float(), requires_grad=False)
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        LR = self.policy.likelihood_ratio(new_dist_info, old_dist_info)
        surr = torch.mean(LR*adv_var)
        infos['old_dist_info']=old_dist_info
        infos['new_dist_info'] = new_dist_info
        infos['lr'] = LR
        
        if self.bc_args['flag']:
            expert_obs = self.bc_args['expert_paths']['observations']
            expert_act = self.bc_args['expert_paths']['actions']
            LL, _, _ = self.policy.new_dist_info(expert_obs, expert_act)
            # Maximize log likelihood
            surr = surr + self.bc_args['reg_coeff']*torch.mean(LL)
        return surr

    def HVP(self, observations, actions, vector, regu_coef=None):
        regu_coef = self.FIM_invert_args['damping'] if regu_coef is None else regu_coef
        vec = Variable(torch.from_numpy(vector).float(), requires_grad=False)
        if self.hvp_subsample is not None and self.hvp_subsample < 0.99:
            num_samples = observations.shape[0]
            rand_idx = np.random.choice(num_samples, size=int(self.hvp_subsample*num_samples))
            obs = observations[rand_idx]
            act = actions[rand_idx]
        else:
            obs = observations
            act = actions
        old_dist_info = self.policy.old_dist_info(obs, act)
        new_dist_info = self.policy.new_dist_info(obs, act)
        mean_kl = self.policy.mean_kl(new_dist_info, old_dist_info)
        grad_fo = torch.autograd.grad(mean_kl, self.policy.trainable_params, create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_fo])
        h = torch.sum(flat_grad*vec)
        hvp = torch.autograd.grad(h, self.policy.trainable_params)
        hvp_flat = np.concatenate([g.contiguous().view(-1).data.numpy() for g in hvp])
        return hvp_flat + regu_coef*vector

    def build_Hvp_eval(self, inputs, regu_coef=None):
        def eval(v):
            full_inp = inputs + [v] + [regu_coef]
            Hvp = self.HVP(*full_inp)
            return Hvp
        return eval

    # ----------------------------------------------------------
    def train_from_paths(self, paths, infos):

        observations, actions, advantages, base_stats, self.running_score = self.process_paths(paths)
        infos['advantages']=advantages
        if self.save_logs: self.log_rollout_statistics(paths)

        # Keep track of times for various computations
        t_gLL = 0.0
        t_FIM = 0.0

        # normalize inputs if necessary
        if self.input_normalization:
            data_in_shift, data_in_scale = np.mean(observations, axis=0), np.std(observations, axis=0)
            pi_in_shift, pi_in_scale = self.policy.model.in_shift.data.numpy(), self.policy.model.in_scale.data.numpy()
            pi_out_shift, pi_out_scale = self.policy.model.out_shift.data.numpy(), self.policy.model.out_scale.data.numpy()
            pi_in_shift = self.input_normalization * pi_in_shift + (1-self.input_normalization) * data_in_shift
            pi_in_scale = self.input_normalization * pi_in_scale + (1-self.input_normalization) * data_in_scale
            self.policy.model.set_transformations(pi_in_shift, pi_in_scale, pi_out_shift, pi_out_scale)

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(observations, actions, advantages, infos).data.numpy().ravel()[0]
        infos['surr_before'] = surr_before
        # VPG
        ts = timer.time()
        vpg_grad = self.flat_vpg(observations, actions, advantages, infos)
        t_gLL += timer.time() - ts

        # NPG
        ts = timer.time()
        hvp = self.build_Hvp_eval([observations, actions],
                                  regu_coef=self.FIM_invert_args['damping'])
        npg_grad = cg_solve(hvp, vpg_grad, x_0=vpg_grad.copy(),
                            cg_iters=self.FIM_invert_args['iters'])
        t_FIM += timer.time() - ts

        # Step size computation
        # --------------------------
        if self.alpha is not None:
            alpha = self.alpha
            n_step_size = (alpha ** 2) * np.dot(vpg_grad.T, npg_grad)
        else:
            n_step_size = self.n_step_size
            alpha = np.sqrt(np.abs(self.n_step_size / (np.dot(vpg_grad.T, npg_grad) + 1e-20)))
            infos['vpg_grad']=vpg_grad
            infos['npg_grad']=npg_grad
        # Policy update
        # --------------------------
        curr_params = self.policy.get_param_values()
        new_params = curr_params + alpha * npg_grad
        self.policy.set_param_values(new_params, set_new=True, set_old=False)
        surr_after = self.CPI_surrogate(observations, actions, advantages, infos).data.numpy().ravel()[0]
        infos['surr_after'] = surr_after
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(new_params, set_new=True, set_old=True)

        # Log information
        if self.save_logs:
            self.logger.log_kv('alpha', alpha)
            self.logger.log_kv('delta', n_step_size)
            self.logger.log_kv('time_vpg', t_gLL)
            self.logger.log_kv('time_npg', t_FIM)
            self.logger.log_kv('kl_dist', kl_dist)
            self.logger.log_kv('surr_improvement', surr_after - surr_before)
            self.logger.log_kv('running_score', self.running_score)
            infos['surr_improvement'] = surr_after - surr_before
            infos['running_score'] = self.running_score
            infos['kl_dist'] = kl_dist
            infos['alpha'] = alpha

            '''
            #evaluate_success not implmented
            try:
                self.env.env.env.evaluate_success(paths, self.logger)
            except:
                # nested logic for backwards compatibility. TODO: clean this up.
                try:
                    success_rate = self.env.env.env.evaluate_success(paths)
                    self.logger.log_kv('success_rate', success_rate)
                except:
                    pass
            '''

        return base_stats