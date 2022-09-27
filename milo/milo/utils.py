import torch
import numpy as np
import os
import os.path as osp
import random
from .logger import init_logger
import shutil
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter
import warnings
import matplotlib.pyplot as plt
#from milo.sampler import sample_points
from .sampler import sample_points

# ========================
# === Evaluation Utils ===
# ========================

def save_checkpoint(dirs, agent, cost_function, tag, agent_type='trpo'):
    save_dir = dirs['models_dir']
    checkpoint = {'policy_params': agent.policy.get_param_values(),
                  'old_policy_params': agent.policy.old_params,
                  'baseline_params': agent.baseline.model.state_dict(),
                  'baseline_optim': agent.baseline.optimizer.state_dict(),
                  'w': cost_function.w,
                  'rff': cost_function.rff.state_dict()}
    if agent_type == 'ppo':
        checkpoint['policy_optim'] = agent.optimizer.state_dict()
    torch.save(checkpoint, osp.join(save_dir, f'checkpoint_{tag}.pt'))


def save_sim_rewards(mb_step, dirs, int_mean, ext_mean, len_mean):
    sim_score_path = osp.join(dirs['data_dir'], 'sim_reward.pt')

    if not osp.exists(sim_score_path):
        saved_scores = {'int_mean': [int_mean],
                        'ext_mean': [ext_mean],
                        'len_mean': [len_mean],
                        'mb_step': [mb_step]}

    else:
        saved_scores = torch.load(sim_score_path, map_location=torch.device('cpu'))
        saved_scores['int_mean'].append(int_mean)
        saved_scores['ext_mean'].append(ext_mean)
        saved_scores['len_mean'].append(len_mean)
        saved_scores['mb_step'].append(mb_step)

    torch.save(saved_scores, sim_score_path)


def save_and_plot(n_iter, args, dirs, scores, mmds):
    mb_step = n_iter * args.samples_per_step * args.pg_iter
    scores_path = osp.join(dirs['data_dir'], 'scores.pt')
    mmds_path = osp.join(dirs['data_dir'], 'mmds.pt')
    # Scores
    if not osp.exists(scores_path):
        saved_scores = {'greedy': [scores['greedy']],
                        'sample': [scores['sample']],
                        'mb_step': [mb_step]}
    else:
        saved_scores = torch.load(scores_path, map_location=torch.device('cpu'))
        saved_scores['greedy'].append(scores['greedy'])
        saved_scores['sample'].append(scores['sample'])
        saved_scores['mb_step'].append(mb_step)
    torch.save(saved_scores, scores_path)
    # MMDS
    if not osp.exists(mmds_path):
        saved_mmds = {'greedy': [mmds['greedy']],
                      'sample': [mmds['sample']],
                      'mb_step': [mb_step]}
    else:
        saved_mmds = torch.load(mmds_path, map_location=torch.device('cpu'))
        saved_mmds['greedy'].append(mmds['greedy'])
        saved_mmds['sample'].append(mmds['sample'])
        saved_mmds['mb_step'].append(mb_step)
    torch.save(saved_mmds, mmds_path)

    # Plots
    plot_data(dirs, saved_scores, labels={'ylabel': 'Score', 'title': 'Performance', 'figname': 'scores.png'})
    plot_data(dirs, saved_mmds, labels={'ylabel': 'MMD', 'title': 'MMD', 'figname': 'mmds.png'})


def plot_data(dirs, data, labels, x_type='iter'):
    y = np.array(data['greedy'])
    if x_type == 'iter':
        x = np.arange(y.shape[0])
    elif x_type == 'timestep':
        x = np.array(data['mb_step'])
    plt.plot(x, y)
    plt.xlabel(x_type)
    plt.ylabel(labels['ylabel'])
    plt.title(labels['title'])
    plt.savefig(osp.join(dirs['plots_dir'], labels['figname']))
    plt.close()


# TODO add in args
def evaluate(n_iter, logger, writer, args, env, policy, reward_func, num_traj=10, imitate_amp=True):
    '''
    This function is used for evaluating policy in the real deepmimic/amp environment, env.

    '''
    # TODO: FIX DEEPMIMIC ARG
    logger.info(f"======Evaluating with seed: {args.seed + n_iter}=====")
    greedy_samples = sample_points(env=env, policy=policy, num_to_collect=num_traj, base_seed=args.seed + n_iter,
                                   num_workers=args.num_cpu, mode='trajectories', eval_mode=True, verbose=False,
                                   deepmimic=True)
    samples = sample_points(env=env, policy=policy, num_to_collect=num_traj, base_seed=args.seed + n_iter,
                            num_workers=args.num_cpu, mode='trajectories', eval_mode=False, verbose=False,
                            deepmimic=True)
    # Compute scores
    if imitate_amp:
        greedy_scores = np.array([traj['env_infos'][-1]['dtw_cost'] for traj in greedy_samples])
        sample_scores = np.array([traj['env_infos'][-1]['dtw_cost'] for traj in samples])
    else:
        greedy_scores = np.array([np.sum(traj['rewards']) for traj in greedy_samples])
        sample_scores = np.array([np.sum(traj['rewards']) for traj in samples])
    greedy_mean_lengths = np.mean([len(traj['rewards']) for traj in greedy_samples])
    sample_mean_lengths = np.mean([len(traj['rewards']) for traj in samples])
    greedy_mean, greedy_max, greedy_min = greedy_scores.mean(), greedy_scores.max(), greedy_scores.min()
    sample_mean, sample_max, sample_min = sample_scores.mean(), sample_scores.max(), sample_scores.min()

    # Compute MMD
    if args.cost_input_type == 'sa':
        greedy_x = np.concatenate(
            [np.concatenate([traj['observations'], traj['actions']], axis=1) for traj in greedy_samples], axis=0)
        sample_x = np.concatenate([np.concatenate([traj['observations'], traj['actions']], axis=1) for traj in samples],
                                  axis=0)
    elif args.cost_input_type == 'ss':
        greedy_x = np.concatenate(
            [np.concatenate([traj['observations'], traj['next_observations']], axis=1) for traj in greedy_samples],
            axis=0)
        sample_x = np.concatenate(
            [np.concatenate([traj['observations'], traj['next_observations']], axis=1) for traj in samples], axis=0)
    greedy_x = torch.from_numpy(greedy_x).float()
    sample_x = torch.from_numpy(sample_x).float()

    greedy_diff = reward_func.get_rep(greedy_x).mean(0) - reward_func.phi_e
    sample_diff = reward_func.get_rep(sample_x).mean(0) - reward_func.phi_e

    greedy_mmd = torch.dot(greedy_diff, greedy_diff)
    sample_mmd = torch.dot(sample_diff, sample_diff)

    # Log
    logger_score_name = 'DTW Cost' if imitate_amp else 'Reward'
    logger.info(
        f'Greedy Evaluation {logger_score_name} mean (min, max): {greedy_mean:.2f} ({greedy_min:.2f}, {greedy_max:.2f})')
    logger.info(f'Greedy Evaluation Trajectory Lengths: {greedy_mean_lengths:.2f}')
    logger.info(f'Greedy MMD: {greedy_mmd}')

    logger.info(
        f'Sampled Evaluation {logger_score_name} mean (min, max): {sample_mean:.2f} ({sample_min:.2f}, {sample_max:.2f})')
    logger.info(f'Sampled Evaluation Trajectory Lengths: {sample_mean_lengths:.2f}')
    logger.info(f'Sampled MMD: {sample_mmd}')
    # Compute average length
    writer_score_name = 'dtw_cost' if imitate_amp else 'reward'
    writer.add_scalars(f'data/inf_greedy_{writer_score_name}', {'min_score': greedy_min,
                                                                'mean_score': greedy_mean,
                                                                'max_score': greedy_max}, n_iter + 1)
    writer.add_scalar('data/inf_greedy_len', greedy_mean_lengths, n_iter + 1)
    writer.add_scalar('data/greedy_mmd', greedy_mmd, n_iter + 1)
    writer.add_scalars(f'data/inf_sampled_{writer_score_name}', {'min_score': sample_min,
                                                                 'mean_score': sample_mean,
                                                                 'max_score': sample_max}, n_iter + 1)
    writer.add_scalar('data/inf_sampled_len', sample_mean_lengths, n_iter + 1)
    writer.add_scalar('data/sampled_mmd', sample_mmd, n_iter + 1)

    scores = {'greedy': greedy_mean, 'sample': sample_mean}
    mmds = {'greedy': greedy_mmd, 'sample': sample_mmd}
    return scores, mmds


# =======================
# ==== Dataset Utils ====
# =======================
def convert_to_veltopos(path, deepmimic, is_db_mjrl, is_expert):
    x = torch.load(path)
    vel_offset = deepmimic.get_vel_offset()  # Should only need vel offset since each state is composed of phase -> pose -> vel
    dt = 1 / deepmimic.get_agent_update_rate()
    if is_db_mjrl:
        x[0][:, vel_offset:] *= dt
        if is_expert:
            x[1][:, vel_offset] *= dt
        else:
            x[2][:, vel_offset:] *= dt
    else:
        for i in x:
            # Trajs have the form {'episode':(all_state, action, reward), 'score':score} while expert have form {'episode'all_states}
            if is_expert:
                i['episode'][:, vel_ffset:] *= dt
            else:
                i['episode'][0][:, vel_offset:] *= dt  # state

    return x


def get_db_mjrl(db_path, num_trajs='all', idx=None, expert=False, imitate_amp=True):
    """
    This converts our saved db (which contains state, action, next state, reward)
    into torch tensors for state, action, next_state.

    """
    saved_db = torch.load(db_path)
    if num_trajs is 'all':
        num_trajs = len(saved_db)
    saved_db = saved_db[:num_trajs]
    if idx is not None:
        saved_db = saved_db[idx]
    states, actions, next_states, total_reward = [], [], [], 0
    for traj in saved_db:
        if expert:
            all_state = traj['episode']
            states.append(all_state[:-1])
            next_states.append(all_state[1:])
        else:
            all_state, action, _ = traj['episode']

            states.append(all_state[:-1])
            actions.append(action)
            next_states.append(all_state[1:])
            total_reward += traj['dtw_cost'] if imitate_amp else traj['ep_rew']
            mean_db_reward = total_reward / num_trajs
    if expert:
        db = (torch.from_numpy(np.concatenate(states, axis=0)).float(),
              torch.from_numpy(np.concatenate(next_states, axis=0)).float())
    else:
        db = (torch.from_numpy(np.concatenate(states, axis=0)).float(),
              torch.from_numpy(np.concatenate(actions, axis=0)).float(),
              torch.from_numpy(np.concatenate(next_states, axis=0)).float())
        print(
            f"{'DB DTW Cost' if imitate_amp else 'DB Mean Reward'}: {mean_db_reward} | DB # Samples: {db[0].shape[0]}")

    return db


# TODO: Do I need to copy before adding to output in get_db_mjrl and get_paths

def get_paths_mjrl(db_path, num_trajs='all', idx=None, expert=False):
    """
    This converts our saved db -> mjrl paths with keys 'observations' and 'actions'.
    Used to create paths that can be fed into BC.
    """
    # TODO ADD EXPERT
    saved_db = torch.load(db_path)
    if num_trajs is 'all':
        num_trajs = len(saved_db)
    saved_db = saved_db[:num_trajs]
    if idx is not None:
        saved_db = [saved_db[idx]]
    paths = []
    for traj in saved_db:
        if expert:
            all_state = traj['episode']
            paths.append({'observations': all_state[:-1], 'next_observation': all_state[1:]})
        else:
            all_state, action, _ = traj['episode']
            paths.append({'observations': all_state[:-1], 'actions': action})
    return paths


# =======================
# ==== Utils ====
# =======================
def set_global_seeds(seed):
    """
    Sets global seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_dynamics_id(args):
    """
    Creates an experiment id using parameters of trained dynamic model.
    """
    seed = args.seed
    dynamic_id = args.dynamic_id
    num_models = args.dynamic_num_models
    return f'id={dynamic_id}-seed={seed}-num_models={num_models}'


def get_milo_id(args):
    seed = args.seed
    lambda_b = args.lambda_b
    id = args.milo_id
    return f'id={id}-seed={seed}-lambda={lambda_b}'


def log_arguments(args, logger):
    """
    Adds arguments used for experiment in logger
    """
    headers = ['Args', 'Value']
    table = tabulate(list(vars(args).items()), headers=headers, tablefmt='pretty')
    logger.info(">>>>> Experiment Running with Arguments >>>>>")
    logger.info("\n" + table)


def setup(args, ask_prompt=True):
    """
    Directory is
    |- root_dir (experiments)
    |    |-Dynamics models
    |    |-    |-offline_dataset_name
    |    |-    |-     |- dynamic_experiment_id
    |    |-    |-     |-    |- logs
    |    |-    |-     |-    |- tensorboard_logs
    |    |-    |-     |-    |- checkpoints


    |    |- MILO models
    |    |-    |- expert_dataset_name
    |    |-    |-   |- MILO_experiment_id
    |    |-    |-   |-    |- logs
    |    |-    |-   |-    |- tensorboard_logs
    |    |-    |-   |-    |- checkpoints
    |    |-    |-   |-    |-  data
    """
    # ==============Setup Global seeds/threads/devices==============#Q
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.set_num_threads(1)
    set_global_seeds(args.seed)
    torch.set_default_tensor_type(torch.FloatTensor)
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # ==============Make root dynamic and milo directories==============#
    dynamics_dir = os.path.join(args.root_path, 'dynamics')
    milo_dir = os.path.join(args.root_path, 'milo')

    if not osp.isdir(args.root_path):
        os.makedirs(dynamics_dir)
        os.makedirs(milo_dir)

    # ==============Make dynamic model experiment directory==============#
    offline_dataset_name = args.offline_data[args.offline_data.rfind('/') + 1:args.offline_data.find('.pt')]
    dynamics_data_dir = os.path.join(dynamics_dir, offline_dataset_name)
    if not osp.isdir(dynamics_data_dir):
        os.makedirs(dynamics_data_dir)

    dynamic_experiment_id = get_dynamics_id(args)
    dynamic_experiment_dir = os.path.join(dynamics_data_dir, dynamic_experiment_id)
    load_dynamics = True if args.load_dynamics else False
    overwrite_log = False
    if osp.isdir(dynamic_experiment_dir):

        delete = True

        if ask_prompt:
            while True:
                reply = input("Dynamic Experiment already exists, delete existing directory? (y/n) ").strip().lower()
                if reply not in ['y', 'n', 'yes', 'no']:
                    print('Please reply with y, n, yes, or no')
                else:
                    delete = False if reply in ['n', 'no'] else True
                    break

        if delete:
            print('Deleting existing, duplicate dynamic experiment')
            overwrite_log = True
            shutil.rmtree(dynamic_experiment_dir)
        else:
            print('Dynamics directory already exists...')
            load_dynamics = True
    else:
        load_dynamics=False
    # ==============Creating dynamic experiment subdirectories==============#
    dynamic_logs_dir = osp.join(dynamic_experiment_dir, 'logs')
    dynamic_tensorboard_dir = osp.join(dynamic_experiment_dir, 'tensorboard_logs')
    dynamic_model_dir = osp.join(dynamic_experiment_dir, 'models_dir')

    dynamic_directories = {'logs_dir': dynamic_logs_dir, 'tensorboard_dir': dynamic_tensorboard_dir,
                           'models_dir': dynamic_model_dir}

    if not osp.isdir(dynamic_experiment_dir):
        print(">>>>>> Dynamic Experiment Directory not there. Creating...")
        os.makedirs(dynamic_experiment_dir)
        os.makedirs(dynamic_logs_dir)
        os.makedirs(dynamic_tensorboard_dir)
        os.makedirs(dynamic_model_dir)
    dynamic_logger = init_logger(dynamic_logs_dir)
    dynamic_writer = SummaryWriter(dynamic_tensorboard_dir)

    if overwrite_log:
        log_arguments(args, dynamic_logger)

    # ==============Make MILO directory==============#
    
    # ==============Make milo model experiment directory==============
    expert_dataset_name = args.expert_data[
                          args.expert_data.rfind('/') + 1:args.expert_data.find('.pt')]  # TODO FIX:
    # expert_dataset_name = args.expert_data[args.expert_data.rfind("\\") + 1:args.expert_data.find('.pt')]

    milo_data_dir = os.path.join(milo_dir, expert_dataset_name)
    if not osp.isdir(milo_data_dir):
        os.makedirs(milo_data_dir)
    milo_experiment_id = get_milo_id(args)
    milo_experiment_dir = os.path.join(milo_data_dir, milo_experiment_id)
    if osp.isdir(milo_experiment_dir):

        delete = True

        if ask_prompt:
            while True:
                reply = input(
                    "MILO Experiment already exists, delete existing directory? (y/n) ").strip().lower()
                if reply not in ['y', 'n', 'yes', 'no']:
                    print('Please reply with y, n, yes, or no')
                else:
                    delete = False if reply in ['n', 'no'] else True
                    break

        if delete:
            print('Deleting existing, duplicate MILO experiment')
            shutil.rmtree(milo_experiment_dir)

    # ==============Creating milo experiment subdirectories==============#
    milo_logs_dir = osp.join(milo_experiment_dir, 'logs')
    milo_tensorboard_dir = osp.join(milo_experiment_dir, 'tensorboard_logs')
    milo_model_dir = osp.join(milo_experiment_dir, 'models')
    milo_data_dir = osp.join(milo_experiment_dir, 'data')
    milo_plots_dir = osp.join(milo_experiment_dir, 'plots')
    milo_directories = {'logs_dir': milo_logs_dir, 'tensorboard_dir': milo_tensorboard_dir,
                        'models_dir': milo_model_dir,
                        'data_dir': milo_data_dir, 'plots_dir': milo_plots_dir}
    if not osp.isdir(milo_experiment_dir):
        os.makedirs(milo_experiment_dir)
        os.makedirs(milo_logs_dir)
        os.makedirs(milo_tensorboard_dir)
        os.makedirs(milo_model_dir)
        os.makedirs(milo_data_dir)
        os.makedirs(milo_plots_dir)
    milo_logger = init_logger(milo_logs_dir)
    milo_writer = SummaryWriter(milo_tensorboard_dir)
    log_arguments(args, milo_logger)

    loggers = {'dynamic': dynamic_logger, 'milo': milo_logger}
    writers = {'dynamic': dynamic_writer, 'milo': milo_writer}

    return milo_directories, dynamic_directories, load_dynamics, loggers, writers, device
