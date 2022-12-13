import os
import sys
import csv
import multiprocessing as mp

import evogym.envs

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

LIB_DIR = os.path.join(ROOT_DIR, 'libs')
sys.path.append(LIB_DIR)
import libs.ns_neat as ns_neat

UTIL_DIR = os.path.join(ROOT_DIR, 'utils')
sys.path.append(UTIL_DIR)
from gym_utils import load_robot
from figure_drawer import EvogymControllerDrawerNEAT, pool_init_func
from experiment_utils import load_experiment

from arguments.ns_neat import get_figure_args


def main():

    args = get_figure_args()

    expt_path = os.path.join('out', 'ns_neat', args.name)
    expt_args = load_experiment(expt_path)


    robot = load_robot(expt_args['robot'], task=expt_args['task'])


    decode_function = ns_neat.FeedForwardNetwork.create


    config_file = os.path.join(expt_path, 'ns_neat.cfg')
    config = ns_neat.make_config(config_file)


    genome_ids = {}
    if args.specified is not None:
        genome_ids = {
            'specified': [args.specified]
        }
    else:
        files = {
            'reward': 'history_reward.csv',
            'novelty': 'history_novelty.csv'
        }
        for metric,file in files.items():

            history_file = os.path.join(expt_path, file)
            with open(history_file, 'r') as f:
                reader = csv.reader(f)
                histories = list(reader)[1:]
                ids = sorted(list(set([hist[1] for hist in histories])))
                genome_ids[metric] = ids


    figure_path = os.path.join(expt_path, 'figure')
    draw_kwargs = {}
    if args.save_type=='gif':
        draw_kwargs = {
            'resolution': (1280*args.resolution_ratio, 720*args.resolution_ratio),
        }
    elif args.save_type=='jpg':
        draw_kwargs = {
            'interval': args.interval,
            'resolution_scale': args.resolution_scale,
            'timestep_interval': args.timestep_interval,
            'distance_interval': args.distance_interval,
            'display_timestep': args.display_timestep,
        }
    drawer = EvogymControllerDrawerNEAT(
        save_path=figure_path,
        env_id=expt_args['task'],
        robot=robot,
        genome_config=config.genome_config,
        decode_function=decode_function,
        overwrite=not args.not_overwrite,
        save_type=args.save_type, **draw_kwargs)

    draw_function = drawer.draw


    if not args.no_multi and args.specified is None:

        lock = mp.Lock()
        pool = mp.Pool(args.num_cores, initializer=pool_init_func, initargs=(lock,))
        jobs = []

        for metric,ids in genome_ids.items():
            for key in ids:
                genome_file = os.path.join(expt_path, 'genome', f'{key}.pickle')
                jobs.append(pool.apply_async(draw_function, args=(key, genome_file), kwds={'directory': metric}))

        for job in jobs:
            job.get(timeout=None)


    else:

        lock = mp.Lock()
        lock = pool_init_func(lock)

        for metric,ids in genome_ids.items():
            for key in ids:
                genome_file = os.path.join(expt_path, 'genome', f'{key}.pickle')
                draw_function(key, genome_file, directory=metric)

if __name__=='__main__':
    main()
