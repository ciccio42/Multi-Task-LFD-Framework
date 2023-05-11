import warnings
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
import os
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import functools
from torch.multiprocessing import Pool, set_start_method
from utils import *
from robosuite_env.controllers.expert_nut_assembly import \
    get_expert_trajectory as nut_expert
from robosuite_env.controllers.expert_pick_place import \
    get_expert_trajectory as place_expert
import sys
import pickle as pkl
import json
import wandb
from collections import OrderedDict

set_start_method('forkserver', force=True)
sys.path.append('/home/Multi-Task-LFD-Framework/repo/mosaic/tasks/test_models')
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# python dataset_analysis.py --model /home/ciccio/Desktop/multi_task_lfd/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/baseline-1/1Task-Pick-Place-Stable-Policy-Batch32-1gpu-Attn2ly128-Act2ly256mix4-headCat-simclr128x512 --step 72900 --task_indx 12 --debug


def single_run(model, agent_env, context, img_formatter, variation, show_image, agent_trj, indx):
    start, end = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)
    eval_fn = TASK_MAP[args.task_name]['eval_fn']
    with torch.no_grad():
        start.record()
        traj, info, context = eval_fn(model=model,
                                      env=agent_env,
                                      context=context,
                                      gpu_id=0,
                                      variation_id=variation,
                                      img_formatter=img_formatter,
                                      max_T=150,
                                      agent_traj=agent_trj,
                                      model_act=True,
                                      show_img=show_image)
        print("Evaluated traj #{}, task#{}, reached? {} picked? {} success? {} ".format(
            indx, variation, info['reached'], info['picked'], info['success']))
        end.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = start.elapsed_time(end)
        print(f"Elapsed time {curr_time}")

    return traj, info


def run_inference(model, conf_file, task_name, task_indx, results_dir_path, training_trj, show_image, experiment_number, file_pair):

    model.cuda(0)
    indx = file_pair[1]
    print(f"Index {indx}")
    demo_file = file_pair[0][2]
    demo_file_name = demo_file.split('/')[-1]
    agent_file = file_pair[0][3]
    agent_file_name = agent_file.split('/')[-1]
    task_indx = task_indx if task_indx != -1 else int(file_pair[0][1])
    results_analysis = [task_name, task_indx, demo_file_name, agent_file_name]

    # print(f"----\nDemo file {demo_file}\nAgent file {agent_file}\n----")
    # open demo and agent data
    with open(demo_file, "rb") as f:
        demo_data = pickle.load(f)
    with open(agent_file, "rb") as f:
        agent_data = pickle.load(f)

    # get target object id
    demo_target = demo_data['traj'].get(0)['obs']['target-object']
    agent_target = agent_data['traj'].get(0)['obs']['target-object']
    if demo_target != agent_target:
        print(f"Sample indx {indx} different target objects")

    # get env function
    env_func = TASK_MAP[task_name]['env_fn']
    agent_name = TASK_MAP[task_name]['agent-teacher'][0]
    variation = task_indx
    ret_env = True
    heights = conf_file.dataset_cfg.height
    widths = conf_file.dataset_cfg.width
    agent_env = create_env(env_fn=env_func, agent_name=agent_name,
                           variation=variation, ret_env=ret_env)

    img_formatter = build_tvf_formatter(conf_file, task_name)

    if training_trj:
        agent_trj = agent_data['traj']
    else:
        agent_trj = None

    if experiment_number == 1:
        cnt = 5
        np.random.seed(0)
    elif experiment_number == 2:
        cnt = 10
        np.random.seed(0)

    for i in range(cnt):
        # select context frames
        context = select_random_frames(
            demo_data['traj'], 4, sample_sides=True, experiment_number=experiment_number)
        # perform normalization on context frames
        context = [img_formatter(i[:, :, ::-1]/255)[None] for i in context]
        if isinstance(context[0], np.ndarray):
            context = torch.from_numpy(
                np.concatenate(context, 0)).float()[None]
        else:
            context = torch.cat(context, dim=0).float()[None]

        traj, info = single_run(model=model,
                                agent_env=agent_env,
                                context=context,
                                img_formatter=img_formatter,
                                variation=variation,
                                show_image=show_image,
                                agent_trj=agent_trj,
                                indx=indx)
        results_analysis.append(info)
        info['demo_file'] = demo_file_name
        info['agent_file'] = agent_file_name
        info['task_name'] = task_name
        pkl.dump(traj, open(results_dir_path +
                 '/traj{}_{}.pkl'.format(indx, i), 'wb'))
        pkl.dump(context, open(results_dir_path +
                 '/context{}_{}.pkl'.format(indx, i), 'wb'))
        res = {}
        for k, v in info.items():
            if v == True or v == False:
                res[k] = int(v)
            else:
                res[k] = v
        json.dump(res, open(results_dir_path +
                  '/traj{}_{}.json'.format(indx, i), 'w'))

    del model
    return results_analysis


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--step', type=int)
    parser.add_argument('--task_indx', type=int)
    parser.add_argument('--results_dir', type=str, default="/")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--project_name', type=str, default=None)
    parser.add_argument('--task_name', type=str, default="pick_place")
    parser.add_argument('--experiment_number', type=int, default=1,
                        help="1: Take samples from list and run 10 times with different demonstrator frames; 2: Take all the file from the training set")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--training_trj', action='store_true')
    parser.add_argument('--show_img', action='store_true')
    parser.add_argument('--run_inference', action='store_true')

    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    # 1. Load Training Dataset
    conf_file = OmegaConf.load(os.path.join(args.model, "config.yaml"))
    # 2. Get the dataset
    dataset = load_trajectories(conf_file=conf_file, mode='train')
    # 3. Load model
    model = load_model(model_path=args.model,
                       step=args.step, conf_file=conf_file)
    model.eval()

    task_name = args.task_name

    # try all training-samples
    file_pairs = [(dataset.all_file_pairs[indx], indx)
                  for indx in range(len(dataset.all_file_pairs))]

    if args.experiment_number == 1 or args.experiment_number == 2:
        if args.project_name:
            model_name = f"{args.model.split('/')[-1]}-{args.step}"
            run = wandb.init(project=args.project_name,
                             job_type='test', group=model_name.split("-1gpu")[0])
            run.name = model_name + f'-Test_{model_name}-Step_{args.step}'
            wandb.config.update(args)

        results_dir_path = os.path.join(
            args.results_dir, f"results_{task_name}", str(f"task-{args.task_indx}"))
        try:
            os.makedirs(results_dir_path)
        except:
            pass

        # model, conf_file, task_name, task_indx, results_dir_path, training_trj, show_image, file_pair
        f = functools.partial(run_inference, model, conf_file, task_name, args.task_indx,
                              results_dir_path, args.training_trj, args.show_img, args.experiment_number)

        if args.num_workers > 1:
            with Pool(args.num_workers) as p:
                task_success_flags = p.map(f, file_pairs)
        else:
            task_success_flags = [f(file_pair) for file_pair in file_pairs]

        # init results analysis dict
        final_results = OrderedDict()
        final_results[task_name] = OrderedDict()
        final_results[task_name][args.task_indx] = OrderedDict()

        for result in task_success_flags:
            demo_file = result[2]
            agent_file = result[3]
            if demo_file not in final_results[task_name][args.task_indx].keys():
                final_results[task_name][args.task_indx][demo_file] = OrderedDict()

            final_results[task_name][args.task_indx][demo_file][agent_file] = OrderedDict(
            )
            cnt_reached = 0
            cnt_picked = 0
            cnt_success = 0
            for run, single_run_result in enumerate(result[4:]):
                final_results[task_name][args.task_indx][demo_file][agent_file][run] = OrderedDict(
                )
                final_results[task_name][args.task_indx][demo_file][agent_file][run]['reached'] = int(
                    single_run_result['reached'])
                final_results[task_name][args.task_indx][demo_file][agent_file][run]['picked'] = int(
                    single_run_result['picked'])
                final_results[task_name][args.task_indx][demo_file][agent_file][run]['success'] = int(
                    single_run_result['success'])
                cnt_reached = cnt_reached + \
                    1 if int(single_run_result['reached']
                             ) == 1 else cnt_reached
                cnt_picked = cnt_picked + \
                    1 if int(single_run_result['picked']) == 1 else cnt_picked
                cnt_success = cnt_success + \
                    1 if int(single_run_result['success']
                             ) == 1 else cnt_success

            log = OrderedDict()
            log[agent_file] = OrderedDict()
            log[agent_file]['avg_success'] = cnt_success/run
            log[agent_file]['avg_reached'] = cnt_reached/run
            log[agent_file]['avg_picked'] = cnt_picked/run
            wandb.log(log)

        with open(f"{results_dir_path}/results.json", "w") as f:
            json.dump(final_results, f, indent=4)
