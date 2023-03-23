import cv2
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import hydra, os
import gc
import copy
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import glob
import functools
import random
from collections import deque
from collections import OrderedDict
from mosaic.datasets import Trajectory
from torch.multiprocessing import Pool, set_start_method
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms.functional import resized_crop
from robosuite import load_controller_config
from robosuite_env.controllers.expert_basketball import \
    get_expert_trajectory as basketball_expert
from robosuite_env.controllers.expert_nut_assembly import \
    get_expert_trajectory as nut_expert
from robosuite_env.controllers.expert_pick_place import \
    get_expert_trajectory as place_expert 
from robosuite_env.controllers.expert_block_stacking import \
    get_expert_trajectory as stack_expert 
from robosuite_env.controllers.expert_drawer import \
    get_expert_trajectory as draw_expert 
from robosuite_env.controllers.expert_button import \
    get_expert_trajectory as press_expert 
from robosuite_env.controllers.expert_door import \
    get_expert_trajectory as door_expert 
import robosuite.utils.transform_utils as T
import sys
import pickle as pkl
import json
import wandb
from utils import *
set_start_method('forkserver', force=True)
sys.path.append('/home/Multi-Task-LFD-Framework/repo/mosaic/tasks/test_models')
from eval_functions import *
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

# pick-place
agent_target = ["traj039", "traj051", "traj085", "traj092"]
demo_target = ["traj027", "traj051", "traj039"]
# target object on the center right [97200, 97201, 97224, 97204]
# target object on the right [97290, 97291, 97294, 97314]
# target object on the left [97470, 97471, 97474, 97494]
# nut assembly
# agent_target = ["traj039", "traj051", "traj059"]
# demo_target = ["traj051", "traj059"]
# target object on the center [16290, 16291, 16304]
# target object on the right [17460, 17461, 17474]
SAMPLE_LIST= [97470, 97471, 97474, 97494]


def perform_detection(model, config, ctr, 
    heights=100, widths=200, size=0, shape=0, color=0, max_T=60, env_name='place', gpu_id=-1, baseline=None, variation=None, seed=None):
    if gpu_id == -1:
        gpu_id = int(ctr % torch.cuda.device_count())
    model = model.cuda(gpu_id)

    img_formatter = build_tvf_formatter(config, env_name)
     
    T_context = config.train_cfg.dataset.get('T_context', None)
    random_frames = config.dataset_cfg.get('select_random_frames', False)
    if not T_context:
        assert 'multi' in config.train_cfg.dataset._target_, config.train_cfg.dataset._target_
        T_context = config.train_cfg.dataset.demo_T
    
    env, context, variation_id, expert_traj = build_env_context(
        img_formatter,
        T_context=T_context, ctr=ctr, env_name=env_name, 
        heights=heights, widths=widths, size=size, shape=shape, color=color, gpu_id=gpu_id, variation=variation, random_frames=random_frames)
    
    build_task = TASK_MAP.get(env_name, None)
    assert build_task, 'Got unsupported task '+env_name
    eval_fn = build_task['eval_fn']
    traj, info = eval_fn(model, env, context, gpu_id, variation_id, img_formatter, baseline=baseline)
    print("Evaluated traj #{}, task#{}, reached? {} picked? {} success? {} ".format(ctr, variation_id, info['reached'], info['picked'], info['success']))
    return traj, info, expert_traj, context

def _proc(model, config, results_dir, heights, widths, size, shape, color, env_name, baseline, variation, seed, n):
    json_name = results_dir + '/traj{}.json'.format(n)
    pkl_name  = results_dir + '/traj{}.pkl'.format(n)
    if os.path.exists(json_name) and os.path.exists(pkl_name):
        f = open(json_name)
        task_success_flags = json.load(f)
        print("Using previous results at {}. Loaded eval traj #{}, task#{}, reached? {} picked? {} success? {} ".format( 
            json_name, n, task_success_flags['variation_id'], task_success_flags['reached'], task_success_flags['picked'], task_success_flags['success']))
    else:
        rollout, task_success_flags, expert_traj, context = perform_detection(
            model, config, n, heights, widths, size, shape, color, 
            max_T=60, env_name=env_name, baseline=baseline, variation=variation, seed=seed)
        pkl.dump(rollout, open(results_dir+'/traj{}.pkl'.format(n), 'wb'))
        pkl.dump(expert_traj, open(results_dir+'/demo{}.pkl'.format(n), 'wb'))
        pkl.dump(context, open(results_dir+'/context{}.pkl'.format(n), 'wb')) 
        json.dump({k: int(v) for k, v in task_success_flags.items()}, open(results_dir+'/traj{}.json'.format(n), 'w'))
    del model
    return task_success_flags

def inference(model, env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=None, seed=None, agent_traj=None, model_act=False, show_img=False, experiment_number=1):

    done, states, images, context, obs, traj, tasks = \
        startup_env(model, env, context, gpu_id, variation_id, baseline=baseline, seed=seed)
    n_steps = 0
    
    if agent_traj is not None:
        # change object position
        print("Set object position based on training sample")
        init_env(env=env, traj=agent_traj, task_name="pick_place")
    else:
        print("Set object position randomly")

    object_name = env.objects[env.object_id].name
    obj_delta_key = object_name + '_to_robot0_eef_pos'
    obj_key = object_name + '_pos'
    start_z = obs[obj_key][2]
    t = 0

    if baseline and len(states) >= 5:
            images = []

    if show_img:
        # convert context from torch tensor to numpy
        context_frames = torch_to_numpy(context)
        number_of_context_frames = len(context_frames)
        demo_height, demo_width, _ = context_frames[0].shape
        # Determine the number of columns and rows to create the grid of frames
        num_cols = 2  # Example value, adjust as needed
        num_rows = (number_of_context_frames + num_cols - 1) // num_cols
        # Create the grid of frames
        frames = []
        for i in range(num_rows):
            row_frames = []
            for j in range(num_cols):
                index = i * num_cols + j
                if index < number_of_context_frames:
                    frame = context_frames[index][:,:,::-1]
                    row_frames.append(frame)
            row = cv2.hconcat(row_frames)
            frames.append(row)
        new_image = np.array(cv2.resize(cv2.vconcat(frames), (demo_width, demo_height)), np.uint8)
        output_frame = cv2.hconcat([new_image, obs['image'][:,:,::-1]])
        # showing the image
        cv2.imshow(f'Frame {t}', output_frame)
        t += 1
        # waiting using waitKey method
        cv2.waitKey(1000)
        cv2.destroyAllWindows()


    images.append(img_formatter(obs['image'][:,:,::-1]/255)[None])
    if isinstance(images[-1], np.ndarray):
        i_t = torch.from_numpy(np.concatenate(images, 0).astype(np.float32))[None] 
    else:
        i_t = images[0][None]
    i_t = i_t.cuda(gpu_id).float()

        
    with torch.no_grad():
        out = model(images=i_t, context=context, eval=True) # to avoid computing ATC loss

    env.close()
    del env
    del states
    del images
    del model
    torch.cuda.empty_cache()
    return traj, tasks, context


def single_run(model, agent_env, context, img_formatter, variation, show_image, agent_trj, indx):
    
    with torch.no_grad():
        traj, info, context =  inference(model=model, env=agent_env, context=context, gpu_id=0, variation_id=variation, img_formatter=img_formatter, max_T=60, agent_traj=agent_trj, model_act=True, show_img=show_image)
    return 

def run_inference(model, conf_file, task_name, task_indx, results_dir_path, training_trj, show_image, experiment_number, file_pair):

    model.cuda(0)
    indx = file_pair[1]
    print(f"Index {indx}")
    demo_file = file_pair[0][2]
    demo_file_name = demo_file.split('/')[-1]
    agent_file = file_pair[0][3]
    agent_file_name = agent_file.split('/')[-1]

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
    # compute demo_target_obj_position
    demo_target_obj_pos = -1
    for id, obj_name in enumerate(ENV_OBJECTS[task_name]['obj_names']):
        if id == demo_target:
            # get object position
            if task_name == 'nut_assembly':
                if id == 0:
                    pos = demo_data['traj'].get(1)['obs']['round-nut_pos']
                else:
                    pos = demo_data['traj'].get(1)['obs'][f'round-nut-{id+1}_pos']
            else:
                pos = demo_data['traj'].get(1)['obs'][f'{obj_name}_pos']        
            for i, pos_range in enumerate(ENV_OBJECTS[task_name]["ranges"]):
                if pos[1] >= pos_range[0] and pos[1] <= pos_range[1]: 
                    demo_target_obj_pos = i
            break   

    # get env function
    env_func = TASK_MAP[task_name]['env_fn']
    agent_name = TASK_MAP[task_name]['agent-teacher'][0]
    variation = task_indx
    ret_env = True
    heights = conf_file.dataset_cfg.height
    widths = conf_file.dataset_cfg.width
    agent_env = create_env(env_fn=env_func, agent_name=agent_name, variation=variation, ret_env=ret_env, heights=heights, widths=widths)
    
    img_formatter = build_tvf_formatter(conf_file, task_name)

    if training_trj:
        agent_trj = agent_data['traj']
    else:
        agent_trj = None   
    
    if experiment_number == 1:
        cnt = 10
        np.random.seed(0)
    elif experiment_number == 5:
        cnt = 10
        np.random.seed(0)
    else:
        cnt = 1
    for i in range(cnt):
        # select context frames
        context = select_random_frames(demo_data['traj'], 4, sample_sides=True, experiment_number=experiment_number)
        # perform normalization on context frames
        context = [img_formatter(i[:,:,::-1]/255)[None] for i in context]
        if isinstance(context[0], np.ndarray):
            context = torch.from_numpy(np.concatenate(context, 0)).float()[None]
        else:
            context = torch.cat(context, dim=0).float()[None]
        
        traj, info = single_run(model=model, agent_env=agent_env, context=context, img_formatter=img_formatter, variation=variation, show_image=show_image, agent_trj=agent_trj, indx=indx)
        results_analysis.append(info)
        info['demo_file'] = demo_file_name
        info['agent_file'] = agent_file_name
        info['task_name'] = task_name
        pkl.dump(traj, open(results_dir_path+'/traj{}_{}.pkl'.format(indx, i), 'wb'))
        pkl.dump(context, open(results_dir_path+'/context{}_{}.pkl'.format(indx, i), 'wb'))
        res = {}
        for k, v in info.items():
            if v==True or v==False:
                res[k] = int(v)
            else:
                res[k] = v
        json.dump(res, open(results_dir_path+'/traj{}_{}.json'.format(indx, i), 'w'))

    del model
    return results_analysis


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--wandb_log', action='store_true')
    parser.add_argument('--project_name', '-pn', default="mosaic", type=str)
    parser.add_argument('--config', default='')
    parser.add_argument('--N', default=-1, type=int)
    parser.add_argument('--use_h', default=-1, type=int)
    parser.add_argument('--use_w', default=-1, type=int)
    parser.add_argument('--num_workers', default=3, type=int)
    parser.add_argument('--size', action='store_true') # for block stacking only!
    parser.add_argument('--shape', action='store_true')
    parser.add_argument('--color', action='store_true')
    parser.add_argument('--env' , '-e', default='door', type=str)
    parser.add_argument('--eval_each_task',  default=30, type=int)
    parser.add_argument('--eval_subsets',  default=0, type=int)
    parser.add_argument('--saved_step', '-s', default=1000, type=int)
    parser.add_argument('--baseline', '-bline', default=None, type=str, help='baseline uses more frames at each test-time step')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--results_name', default=None, type=str)
    parser.add_argument('--variation', default=None, type=int)
    parser.add_argument('--seed', default=None, type=int)

    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    args = parser.parse_args()

    # 1. Load Training Dataset
    conf_file = OmegaConf.load(os.path.join(args.model, "config.yaml"))
    # 2. Get the dataset
    dataset = load_trajectories(conf_file=conf_file)
    # 3. Load model
    model = load_model(model_path=args.model, step=args.step, conf_file=conf_file)
    model.eval()
    # get the sample indices for the desired subtask
    task_name = args.task_name #dataset.subtask_to_idx.keys()
    if args.task_indx < 10:
        variation_str = f"{args.task_indx}"
    else:
        variation_str = str(args.task_indx)

    

    if args.experiment_number==1 or args.experiment_number==4:
        # use specific indices from the list
        subtask_indices = SAMPLE_LIST
        file_pairs = [(dataset.all_file_pairs[indx], indx) for indx in subtask_indices]
    elif args.experiment_number==2:
        # Try a specific sub-task
        subtask_indices = dataset.subtask_to_idx[task_name][f"task_{variation_str}"]
        file_pairs = [(dataset.all_file_pairs[indx], indx) for indx in subtask_indices]
    elif args.experiment_number == 3:  
        for sample_indx in dataset.all_file_pairs.keys():
            sample = dataset.all_file_pairs[sample_indx]
            sample_task_name = sample[0]
            sample_task_id = sample[1]
            sample_demo_file = sample[2]
            sample_agent_file = sample[3]
            if sample_task_name == task_name and sample_task_id == args.task_indx:
                for demo_target_trj in demo_target:
                    if demo_target_trj in sample_demo_file:
                        for agent_target_trj in agent_target:
                            if agent_target_trj in sample_agent_file:
                                print(f"Sample number {sample_indx} - Demo file {demo_target_trj} - Agent file {agent_target_trj}")

    if  args.experiment_number==1 or args.experiment_number==2 or args.experiment_number==5:
        if args.project_name:
            model_name = f"{args.model.split('/')[-1]}-{args.step}"
            run = wandb.init(project=args.project_name, job_type='test', group=model_name.split("-1gpu")[0])
            run.name = model_name + f'-Test_{model_name}-Step_{args.step}' 
            wandb.config.update(args)

        results_dir_path = os.path.join(args.results_dir, f"results_{task_name}", str(f"task-{args.task_indx}"))
        try:
            os.makedirs(results_dir_path)
        except:
            pass

    # model, conf_file, task_name, task_indx, results_dir_path, training_trj, show_image, file_pair
    f =  functools.partial(run_inference, model, conf_file, task_name, args.task_indx, results_dir_path, args.training_trj, args.show_img, args.experiment_number)

    if args.num_workers > 1:
        with Pool(args.num_workers) as p:
            task_success_flags = p.map(f, file_pairs)
    else:
        task_success_flags = [f(file_pair) for file_pair in file_pairs]