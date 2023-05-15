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
import sys
import pickle as pkl
import json
from utils import *
import wandb
set_start_method('forkserver', force=True)
sys.path.append('/home/Multi-Task-LFD-Framework/repo/mosaic/tasks/test_models')
from eval_functions import *
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

# python dataset_analysis.py --model /home/ciccio/Desktop/multi_task_lfd/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/baseline-1/1Task-Pick-Place-Stable-Policy-Batch32-1gpu-Attn2ly128-Act2ly256mix4-headCat-simclr128x512 --step 72900 --task_indx 12 --debug

TASK_MAP = {
    'basketball': {
        'num_variations':   12, 
        'env_fn':   basketball_expert,
        'eval_fn':  basketball_eval,
        'agent-teacher': ('PandaBasketball', 'SawyerBasketball'),
        'render_hw': (100, 180),  
        },
    'nut_assembly':  {
        'num_variations':   9, 
        'env_fn':   nut_expert,
        'eval_fn':  nut_assembly_eval,
        'agent-teacher': ('PandaNutAssemblyDistractor', 'SawyerNutAssemblyDistractor'),
        'render_hw': (100, 180), 
        },
    'pick_place': {
        'num_variations':   16, 
        'env_fn':   place_expert,
        'eval_fn':  pick_place_eval,
        'agent-teacher': ('PandaPickPlaceDistractor', 'SawyerPickPlaceDistractor'),
        'render_hw': (100, 180), #(150, 270)
        },
    'stack_block': {
        'num_variations':   6, 
        'env_fn':   stack_expert,
        'eval_fn':  block_stack_eval,
        'agent-teacher': ('PandaBlockStacking', 'SawyerBlockStacking'),
        'render_hw': (100, 180), ## older models used 100x200!!
        },
    'drawer': {
        'num_variations':   8,
        'env_fn':   draw_expert,
        'eval_fn':  draw_eval,
        'agent-teacher': ('PandaDrawer', 'SawyerDrawer'),
        'render_hw': (100, 180),
    },
    'button': {
        'num_variations':   6,
        'env_fn':   press_expert,
        'eval_fn':  press_button_eval,
        'agent-teacher': ('PandaButton', 'SawyerButton'),
        'render_hw': (100, 180),
    },
    'door': {
        'num_variations':   4,
        'env_fn':   door_expert,
        'eval_fn':  open_door_eval,
        'agent-teacher': ('PandaDoor', 'SawyerDoor'),
        'render_hw': (100, 180),
    },
}

# pick-place
#agent_target = ["traj039", "traj051", "traj085", "traj092"]
#demo_target = ["traj027", "traj051", "traj039"]
# target object on the center right [97200, 97201, 97224, 97204]
# target object on the right [97290, 97291, 97294, 97314]
# target object on the left [97470, 97471, 97474, 97494]
# nut assembly
# agent_target = ["traj039", "traj051", "traj059"]
# demo_target = ["traj051", "traj059"]
# target object on the center [16290, 16291, 16304]
# target object on the right [17460, 17461, 17474]
SAMPLE_LIST= [97470, 97471, 97474, 97494]


OBJECT_DISTRIBUTION = {
    'pick_place':{
        'milk' : [0, 0, 0, 0],
        'bread' : [0, 0, 0, 0],
        'cereal' : [0, 0, 0, 0],
        'can' : [0, 0, 0, 0],
        'ranges':  [[0.16, 0.19], [0.05, 0.09], [-0.08, -0.03], [-0.19, -0.15]]
    },
    'nut_assembly':{
        'nut0': [0, 0, 0],
        'nut1': [0, 0, 0],
        'nut2': [0, 0, 0],
        'ranges': [[0.10, 0.31], [-0.10, 0.10], [-0.31, -0.10]]
    }
}

def build_tvf_formatter(config, env_name='stack'):
    """Use this for torchvision.transforms in multi-task dataset, 
    note eval_fn always feeds in traj['obs']['images'], i.e. shape (h,w,3)
    """
    dataset_cfg = config.train_cfg.dataset
    height, width = dataset_cfg.get('height', 100), dataset_cfg.get('width', 180)
    task_spec = config.get(env_name, dict())
    crop_params = task_spec.get('crop', [0,0,0,0])
    top, left = crop_params[0], crop_params[2]
    def resize_crop(img):
        if len(img.shape) == 4:
            img = img[0]
        img_h, img_w = img.shape[0], img.shape[1]
        assert img_h != 3 and img_w != 3, img.shape 
        box_h, box_w = img_h - top - crop_params[1], img_w - left - crop_params[3]
        
        img = img.copy()
        obs = ToTensor()(img)
        obs = resized_crop(obs, top=top, left=left, height=box_h, width=box_w,
                        size=(height, width))
 
        obs = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(obs)
        
        return obs 
    return resize_crop 

def create_env(env_fn, agent_name, variation, ret_env, seed=None, heights=100, widths=180, gpu_id=0):
    if seed is None:
        create_seed = random.Random(None)
        create_seed = create_seed.getrandbits(32)
    else:
        create_seed = seed
    print(f"Creating environment with variation {variation}")
    controller = load_controller_config(default_controller='IK_POSE')
    return env_fn( agent_name, controller_type=controller, task=variation, ret_env=True, seed=create_seed, heights=heights, widths=widths, gpu_id=gpu_id)

def select_random_frames(frames, n_select, sample_sides=True, experiment_number=1):
    selected_frames = []
    if experiment_number != 5:
        clip = lambda x : int(max(0, min(x, len(frames) - 1)))
        per_bracket = max(len(frames) / n_select, 1)
        for i in range(n_select):
            n = clip(np.random.randint(int(i * per_bracket), int((i + 1) * per_bracket)))
            if sample_sides and i == n_select - 1:
                n = len(frames) - 1
            elif sample_sides and i == 0:
                n = 0
            selected_frames.append(n)
    elif experiment_number == 5:
        for i in range(n_select):
            # get first frame
            if i == 0:
                n = 0
            # get the last frame
            elif i == n_select - 1:
                n = len(frames) - 1
            elif i == 1:
                obj_in_hand = 0
                # get the first frame with obj_in_hand and the gripper is closed
                for t in range(1, len(frames)):
                    state = frames.get(t)['info']['status']
                    trj_t = frames.get(t)
                    gripper_act = trj_t['action'][-1] 
                    if state == 'obj_in_hand' and gripper_act == 1:
                        obj_in_hand = t
                        n = t
                        break
            elif i == 2:
                # get the middle moving frame
                start_moving = 0
                end_moving = 0
                for t in range(obj_in_hand, len(frames)):
                    state = frames.get(t)['info']['status']
                    if state == 'moving' and start_moving == 0:
                        start_moving = t
                    elif state != 'moving' and start_moving != 0 and end_moving == 0:
                        end_moving = t
                        break
                n = start_moving + int((end_moving-start_moving)/2)
            
            selected_frames.append(n)

    if isinstance(frames, (list, tuple)):
        return [frames[i] for i in selected_frames]
    elif isinstance(frames, Trajectory):     
        return [frames[i]['obs']['image'] for i in selected_frames]
        #return [frames[i]['obs']['image-state'] for i in selected_frames]
    return frames[selected_frames]

def single_run(model, agent_env, context, img_formatter, variation, show_image, agent_trj, indx):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    eval_fn = TASK_MAP[args.task_name]['eval_fn']
    with torch.no_grad():
        start.record()
        traj, info, context =  eval_fn(model=model, env=agent_env, context=context, gpu_id=0, variation_id=variation, img_formatter=img_formatter, max_T=60, agent_traj=agent_trj, model_act=True, show_img=show_image)
        print("Evaluated traj #{}, task#{}, reached? {} picked? {} success? {} ".format(indx, variation, info['reached'], info['picked'], info['success']))
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
    parser.add_argument('--model', type=str)
    parser.add_argument('--step', type=int)
    parser.add_argument('--task_indx', type=int)
    parser.add_argument('--results_dir', type=str, default="/")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--project_name', type=str, default=None)
    parser.add_argument('--task_name', type=str, default="pick_place")
    parser.add_argument('--experiment_number', type=int, default=1, help="1: Take samples from list and run 10 times with different demonstrator frames; 2: Take all the file from the training set")
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
    
    if args.experiment_number==1 or args.experiment_number==5:
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
    elif args.experiment_number == 4:
        # Compute object position distribution, the counter is incremented when the object is the target one
        cnt = 0
        for task_indx in dataset.agent_files[task_name]:
            agent_files = dataset.agent_files[task_name][task_indx]
            for agent_file in agent_files:
                # load pickle file
                with open(agent_file, "rb") as f:
                    agent_file_data = pickle.load(f)
                # take trj
                trj = agent_file_data['traj']
                # take target object id
                target_obj_id = trj[1]['obs']['target-object']
                y_ranges = OBJECT_DISTRIBUTION[task_name]['ranges']
                object_distribution = OBJECT_DISTRIBUTION[task_name]
                for id, obj_name in enumerate(object_distribution.keys()):
                    if id == target_obj_id:
                        # get object position
                        if task_name == 'nut_assembly':
                            if id == 0:
                                pos = trj[1]['obs']['round-nut_pos']
                            else:
                                pos = trj[1]['obs'][f'round-nut-{id+1}_pos']
                        else:
                            pos = trj[1]['obs'][f'{obj_name}_pos']        
                        prev_cnt = cnt
                        for i, pos_range in enumerate(y_ranges):
                            if pos[1] >= pos_range[0] and pos[1] <= pos_range[1]:
                                cnt += 1
                                object_distribution[obj_name][i] += 1
                        if prev_cnt == cnt:
                            print(f"Task {task_indx} - {agent_file}")
        print(cnt)
        print(object_distribution)
       
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

        # init results analysis dict
        final_results = OrderedDict()
        final_results[task_name] = OrderedDict()
        final_results[task_name][args.task_indx] = OrderedDict()

        for result in task_success_flags:
            demo_file = result[2]
            agent_file = result[3]
            if demo_file not in final_results[task_name][args.task_indx].keys():
                final_results[task_name][args.task_indx][demo_file] = OrderedDict()

            final_results[task_name][args.task_indx][demo_file][agent_file] = OrderedDict()
            cnt_reached = 0
            cnt_picked = 0
            cnt_success = 0
            for run, single_run_result in enumerate(result[4:]):
                final_results[task_name][args.task_indx][demo_file][agent_file][run] = OrderedDict()
                final_results[task_name][args.task_indx][demo_file][agent_file][run]['reached'] = int(single_run_result['reached'])
                final_results[task_name][args.task_indx][demo_file][agent_file][run]['picked'] = int(single_run_result['picked'])
                final_results[task_name][args.task_indx][demo_file][agent_file][run]['success'] = int(single_run_result['success'])
                cnt_reached = cnt_reached + 1 if  int(single_run_result['reached']) == 1 else cnt_reached
                cnt_picked = cnt_picked + 1 if  int(single_run_result['picked']) == 1 else cnt_picked
                cnt_success = cnt_success + 1 if  int(single_run_result['success']) == 1 else cnt_success
                

            log = OrderedDict()
            log[agent_file] = OrderedDict()
            log[agent_file]['avg_success'] = cnt_success/run
            log[agent_file]['avg_reached'] = cnt_reached/run 
            log[agent_file]['avg_picked'] = cnt_picked/run
            wandb.log(log)

        with open(f"{results_dir_path}/results.json", "w") as f:
            json.dump(final_results,f, indent=4)