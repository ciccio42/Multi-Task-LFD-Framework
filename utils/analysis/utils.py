import warnings
import cv2
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
import os
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
from multi_task_il.datasets import Trajectory
from torch.multiprocessing import Pool, set_start_method
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms.functional import resized_crop
from robosuite import load_controller_config
from multi_task_robosuite_env.controllers.controllers.expert_nut_assembly import \
    get_expert_trajectory as nut_expert
from multi_task_robosuite_env.controllers.controllers.expert_pick_place import \
    get_expert_trajectory as place_expert
from multi_task_robosuite_env.controllers.controllers.expert_block_stacking import \
    get_expert_trajectory as stack_expert
import sys
import pickle as pkl
import json
import wandb
from natsort import natsorted
import robosuite.utils.transform_utils as T
set_start_method('forkserver', force=True)
sys.path.append('/home/Multi-Task-LFD-Framework/repo/mosaic/tasks/test_models')
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def torch_to_numpy(original_tensor):
    tensor = copy.deepcopy(original_tensor)
    tensor = Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225],
                       std=[1/0.229, 1/0.224, 1/0.225])(tensor)
    tensor = torch.mul(tensor, 255)
    # convert the tensor to a numpy array
    numpy_array = tensor.cpu().numpy()
    # transpose the numpy array to [y,h,w,c]
    numpy_array_transposed = np.transpose(
        numpy_array, (1, 3, 4, 2, 0))[:, :, :, :, 0]
    return numpy_array_transposed


def load_trajectories(conf_file, mode='train'):
    conf_file.dataset_cfg.mode = mode
    return hydra.utils.instantiate(conf_file.dataset_cfg)


def load_pkl_files(conf_file, task_name='pick_place', step="29348_1"):
    # Get saved folder
    save_path = conf_file.save_path
    exp_name = conf_file.exp_name
    results_task = f"results_{task_name}"
    step = f"step-{step}"
    pkls_path = os.path.join(save_path, exp_name, results_task, step)

    filelist = glob.glob(os.path.join(pkls_path, '*.pkl'))
    file_paths = OrderedDict()
    file_paths['context'] = []
    file_paths['agent'] = []

    for file in natsorted(filelist):
        if "context" in file:
            file_paths['context'].append(os.path.join(pkls_path, file))
        elif "traj" in file:
            file_paths['agent'].append(os.path.join(pkls_path, file))

    return file_paths


def load_model(model_path=None, step=0, conf_file=None):
    if model_path:
        # 1. Create the model starting from configuration
        model = hydra.utils.instantiate(conf_file.policy)
        # 2. Load weights
        weights = torch.load(os.path.join(
            model_path, f"model_save-{step}.pt"), map_location=torch.device('cpu'))
        model.load_state_dict(weights)
        return model
    else:
        raise ValueError("Model path cannot be None")


def startup_env(model, env, context, gpu_id, variation_id, baseline=None):
    done, states, images = False, [], []
    if baseline is None:
        states = deque(states, maxlen=1)
        images = deque(images, maxlen=1)  # NOTE: always use only one frame
    context = context.cuda(gpu_id).float()
    np.random.seed(None)
    while True:
        try:
            obs = env.reset()
            # make a "null step" to stabilize all objects
            current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
            current_gripper_orientation = T.quat2axisangle(T.mat2quat(np.reshape(
                env.sim.data.site_xmat[env.robots[0].eef_site_id], (3, 3))))
            current_gripper_pose = np.concatenate(
                (current_gripper_position, current_gripper_orientation, np.array([1])), axis=-1)
            obs, reward, env_done, info = env.step(current_gripper_pose)
            break
        except:
            pass
    traj = Trajectory()
    traj.append(obs)
    tasks = {'success': False, 'reached': False,
             'picked': False, 'variation_id': variation_id}
    return done, states, images, context, obs, traj, tasks


def init_env(env, traj, task_name):
    # get objects id
    if task_name == 'pick_place':
        for obj_name in env.object_to_id.keys():
            obj = env.objects[env.object_to_id[obj_name]]
            # set object position based on trajectory file
            obj_pos = traj[1]['obs'][f"{obj_name}_pos"]
            obj_quat = traj[1]['obs'][f"{obj_name}_quat"]
            env.sim.data.set_joint_qpos(
                obj.joints[0], np.concatenate([obj_pos, obj_quat]))
    elif task_name == 'nut_assembly':
        for obj_name in env.env.nut_to_id.keys():
            obj = env.env.nuts[env.env.nut_to_id[obj_name]]
            obj_id = env.env.nut_to_id[obj_name]
            if obj_id == 0:
                obj_pos = traj[1]['obs']['round-nut_pos']
                obj_quat = traj[1]['obs']['round-nut_quat']
            else:
                obj_pos = traj[1]['obs'][f'round-nut-{obj_id+1}_pos']
                obj_quat = traj[1]['obs'][f'round-nut-{obj_id+1}_quat']
            # set object position based on trajectory file
            env.sim.data.set_joint_qpos(
                obj.joints[0], np.concatenate([obj_pos, obj_quat]))


def get_action(model, states, images, context, gpu_id, n_steps, max_T=80, baseline=None):
    s_t = torch.from_numpy(np.concatenate(states, 0).astype(np.float32))[None]
    if isinstance(images[-1], np.ndarray):
        i_t = torch.from_numpy(np.concatenate(
            images, 0).astype(np.float32))[None]
    else:
        i_t = images[0][None]
    s_t, i_t = s_t.cuda(gpu_id), i_t.cuda(gpu_id).float()

    if baseline == 'maml':
        learner = model.clone()
        learner.adapt(
            learner(None, context[0], learned_loss=True)['learned_loss'], allow_nograd=True, allow_unused=True)
        out = learner(states=s_t[0], images=i_t[0], ret_dist=True)
        action = out['action_dist'].sample()[-1].cpu().detach().numpy()

    else:
        with torch.no_grad():
            out = model(states=s_t, images=i_t, context=context,
                        eval=True)  # to avoid computing ATC loss
            action = out['bc_distrib'].sample()[0, -1].cpu().numpy()
    # if TASK_NAME == 'nut_assembly':
    #     action[3:7] = [1.0, 1.0, 0.0, 0.0]
    action[-1] = 1 if action[-1] > 0 and n_steps < max_T - 1 else -1
    return action


def select_random_frames(frames, n_select, sample_sides=True, random_frames=True):
    selected_frames = []
    def clip(x): return int(max(0, min(x, len(frames) - 1)))
    per_bracket = max(len(frames) / n_select, 1)

    if random_frames:
        for i in range(n_select):
            n = clip(np.random.randint(
                int(i * per_bracket), int((i + 1) * per_bracket)))
            if sample_sides and i == n_select - 1:
                n = len(frames) - 1
            elif sample_sides and i == 0:
                n = 1
            selected_frames.append(n)
    else:
        for i in range(n_select):
            # get first frame
            if i == 0:
                n = 1
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
        return [frames[i]['obs']['camera_front_image'] for i in selected_frames]
        # return [frames[i]['obs']['image-state'] for i in selected_frames]
    return frames[selected_frames]


def build_tvf_formatter(config, env_name='stack'):
    """Use this for torchvision.transforms in multi-task dataset, 
    note eval_fn always feeds in traj['obs']['images'], i.e. shape (h,w,3)
    """
    dataset_cfg = config.train_cfg.dataset
    height, width = dataset_cfg.get(
        'height', 100), dataset_cfg.get('width', 180)
    task_spec = config.tasks_cfgs.get(env_name, dict())

    crop_params = task_spec.get('crop', [0, 0, 0, 0])
    # print(crop_params)
    top, left = crop_params[0], crop_params[2]

    def resize_crop(img):
        if len(img.shape) == 4:
            img = img[0]
        img_h, img_w = img.shape[0], img.shape[1]
        assert img_h != 3 and img_w != 3, img.shape
        box_h, box_w = img_h - top - \
            crop_params[1], img_w - left - crop_params[3]

        obs = ToTensor()(img.copy())
        obs = resized_crop(obs, top=top, left=left, height=box_h, width=box_w,
                           size=(height, width))

        # obs = Normalize(mean=[0.485, 0.456, 0.406],
        #                 std=[0.229, 0.224, 0.225])(obs)
        cv2.imwrite("random_resized_crop_test.png",
                    np.moveaxis(obs.numpy(), 0, -1)*255)
        return obs
    return resize_crop


def build_env_context(img_formatter, T_context=4, ctr=0, env_name='nut',
                      heights=100, widths=200, size=False, shape=False, color=False, gpu_id=0, variation=None, random_frames=True, controller_path=None):
    create_seed = random.Random(None)
    create_seed = create_seed.getrandbits(32)
    if controller_path == None:
        controller = load_controller_config(default_controller='IK_POSE')
    else:
        # load custom controller
        controller = load_controller_config(
            custom_fpath=controller_path)
    assert gpu_id != -1
    build_task = TASK_MAP.get(env_name, None)
    assert build_task, 'Got unsupported task '+env_name
    div = int(build_task['num_variations'])
    env_fn = build_task['env_fn']
    agent_name, teacher_name = build_task['agent-teacher']

    if variation == None:
        variation = ctr % div
    else:
        variation = variation

    if 'Stack' in teacher_name:
        teacher_expert_rollout = env_fn(teacher_name,
                                        controller_type=controller,
                                        task=variation,
                                        size=size,
                                        shape=shape,
                                        color=color,
                                        seed=create_seed,
                                        gpu_id=gpu_id,
                                        object_set=TASK_MAP[env_name]['object_set'])
        agent_env = env_fn(agent_name,
                           size=size,
                           shape=shape,
                           color=color,
                           controller_type=controller,
                           task=variation,
                           ret_env=True,
                           seed=create_seed,
                           gpu_id=gpu_id,
                           object_set=TASK_MAP[env_name]['object_set'])
    else:
        teacher_expert_rollout = env_fn(teacher_name,
                                        controller_type=controller,
                                        task=variation,
                                        seed=create_seed,
                                        gpu_id=gpu_id,
                                        object_set=TASK_MAP[env_name]['object_set'])

        agent_env = env_fn(agent_name,
                           controller_type=controller,
                           task=variation,
                           ret_env=True,
                           seed=create_seed,
                           gpu_id=gpu_id,
                           object_set=TASK_MAP[env_name]['object_set'])

    assert isinstance(teacher_expert_rollout, Trajectory)
    context = select_random_frames(
        teacher_expert_rollout, T_context, sample_sides=True, random_frames=random_frames)
    # convert BGR context image to RGB and scale to 0-1
    for i, img in enumerate(context):
        cv2.imwrite(f"context_{i}.png", np.array(img[:, :, ::-1]))
    context = [img_formatter(i[:, :, ::-1])[None] for i in context]
    # assert len(context ) == 6
    if isinstance(context[0], np.ndarray):
        context = torch.from_numpy(np.concatenate(context, 0))[None]
    else:
        context = torch.cat(context, dim=0)[None]

    return agent_env, context, variation, teacher_expert_rollout


def create_env(env_fn, agent_name, variation, ret_env, seed=None, heights=100, widths=180, gpu_id=0):
    if seed is None:
        create_seed = random.Random(None)
        create_seed = create_seed.getrandbits(32)
    else:
        create_seed = seed
    print(f"Creating environment with variation {variation}")
    controller = load_controller_config(default_controller='IK_POSE')
    return env_fn(agent_name, controller_type=controller, task=variation, ret_env=True, seed=create_seed, heights=heights, widths=widths, gpu_id=gpu_id)


def pick_place_eval(model, env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=None, seed=None, agent_traj=None, model_act=False, show_img=False, experiment_number=1):

    done, states, images, context, obs, traj, tasks = \
        startup_env(model, env, context, gpu_id, variation_id,
                    baseline=baseline, seed=seed)
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
    while not done:
        tasks['reached'] = tasks['reached'] or np.linalg.norm(
            obs[obj_delta_key][:2]) < 0.03
        tasks['picked'] = tasks['picked'] or (
            tasks['reached'] and obs[obj_key][2] - start_z > 0.05)
        if baseline and len(states) >= 5:
            states, images = [], []
        states.append(np.concatenate(
            (obs['ee_aa'], obs['gripper_qpos'])).astype(np.float32)[None])

        if True:
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
                        frame = context_frames[index]
                        row_frames.append(frame)
                row = cv2.hconcat(row_frames)
                frames.append(row)
            new_image = np.array(cv2.resize(cv2.vconcat(
                frames), (demo_width, demo_height)), np.uint8)
            output_frame = cv2.hconcat(
                [new_image, obs['image'][:, :, ::-1]])
            # showing the image
            cv2.imwrite(
                f'/home/frosa_loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/utils/test_img/frame_{t}.png', output_frame)
            t += 1
            # waiting using waitKey method
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()

        images.append(img_formatter(
            obs['image'][:, :, ::-1]/255)[None])
        if model_act:
            action = get_action(model, states, images, context,
                                gpu_id, n_steps, max_T, baseline)
        else:
            action = agent_traj[t]['action']

        obs, reward, env_done, info = env.step(action)
        traj.append(obs, reward, done, info, action)

        tasks['success'] = reward or tasks['success']
        n_steps += 1
        if env_done or reward or n_steps > max_T:
            done = True
    env.close()
    del env
    del states
    del images
    del model
    torch.cuda.empty_cache()
    return traj, tasks, context


def nut_assembly_eval(model, env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=None, seed=None, agent_traj=None, model_act=False, show_img=False):

    done, states, images, context, obs, traj, tasks = \
        startup_env(model, env, context, gpu_id, variation_id,
                    baseline=baseline, seed=seed)
    n_steps = 0

    if agent_traj is not None:
        # change object position
        print("Set object position based on training sample")
        init_env(env=env, traj=agent_traj)
    else:
        print("Set object position randomly")

    object_name = env.nuts[env.nut_id].name
    if env.nut_id == 0:
        handle_loc = env.sim.data.site_xpos[env.sim.model.site_name2id(
            'round-nut_handle_site')]
    elif env.nut_id == 1:
        handle_loc = env.sim.data.site_xpos[env.sim.model.site_name2id(
            'round-nut-2_handle_site')]
    else:
        handle_loc = env.sim.data.site_xpos[env.sim.model.site_name2id(
            'round-nut-3_handle_site')]

    obj_key = object_name + '_pos'
    start_z = obs[obj_key][2]
    n_steps = 0
    while not done:
        tasks['reached'] = tasks['reached'] or np.linalg.norm(
            handle_loc - obs['eef_pos']) < 0.045
        tasks['picked'] = tasks['picked'] or (
            tasks['reached'] and obs[obj_key][2] - start_z > 0.05)
        if baseline and len(states) >= 5:
            states, images = [], []

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
                        frame = context_frames[index][:, :, ::-1]
                        row_frames.append(frame)
                row = cv2.hconcat(row_frames)
                frames.append(row)
            new_image = np.array(cv2.resize(cv2.vconcat(
                frames), (demo_width, demo_height)), np.uint8)
            output_frame = cv2.hconcat([new_image, obs['image'][:, :, ::-1]])
            # showing the image
            cv2.imshow(f'Frame {t}', output_frame)
            t += 1
            # waiting using waitKey method
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

        states.append(np.concatenate(
            (obs['ee_aa'], obs['gripper_qpos'])).astype(np.float32)[None])
        images.append(img_formatter(obs['image'][:, :, ::-1]/255)[None])
        action = get_action(model, states, images, context,
                            gpu_id, n_steps, max_T, baseline)

        obs, reward, env_done, info = env.step(action)
        traj.append(obs, reward, done, info, action)
        tasks['success'] = (reward and tasks['reached']) or tasks['success']
        n_steps += 1
        if env_done or reward or n_steps > max_T:
            done = True
    env.close()
    del env
    del states
    del images
    del model
    torch.cuda.empty_cache()
    return traj, tasks, context


OBJECT_DISTRIBUTION = {
    'pick_place': {
        'greenbox': [0, 0, 0, 0],
        'yellowbox': [0, 0, 0, 0],
        'bluebox': [0, 0, 0, 0],
        'redbox': [0, 0, 0, 0],
        'ranges':  [[0.195, 0.255], [0.045, 0.105], [-0.105, -0.045], [-0.255, -0.195]]
    },
    'nut_assembly': {
        'nut0': [0, 0, 0],
        'nut1': [0, 0, 0],
        'nut2': [0, 0, 0],
        'ranges':  [[0.10, 0.31], [-0.10, 0.10], [-0.31, -0.10]]
    }
}

ENV_OBJECTS = {
    'pick_place': {
        'obj_names': ['greenbox', 'yellowbox', 'bluebox', 'redbox'],
        'ranges': [[-0.255, -0.195], [-0.105, -0.045], [0.045, 0.105], [0.195, 0.255]],
    },
    'nut_assembly': {
        'obj_names': ['nut0', 'nut1', 'nut2'],
        'ranges': [[0.10, 0.31], [-0.10, 0.10], [-0.31, -0.10]]
    }
}

TASK_MAP = {
    'nut_assembly':  {
        'num_variations':   9,
        'env_fn':   nut_expert,
        'eval_fn':  nut_assembly_eval,
        'agent-teacher': ('UR5e_NutAssemblyDistractor', 'Panda_NutAssemblyDistractor'),
        'render_hw': (200, 360),
        'object_set': 1,
    },
    'pick_place': {
        'num_variations':   16,
        'env_fn':   place_expert,
        'eval_fn':  pick_place_eval,
        'agent-teacher': ('UR5e_PickPlaceDistractor', 'Panda_PickPlaceDistractor'),
        'render_hw': (200, 360),  # (150, 270)
        'object_set': 2,
    },
}
