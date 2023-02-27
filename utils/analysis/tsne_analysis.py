import torch
from omegaconf import DictConfig, OmegaConf
import hydra, os
import numpy as np
import gc
import copy
import tqdm
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms.functional import resized_crop
import sys 
import glob
import json
import re

STEP_PATH="/user/frosa/robotic/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/baseline-1/1Task-Pick-Place-Stable-Policy-Batch32-1gpu-Attn2ly128-Act2ly256mix4-headCat-simclr128x512/results_pick_place/step-72900"
TASK_NAME=STEP_PATH.split('/')[-2].replace("results_","")

start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

def find_number(name):
    return int(re.search(r"\d+", name).group())

def torch_to_numpy(tensor):
    tensor = Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])(tensor)
    tensor = torch.mul(tensor, 255)
    # convert the tensor to a numpy array
    numpy_array = tensor.cpu().numpy()
    # transpose the numpy array to [y,h,w,c]
    numpy_array_transposed = np.transpose(numpy_array, (1, 3, 4, 2, 0))[:,:,:,:,0]
    return numpy_array_transposed

# Define a custom sorting key function
def sort_key(file_name):
    # Extract the number X from the file name using a regular expression
    pkl_name = file_name.split('/')[-1].split('.')[0]
    match = find_number(pkl_name)
    if match:
        return match
    else:
        return 0  # Return 0 if the file name doesn't contain a number

def load_trajectories(conf_file, test_folder):
    if not test_folder:
        conf_file.dataset_cfg.mode='val'
        return hydra.utils.instantiate(conf_file.dataset_cfg)
    else:
        step_path = STEP_PATH
        step = step_path.split("-")[-1]
        print(f"---- Step {step} ----")
        context_files = glob.glob(os.path.join(step_path, "context*.pkl"))
        context_files.sort(key=sort_key)
        traj_files = glob.glob(os.path.join(step_path, "traj*.pkl"))
        traj_files.sort(key=sort_key)
        return context_files, traj_files

def load_model(model_path=None, step=0, conf_file=None):
    if model_path:
        # 1. Create the model starting from configuration
        model = hydra.utils.instantiate(conf_file.policy)
        # 2. Load weights
        weights = torch.load(os.path.join(model_path, f"model_save-{step}.pt"),map_location=torch.device('cpu'))
        model.load_state_dict(weights)
        return model
    else:
        raise ValueError("Model path cannot be None")

def build_tvf_formatter(config, env_name='stack'):
    """Use this for torchvision.transforms in multi-task dataset, 
    note eval_fn always feeds in traj['obs']['images'], i.e. shape (h,w,3)
    """
    dataset_cfg = config.train_cfg.dataset
    height, width = dataset_cfg.get('height', 100), dataset_cfg.get('width', 180)
    task_spec = config.get(env_name, dict())
    # if 'baseline' in config.policy._target_: # yaml for the CMU baseline is messed up
    #     crop_params = [10, 50, 70, 70] if env_name == 'place' else [0,0,0,0]

    #assert task_spec, 'Must go back to the saved config file to get crop params for this task: '+env_name 
    crop_params = task_spec.get('crop', [0,0,0,0])
    #print(crop_params)
    top, left = crop_params[0], crop_params[2]
    def resize_crop(img):
        if len(img.shape) == 4:
            img = img[0]
        img_h, img_w = img.shape[0], img.shape[1]
        assert img_h != 3 and img_w != 3, img.shape 
        box_h, box_w = img_h - top - crop_params[1], img_w - left - crop_params[3]
        
        obs = ToTensor()(img.copy())
        obs = resized_crop(obs, top=top, left=left, height=box_h, width=box_w,
                        size=(height, width))
 
        obs = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(obs)
        
        return obs 
    return resize_crop 

def get_key_point_frames(agent_data, img_formatter):
    states, images = None, None

    # 1. Get trajectory
    traj = agent_data['traj']
    traj_len = agent_data['len']

    # 2. Get state and image at time 1
    states = np.concatenate((traj[1]['obs']['ee_aa'], traj[1]['obs']['gripper_qpos'])).astype(np.float32)[None]
    images = img_formatter(traj[1]['obs']['image'])[None]

    # 3. Get the indices of different states
    reaching_obj_time = []
    obj_in_hand_time = []
    move_time = []

    for i in range(traj_len):
        if 'reaching' in agent_data['traj'][i]['info']['status']:
            reaching_obj_time.append(i)
        elif 'hand' in agent_data['traj'][i]['info']['status']:
            obj_in_hand_time.append(i)
        elif 'moving' in agent_data['traj'][i]['info']['status']:
            move_time.append(i)

    # 4. Get states and frames for each state
    reaching_indx = reaching_obj_time[int(len(reaching_obj_time)/2)]
    states =  np.concatenate((states, np.concatenate((traj[reaching_indx]['obs']['ee_aa'],traj[reaching_indx]['obs']['gripper_qpos'])).astype(np.float32)[None]))
    images = np.concatenate((images, img_formatter(traj[reaching_indx]['obs']['image'])[None]))

    obj_in_hand_indx = obj_in_hand_time[int(len(obj_in_hand_time)/2)]
    states = np.concatenate((states, np.concatenate((traj[obj_in_hand_indx]['obs']['ee_aa'],traj[obj_in_hand_indx]['obs']['gripper_qpos']))[None]))
    images = np.concatenate((images, img_formatter(traj[obj_in_hand_indx]['obs']['image'])[None]))

    move_indx = move_time[int(len(move_time)/2)]
    states = np.concatenate((states,  np.concatenate((traj[move_indx]['obs']['ee_aa'],traj[move_indx]['obs']['gripper_qpos']))[None]))
    images = np.concatenate((images, img_formatter(traj[move_indx]['obs']['image'])[None]))

    # 5. Get the last frame and state
    states = np.concatenate((states,  np.concatenate((traj[-2]['obs']['ee_aa'],traj[-2]['obs']['gripper_qpos']))[None]))
    images = np.concatenate((images, img_formatter(traj[-2]['obs']['image'])[None]))

    return images, states

def get_full_traj(agent_data, img_formatter):

    states, images = None, None

    # 1. Get trajectory
    traj = agent_data['traj']
    traj_len = agent_data['len']
    for t in range(traj_len):
        # 2. Get state and image at time 1
        if t == 0:
            states = np.concatenate((traj[t]['obs']['ee_aa'], traj[t]['obs']['gripper_qpos'])).astype(np.float32)[None]
            images = img_formatter(traj[t]['obs']['image'])[[None]]
        else:
            states = np.concatenate((states,  np.concatenate((traj[t]['obs']['ee_aa'],traj[t]['obs']['gripper_qpos']))[None]))
            images = np.concatenate((images, img_formatter(traj[t]['obs']['image'])[None]))

    return images, states    

def get_full_traj_test(agent_data, img_formatter):
    states, images = None, None

    # 1. Get trajectory
    traj = agent_data
    traj_len = len(agent_data)
    for t in range(traj_len):
        # 2. Get state and image at time 1
        if t == 0:
            states = np.concatenate((traj[t]['obs']['ee_aa'], traj[t]['obs']['gripper_qpos'])).astype(np.float32)[None]
            images = img_formatter(traj[t]['obs']['image'])[[None]]
        else:
            states = np.concatenate((states,  np.concatenate((traj[t]['obs']['ee_aa'],traj[t]['obs']['gripper_qpos']))[None]))
            images = np.concatenate((images, img_formatter(traj[t]['obs']['image'])[None]))

    return images, states   


def get_obs_state(agent_data, img_formatter, get_obs_func):
    return get_obs_func(agent_data=agent_data, img_formatter=img_formatter)

def run_inference(model, dataset, average, config, validation_frames, full_traj, test_folder):
    model.cuda(0)
    model.eval()
    with torch.no_grad():
        embedding_val = dict()
        if not test_folder:
            print(f"Running {len(dataset)} inference steps")
            number_of_sample = len(dataset)-1
            indx = [i for i in range(number_of_sample)]
            for i in tqdm.tqdm(indx):
               
                if validation_frames:
                    sample = dataset[i]
                    # get the current observation
                    obs = sample['traj']['images']
                    # get the current state
                    state = torch.from_numpy(sample['traj']['states'])
                    
                else:
                    with open(dataset.all_file_pairs[i][3], 'rb') as f:
                        agent_data = pickle.load(f)
                        sample = dataset[i]
                        # get the current observation, and state
                        img_formatter = build_tvf_formatter(config=config)
                        if full_traj:
                            obs, state = get_obs_state(agent_data=agent_data, img_formatter=img_formatter, get_obs_func=get_full_traj)
                        else:
                            obs, state = get_obs_state(agent_data=agent_data, img_formatter=img_formatter, get_obs_func=get_key_point_frames)
                        state = torch.from_numpy(state)
                        obs = torch.from_numpy(obs)
                
                # check if the task_name key is already present
                if sample['task_name'] not in embedding_val:
                    embedding_val[sample['task_name']] = {}

                # check if the task_id key is already present
                if sample['task_id'] not in embedding_val[sample['task_name']]:
                    embedding_val[sample['task_name']][sample['task_id']] = dict()
                embedding = []

                context = torch.unsqueeze(sample['demo_data']['demo'], 0)
                embedding_val[sample['task_name']][sample['task_id']][i] = []
                context, obs, state = context.cuda(0), obs.cuda(0), state.cuda(0)
                
                start.record()
                for s_t, o_t in zip(state, obs):
                    s_t = s_t[None, None, :]
                    o_t = o_t[None, None, :]
                    out = model(states=s_t, images=o_t, context=context, eval=True)
                    end.record()
                    # WAIT FOR GPU SYNC
                    torch.cuda.synchronize()
                    curr_time = start.elapsed_time(end)
                    # print(f"Inference time {curr_time/1000} s")
                    # store the computed embedding for the current task
                    if not average:
                        embedding_val[sample['task_name']][sample['task_id']][i].append(out['img_embed'].cpu())
                    else:
                        embedding_val[sample['task_name']][sample['task_id']][i].append(torch.mean(out['img_embed'], 1, True).cpu())
                
                    del s_t, o_t
                    torch.cuda.empty_cache()

                del context, obs, state, out, sample
                torch.cuda.empty_cache()
        
        else:
            context_files, traj_files = dataset[0], dataset[1]
                
            # open file
            i = 0
            for context_file, traj_file in zip(context_files, traj_files):
                print(f"Inference {i}")
                with open(context_file, "rb") as f:
                    context_data = pickle.load(f)
                with open(traj_file, "rb") as f:
                    agent_data = pickle.load(f)
                # open json file
                json_file = traj_file.split('.')[-2]
                with open(f"{json_file}.json", "rb") as f:
                    traj_result = json.load(f)

                # get observation and state
                img_formatter = build_tvf_formatter(config=config)
                obs, state = get_obs_state(agent_data=agent_data, img_formatter=img_formatter, get_obs_func=get_full_traj_test)
                
                # check if the task_name key is already present
                if TASK_NAME not in embedding_val:
                    embedding_val[TASK_NAME] = {}

                task_id = traj_result['variation_id']    
                # check if the task_id key is already present
                if task_id not in embedding_val[TASK_NAME]:
                    embedding_val[TASK_NAME][task_id] = dict()
                embedding_val[TASK_NAME][task_id][i] = []
                
                context = context_data
                state = torch.from_numpy(state)
                obs = torch.from_numpy(obs)
                context, obs, state = context.cuda(0), obs.cuda(0), state.cuda(0)
                
                start.record()
                for s_t, o_t in zip(state, obs):
                    s_t = s_t[None, None, :]
                    o_t = o_t[None, None, :]
                    out = model(states=s_t, images=o_t, context=context, eval=True)
                    end.record()
                    # WAIT FOR GPU SYNC
                    torch.cuda.synchronize()
                    curr_time = start.elapsed_time(end)
                    # print(f"Inference time {curr_time/1000} s")
                    # store the computed embedding for the current task
                    if not average:
                        embedding_val[TASK_NAME][task_id][i].append(out['img_embed'].cpu())
                    else:
                        embedding_val[TASK_NAME][task_id][i].append(torch.mean(out['img_embed'], 1, True).cpu())
                
                    del s_t, o_t
                    torch.cuda.empty_cache()

                del context, obs, state, out
                torch.cuda.empty_cache()
                i += 1


    return embedding_val

def compute_tsne(embedding):
    tsne = TSNE(n_components=2, learning_rate='auto', init='pca', n_iter=10000)
    tsne_embedding = tsne.fit_transform(embedding)
    return tsne_embedding

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--step', type=int)
    parser.add_argument('--embedding_file', type=str)
    parser.add_argument('--average',action='store_true')
    parser.add_argument('--full_traj',action='store_true')
    parser.add_argument('--validation_frames',action='store_true')
    parser.add_argument('--debug',action='store_true')
    parser.add_argument('--test_folder',action='store_true')
    args = parser.parse_args()

    TEST_FOLDER=args.test_folder

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    # 1. Load configuration file
    conf_file = OmegaConf.load(os.path.join(args.model, "config.yaml"))
    # 2. Get the validation dataset
    dataset = load_trajectories(conf_file=conf_file, test_folder=TEST_FOLDER)

    if args.embedding_file is None:
        model = load_model(model_path=args.model, step=args.step, conf_file=conf_file)
        # 3. Run inference over validation sample
        embedding_eval = run_inference(model=model, dataset=dataset, average=args.average, config=conf_file, validation_frames=args.validation_frames, full_traj=args.full_traj, test_folder=TEST_FOLDER)

        with open(os.path.join(args.model, f"tsne_step_{args.step}_average_{args.average}_full_traj_{args.full_traj}_val_frames_{args.validation_frames}_test_folder_{TEST_FOLDER}"), 'wb') as f:
            pickle.dump(embedding_eval, f) 
    else:
        with open(args.embedding_file, 'rb') as f:
            embedding_eval = pickle.load(f)

    # 4. Compute tsne
    # Create a matrix of embedding
    embedding_cat = None
    labels = []
    first = True
    cnt = 0
    TASK_ID = [0]
    INDX = [0]
    for task in embedding_eval.keys():
        for task_id in embedding_eval[task].keys():
            if task_id in TASK_ID:
                for indx in embedding_eval[task][task_id].keys():
                    if indx in INDX:
                        if not TEST_FOLDER:
                            print(f"{dataset.all_file_pairs[indx]}")
                        for i, embedding in enumerate(embedding_eval[task][task_id][indx]):
                            cnt += 1
                            if first:
                                embedding_cat = torch.squeeze(embedding, dim=0)
                                first=False
                            else:
                                embedding_cat = torch.cat((embedding_cat, torch.squeeze(embedding, dim=0)), dim=0)
                            if not args.average:
                                if i % 10 == 0:
                                    labels.append(f"Task id {task_id} - indx {indx} - frame {i}")
                                else:
                                    labels.append(f"Task id {task_id} - indx {indx}")
                            else:
                                labels.append(f"{task_id}")

    number_of_labels = len(set(labels))
    embedding_cat = embedding_cat.numpy()
    
    embedding_tsne = compute_tsne(embedding=embedding_cat)

    embedding_tsne_norm = np.sqrt(np.einsum("ij,ij->j", embedding_tsne.transpose(), embedding_tsne.transpose()))
    embedding_tsne_norm_indx = np.argsort(embedding_tsne_norm)
    print(embedding_tsne_norm_indx)

    # 5. Plot results
    plt.figure(figsize=(10,8))
    sns.scatterplot(x=embedding_tsne[:, 0], y=embedding_tsne[:, 1], hue=labels,
                palette=sns.color_palette(palette='hls', n_colors=number_of_labels)).set(title="t-SNE")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(args.model, f"tsne_step_{args.step}_average_{args.average}_full_traj_{args.full_traj}_val_frames_{args.validation_frames}_test_folder_{TEST_FOLDER}.png"))