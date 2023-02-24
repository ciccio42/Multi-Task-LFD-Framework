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

start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

def load_trajectories(conf_file):
    conf_file.dataset_cfg.mode='val'
    return hydra.utils.instantiate(conf_file.dataset_cfg)

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


def get_obs_state(agent_data, img_formatter):
    print("Creating observation and state")
    states, images = [], []

    # 1. Get trajectory
    traj = agent_data['traj']
    traj_len = agent_data['len']

    # 2. Get state and image at time 1
    states.append(np.concatenate((traj[1]['obs']['ee_aa'], traj[1]['obs']['gripper_qpos'])).astype(np.float32))
    images.append(img_formatter(traj[1]['obs']['image']))

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
    states = np.concatenate((traj[reaching_indx]['obs']['ee_aa'],traj[reaching_indx]['obs']['gripper_qpos'])).astype(np.float32)[None]
    images = img_formatter(traj[reaching_indx]['obs']['image'])[None]
    
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


    
def run_inference(model, dataset, average, config):
    model.cuda(0)
    model.eval()
    with torch.no_grad():
        embedding_val = dict()
        print(f"Running {len(dataset)} inference steps")
        number_of_sample = len(dataset)-1
        indx = [i for i in range(0, 900, 100)]
        print(indx)
        for i in tqdm.tqdm(indx):
            print(dataset.all_file_pairs[i])
            with open(dataset.all_file_pairs[i][3], 'rb') as f:
                agent_data = pickle.load(f)
                sample = dataset[i]
                # check if the task_name key is already present
                if sample['task_name'] not in embedding_val:
                    embedding_val[sample['task_name']] = {}

                # check if the task_id key is already present
                print(f"{sample['task_name']} - {sample['task_id']}")
                if sample['task_id'] not in embedding_val[sample['task_name']]:
                    embedding_val[sample['task_name']][sample['task_id']] = []
                embedding = []

                context = torch.unsqueeze(sample['demo_data']['demo'], 0)
                
                # get the current observation, and state
                img_formatter = build_tvf_formatter(config=config)
                obs, state = get_obs_state(agent_data=agent_data, img_formatter=img_formatter)
                state = torch.from_numpy(state)
                obs = torch.from_numpy(obs)
                state = state[None, :]
                obs = obs[None, :]
                context, obs, state = context.cuda(0), obs.cuda(0), state.cuda(0)
                start.record()
                out = model(states=state, images=obs, context=context, eval=True)
                end.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = start.elapsed_time(end)
                # print(f"Inference time {curr_time/1000} s")
                # store the computed embedding for the current task
                if not average:
                    embedding_val[sample['task_name']][sample['task_id']].append(out['img_embed'].cpu())
                else:
                    embedding_val[sample['task_name']][sample['task_id']].append(torch.mean(out['img_embed'], 1, True).cpu())
                del context, obs, state, out, sample
                torch.cuda.empty_cache()

    return embedding_val

def compute_tsne(embedding):
    tsne = TSNE(n_components=2, learning_rate='auto', init='pca', )
    tsne_embedding = tsne.fit_transform(embedding)
    return tsne_embedding

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--step', type=int)
    parser.add_argument('--average',action='store_true')
    args = parser.parse_args()

    import debugpy
    debugpy.listen(('0.0.0.0', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()

    # 1. Load configuration file
    conf_file = OmegaConf.load(os.path.join(args.model, "config.yaml"))
    model = load_model(model_path=args.model, step=args.step, conf_file=conf_file)
    # 2. Get the validation dataset
    val_dataset = load_trajectories(conf_file=conf_file)
    # 3. Run inference over validation sample
    embedding_eval = run_inference(model=model, dataset=val_dataset, average=args.average, config=conf_file)
    # 4. Compute tsne
    # Create a matrix of embedding
    embedding_cat = None
    labels = []
    first = True
    cnt = 0
    for task in embedding_eval.keys():
        number_of_labels = len(embedding_eval[task].keys())
        for task_id in embedding_eval[task].keys():
            for i, embedding in enumerate(embedding_eval[task][task_id]):
                cnt += 1
                if first:
                    embedding_cat = torch.squeeze(embedding, dim=0)
                    first=False
                else:
                    embedding_cat = torch.cat((embedding_cat, torch.squeeze(embedding, dim=0)), dim=0)
                if not args.average:
                    for i in range(embedding.shape[1]):
                        labels.append(f"{task_id}")
                else:
                    labels.append(task_id)

    embedding_cat = embedding_cat.numpy()
    
    embedding_tsne = compute_tsne(embedding=embedding_cat)

    embedding_tsne_norm = np.sqrt(np.einsum("ij,ij->j", embedding_tsne.transpose(), embedding_tsne.transpose()))
    embedding_tsne_norm_indx = np.argsort(embedding_tsne_norm)
    print(embedding_tsne_norm_indx)

    # 5. Plot results
    sns.scatterplot(x=embedding_tsne[:, 0], y=embedding_tsne[:, 1], hue=labels,
                palette=sns.color_palette("hls", number_of_labels)).set(title="t-SNE")
    plt.show()
    plt.savefig(os.path.join(args.model, f"tsne_step_{args.step}.png"))