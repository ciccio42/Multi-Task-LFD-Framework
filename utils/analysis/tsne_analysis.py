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

def run_inference(model, dataset):
    model.cuda(0)
    model.eval()
    with torch.no_grad():
        embedding_val = dict()
        print(f"Running {len(dataset)} inference steps")
        number_of_sample = len(dataset)-1
        for i in tqdm.tqdm(range(number_of_sample)):
            sample = dataset[i]
            # check if the task_name key is already present
            if sample['task_name'] not in embedding_val:
                embedding_val[sample['task_name']] = {}
            
            # check if the task_id key is already present
            if sample['task_id'] not in embedding_val[sample['task_name']]:
                embedding_val[sample['task_name']][sample['task_id']] = []
            embedding = [] 

            context = torch.unsqueeze(sample['demo_data']['demo'], 0) 
            # get the current observation
            obs = torch.unsqueeze(sample['traj']['images'], 0)
            # get the current state
            state = torch.unsqueeze(torch.from_numpy(sample['traj']['states']), 0)
            context, obs, state = context.cuda(0), obs.cuda(0), state.cuda(0) 
            start.record()
            out = model(states=state, images=obs, context=context, eval=True)
            end.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = start.elapsed_time(end)
            #print(f"Inference time {curr_time/1000} s")
            # store the computed embedding for the current task
            embedding_val[sample['task_name']][sample['task_id']].append(out['img_embed'].cpu())
            del context, obs, state, out, sample
            torch.cuda.empty_cache()
    return embedding_val

def compute_tsne(embedding):
    tsne = TSNE(n_components=2, learning_rate='auto', init='pca')
    tsne_embedding = tsne.fit_transform(embedding)
    return tsne_embedding
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--step', type=int)
    args = parser.parse_args()
    
    # import debugpy
    # debugpy.listen(('0.0.0.0', 5678))
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()
    
    # 1. Load configuration file
    conf_file = OmegaConf.load(os.path.join(args.model, "config.yaml"))
    model = load_model(model_path=args.model, step=args.step, conf_file=conf_file)
    # 2. Get the validation dataset
    val_dataset = load_trajectories(conf_file=conf_file)
    # 3. Run inference over validation sample
    embedding_eval = run_inference(model=model, dataset=val_dataset)
    # 4. Compute tsne
    # Create a matrix of embedding
    embedding_cat = None
    labels = []
    first = True
    for task in embedding_eval.keys():
        number_of_labels = len(embedding_eval[task].keys())
        for task_id in embedding_eval[task].keys():
            for embedding in embedding_eval[task][task_id]:
                if first:
                    embedding_cat = torch.squeeze(embedding, dim=0)
                    first=False
                else:
                    embedding_cat = torch.cat((embedding_cat, torch.squeeze(embedding, dim=0)), dim=0)
                for i in range(embedding.shape[1]):
                    labels.append(task_id)
    embedding_cat = embedding_cat.numpy()
    embedding_tsne = compute_tsne(embedding=embedding_cat)
    # 5. Plot results
    sns.scatterplot(x=embedding_tsne[:, 0], y=embedding_tsne[:, 1], hue=labels,
                palette=sns.color_palette("hls", number_of_labels)).set(title="t-SNE")
    plt.show()