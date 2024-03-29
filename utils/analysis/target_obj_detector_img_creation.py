import os
import pickle
import cv2
import re
import glob
from PIL import Image
import numpy as np
# img=cv2.imread('/home/ciccio/Pictures/conf_1_v3.png')
# cv2.imshow('Window',img)
# cv2.destroyAllWindows()
import torch
from torchvision.transforms import Normalize
import json

def find_number(name):
    #return int(re.search(r"\d+", name).group())
    #regex = r'(\d+)_(\d+)'
    regex = r'(\d+)'
    res = re.search(regex, name)
    return res.group()

# Define a custom sorting key function
def sort_key(file_name):
    # Extract the number X from the file name using a regular expression
    pkl_name = file_name.split('/')[-1].split('.')[0]
    match = find_number(pkl_name)
    if match:
        return match
    else:
        return 0  # Return 0 if the file name doesn't contain a number

def torch_to_numpy(tensor):
    tensor = Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])(tensor)
    tensor = torch.mul(tensor, 255)
    # convert the tensor to a numpy array
    numpy_array = tensor.cpu().numpy()
    # transpose the numpy array to [y,h,w,c]
    numpy_array_transposed = np.transpose(numpy_array, (1, 3, 4, 2, 0))[:,:,:,:,0]
    return numpy_array_transposed

def create_img(base_path="/", task_name="pick_place"):
    
    results_folder = f"results_{task_name}"
    step_pattern = os.path.join(base_path, results_folder, "task-*")
    correct_prediction = 0
    cnt = 0
    for step_path in glob.glob(step_pattern):    
        step = step_path.split("-")[-1]
        print(f"---- Step {step} ----")
        context_files = glob.glob(os.path.join(step_path, "context*.pkl"))
        context_files.sort(key=sort_key)
        traj_files = glob.glob(os.path.join(step_path, "traj*.pkl"))
        traj_files.sort(key=sort_key)
        
        try:
            print("Creating folder {}".format(os.path.join(step_path, "image")))
            image_path = os.path.join(step_path, "image")
            os.makedirs(image_path)
        except:
            pass

        for context_file, traj_file in zip(context_files, traj_files):
            cnt += 1 
            print(context_file, traj_file)
            with open(context_file, "rb") as f:
                context_data = pickle.load(f)
            with open(traj_file, "rb") as f:
                traj_data = pickle.load(f)
            # open json file
            traj_result=None
            try:
                json_file = traj_file.split('.')[-2]
                with open(f"{json_file}.json", "rb") as f:
                    traj_result = json.load(f)
            except:
                pass

            # convert context from torch tensor to numpy
            context_frames = torch_to_numpy(context_data)

            traj_frame = traj_data
            number_of_context_frames = len(context_frames)
            demo_height, demo_width, _ = context_frames[0].shape
            traj_height, traj_width, _ = traj_frame.shape

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
            new_image = np.array(cv2.resize(cv2.vconcat(frames), (traj_width, traj_height)), np.uint8)
            output_width = 2*traj_width 
            output_height = traj_height
            context_number = find_number(context_file.split('/')[-1].split('.')[0])
            trj_number = find_number(traj_file.split('/')[-1].split('.')[0])
            pred_prob = [round(prob, 3) for prob in traj_result['pred_prob'][0][0]]
            try:
                os.makedirs(os.path.join(step_path, "images"))
            except:
                pass
            out_path = os.path.join(step_path, "images", f"{task_name}_step_{step}_demo_{context_number}_traj_{trj_number}.png")
            # create the string to put on each frame
            if traj_result:
                res_string_1 = f"Predicted target slot {traj_result['predicted_slot']} - GT slot {traj_result['gt_slot']}"
                res_string_2 = f"Pred probabilities {pred_prob}"
            else:
                res_string = f"Sample index {step}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35
            thickness = 1
            output_frame = cv2.hconcat([new_image, traj_frame[:,:,::-1]])
            cv2.putText(output_frame, res_string_1, (0, 80), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
            cv2.putText(output_frame, res_string_2, (0, 99), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
            cv2.imwrite(out_path, output_frame)

            if traj_result['pred_correctness']:
                correct_prediction += 1
        
        print(f"Correct predictions {correct_prediction} - Number sample {cnt} - {correct_prediction/cnt}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default="/", help="Path to checkpoint folder")
    parser.add_argument('--task', type=str, default="pick_place", help="Task name")        
    args = parser.parse_args()
    
    # 1. create video
    create_img(base_path=args.base_path, task_name=args.task)
    
