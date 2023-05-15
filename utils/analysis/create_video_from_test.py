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
    # return int(re.search(r"\d+", name).group())
    # regex = r'(\d+)_(\d+)'
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
    tensor = Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225],
                       std=[1/0.229, 1/0.224, 1/0.225])(tensor)
    tensor = torch.mul(tensor, 255)
    # convert the tensor to a numpy array
    numpy_array = tensor.cpu().numpy()
    # transpose the numpy array to [y,h,w,c]
    numpy_array_transposed = np.transpose(
        numpy_array, (1, 3, 4, 2, 0))[:, :, :, :, 0]
    return numpy_array_transposed


def create_video_for_each_trj(base_path="/", task_name="pick_place"):

    results_folder = f"results_{task_name}"
    step_pattern = os.path.join(base_path, results_folder, "step-*")
    for step_path in glob.glob(step_pattern):

        step = step_path.split("-")[-1]
        print(f"---- Step {step} ----")
        context_files = glob.glob(os.path.join(step_path, "context*.pkl"))
        context_files.sort(key=sort_key)
        traj_files = glob.glob(os.path.join(step_path, "traj*.pkl"))
        traj_files.sort(key=sort_key)

        try:
            print("Creating folder {}".format(
                os.path.join(step_path, "video")))
            video_path = os.path.join(step_path, "video")
            os.makedirs(video_path)
        except:
            pass

        for context_file, traj_file in zip(context_files, traj_files):
            print(context_file, traj_file)
            with open(context_file, "rb") as f:
                context_data = pickle.load(f)
            with open(traj_file, "rb") as f:
                traj_data = pickle.load(f)
            # open json file
            traj_result = None
            try:
                json_file = traj_file.split('.')[-2]
                with open(f"{json_file}.json", "rb") as f:
                    traj_result = json.load(f)
            except:
                pass

            # convert context from torch tensor to numpy
            context_frames = torch_to_numpy(context_data)
            traj_frames = [t["obs"]['camera_front_image'] for t in traj_data]

            # get predicted slot
            predicted_slot = []
            if 'info' in traj_data.get(0):
                for i in range(1, len(traj_data)):
                    predicted_slot.append(
                        np.argmax(traj_data.get(i)['info']['target_pred']))

            number_of_context_frames = len(context_frames)
            demo_height, demo_width, _ = context_frames[0].shape
            traj_height, traj_width, _ = traj_frames[0].shape

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
                frames), (traj_width, traj_height)), np.uint8)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_width = 2*traj_width
            output_height = traj_height
            context_number = find_number(
                context_file.split('/')[-1].split('.')[0])
            trj_number = find_number(traj_file.split('/')[-1].split('.')[0])
            out_path = f"{task_name}_step_{step}_demo_{context_number}_traj_{trj_number}.mp4"
            out = cv2.VideoWriter(os.path.join(
                video_path, out_path), fourcc, 30, (output_width, output_height))

            # create the string to put on each frame
            if traj_result:
                res_string = f"Step {step} - Task {traj_result['variation_id']} - Reached {traj_result['reached']} - Picked {traj_result['picked']} - Success {traj_result['success']}"
            else:
                res_string = f"Sample index {step}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35
            thickness = 1
            for i, traj_frame in enumerate(traj_frames):
                output_frame = cv2.hconcat([new_image, traj_frame[:, :, ::-1]])
                if i != len(traj_frames)-1 and len(predicted_slot) != 0:
                    cv2.putText(output_frame,  res_string, (0, 80), font,
                                font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                    cv2.putText(output_frame,  f"Predicted slot {predicted_slot[i]}", (
                        0, 99), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                else:
                    cv2.putText(output_frame,  res_string, (0, 99), font,
                                font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

                out.write(output_frame)

            out.release()


if __name__ == '__main__':
    import argparse
    import debugpy
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default="/",
                        help="Path to checkpoint folder")
    parser.add_argument('--task', type=str,
                        default="pick_place", help="Task name")
    args = parser.parse_args()
    # debugpy.listen(('0.0.0.0', 5678))
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()
    # 1. create video
    create_video_for_each_trj(base_path=args.base_path, task_name=args.task)
