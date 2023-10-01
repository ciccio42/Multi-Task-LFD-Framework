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
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.functional import resized_crop
import multi_task_il


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
    # tensor = Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225],
    #                    std=[1/0.229, 1/0.224, 1/0.225])(tensor)
    tensor = torch.mul(tensor, 255)
    # convert the tensor to a numpy array
    numpy_array = tensor.cpu().numpy()
    # transpose the numpy array to [y,h,w,c]
    numpy_array_transposed = np.transpose(
        numpy_array, (1, 3, 4, 2, 0))[:, :, :, :, 0]
    return numpy_array_transposed


def adjust_bb(bb, crop_params=[20, 25, 80, 75]):

    x1_old, y1_old, x2_old, y2_old = bb
    x1_old = int(x1_old)
    y1_old = int(y1_old)
    x2_old = int(x2_old)
    y2_old = int(y2_old)

    top, left = crop_params[0], crop_params[2]
    img_height, img_width = 200, 360
    box_h, box_w = img_height - top - \
        crop_params[1], img_width - left - crop_params[3]

    # Modify bb based on computed resized-crop
    # 1. Take into account crop and resize
    x_scale = 180/box_w
    y_scale = 100/box_h
    x1 = int((x1_old/x_scale)+left)
    x2 = int((x2_old/x_scale)+left)
    y1 = int((y1_old/y_scale)+top)
    y2 = int((y2_old/y_scale)+top)
    return [x1, y1, x2, y2]


def create_video_for_each_trj(base_path="/", task_name="pick_place"):
    from omegaconf import DictConfig, OmegaConf

    results_folder = f"results_{task_name}"

    # Load config
    # config_path = os.path.join(base_path, "../../config.yaml")
    config_path = "/raid/home/frosa_Loc/checkpoint_save_folder/1Task-Pick-Place-100-180-BB-inference-Batch32/config.yaml"
    config = OmegaConf.load(config_path)

    # step_pattern = os.path.join(base_path, results_folder, "step-*")
    step_pattern = base_path
    for step_path in glob.glob(step_pattern):

        step = step_path.split("-")[-1]
        print(f"---- Step {step} ----")
        context_files = glob.glob(os.path.join(step_path, "context*.pkl"))
        context_files.sort(key=sort_key)
        traj_files = glob.glob(os.path.join(step_path, "traj*.pkl"))
        traj_files.sort(key=sort_key)
        print(context_files)
        try:
            print("Creating folder {}".format(
                os.path.join(step_path, "video")))
            video_path = os.path.join(step_path, "video")
            os.makedirs(video_path)
        except:
            pass
        if len(context_files) != 0:
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
                try:
                    context_frames = torch_to_numpy(context_data)
                except:
                    print("Exception")
                    context_frames = list()
                    for i in range(len(context_data)):
                        context_frames.append(context_data[i][:, :, ::-1])

                traj_frames = []
                bb_frames = []
                gt_bb = []
                predicted_conf_score = []
                iou = []
                predicted_bb = False
                for t, step in enumerate(traj_data):
                    if 'camera_front_image' in traj_data.get(t)["obs"].keys():
                        traj_frames.append(step["obs"]['camera_front_image'])
                    else:
                        traj_frames.append(step["obs"]['image'])

                    if 'predicted_bb' in traj_data.get(t)["obs"].keys():
                        predicted_bb = True
                        if isinstance(step["obs"]['predicted_bb'], np.ndarray):
                            bb_frames.append(
                                step["obs"]['predicted_bb'].tolist())
                        else:
                            bb_frames.append(step["obs"]['predicted_bb'])
                        predicted_conf_score.append(
                            step["obs"]['predicted_score'])
                        iou.append(step["obs"].get('iou', 0))
                        if isinstance(step["obs"]['predicted_bb'], np.ndarray):
                            gt_bb.append(
                                step["obs"]['predicted_bb'].tolist())
                        else:
                            gt_bb.append(step["obs"]['predicted_bb'])

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
                num_rows = (number_of_context_frames +
                            num_cols - 1) // num_cols

                # Create the grid of frames
                frames = []
                for i in range(num_rows):
                    row_frames = []
                    for j in range(num_cols):
                        index = i * num_cols + j
                        if index < number_of_context_frames:
                            frame = context_frames[index]
                            # cv2.imwrite(f"context_{index}", context_frames[index])
                            row_frames.append(frame)
                    row = cv2.hconcat(row_frames)
                    frames.append(row)

                new_image = np.array(cv2.resize(cv2.vconcat(
                    frames), (traj_width, traj_height)), np.uint8)

                context_number = find_number(
                    context_file.split('/')[-1].split('.')[0])
                trj_number = find_number(
                    traj_file.split('/')[-1].split('.')[0])
                out = None
                if len(traj_data) > 2:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    output_width = 2*traj_width
                    output_height = traj_height
                    # out_path = f"{task_name}_step_{step}_demo_{context_number}_traj_{trj_number}.mp4"
                    out_path = f"demo_{context_number}_traj_{trj_number}.mp4"
                    print(video_path)
                    print(out_path)
                    out = cv2.VideoWriter(os.path.join(
                        video_path, out_path), fourcc, 30, (output_width, output_height))
                else:
                    out_path = os.path.join(
                        video_path, f"demo_{context_number}_traj_{trj_number}.png")

                # create the string to put on each frame
                if traj_result:
                    # res_string = f"Step {step} - Task {traj_result['variation_id']} - Reached {traj_result['reached']} - Picked {traj_result['picked']} - Success {traj_result['success']}"
                    # res_string = f"Reached {traj_result['reached']} - Picked {traj_result['picked']} - Success {traj_result['success']}"
                    # res_string = ""
                    if predicted_bb:
                        res_string = f"Step {step} - Task {traj_result['variation_id']}"
                else:
                    res_string = f"Sample index {step}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.35
                thickness = 1
                for i, traj_frame in enumerate(traj_frames):
                    if len(bb_frames) != 0 and i != 0 and len(bb_frames) >= i+1:
                        if len(bb_frames[i-1]) == 4:
                            bb = adjust_bb(bb_frames[i-1])
                            gt_bb_t = adjust_bb(gt_bb[i-1])
                        else:
                            bb = adjust_bb(bb_frames[i-1][0])
                            gt_bb_t = adjust_bb(gt_bb[i-1][0])
                        traj_frame = np.array(cv2.rectangle(
                            traj_frame,
                            (int(bb[0]),
                             int(bb[1])),
                            (int(bb[2]),
                                int(bb[3])),
                            (0, 0, 255), 1))
                        traj_frame = np.array(cv2.rectangle(
                            traj_frame,
                            (int(gt_bb_t[0]),
                             int(gt_bb_t[1])),
                            (int(gt_bb_t[2]),
                                int(gt_bb_t[3])),
                            (0, 255, 0), 1))

                        if i != len(traj_frames)-1 and len(predicted_slot) != 0:
                            cv2.putText(traj_frame,  res_string, (0, 80), font,
                                        font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                            cv2.putText(traj_frame,  f"Predicted slot {predicted_slot[i]}", (
                                0, 99), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                        elif predicted_bb:
                            cv2.putText(traj_frame,
                                        f"Conf-Score {round(float(predicted_conf_score[i-1]), 2)} - IoU {round(float(iou[i-1]), 2)}", (
                                            0, 99),
                                        font,
                                        font_scale,
                                        (0, 0, 255),
                                        thickness,
                                        cv2.LINE_AA)
                        else:
                            cv2.putText(output_frame,  res_string, (0, 99), font,
                                        font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

                    output_frame = cv2.hconcat(
                        [new_image, traj_frame[:, :, ::-1]])

                    # cv2.imwrite("frame.png", output_frame)
                    if out is not None:
                        out.write(output_frame)
                    else:
                        cv2.imwrite(out_path, output_frame)
                if out is not None:
                    out.release()
        else:
            for traj_file in traj_files:
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
                if traj_result["success"] == 1:
                    if 'camera_front_image' in traj_data.get(0)["obs"].keys():
                        traj_frames = [t["obs"]['camera_front_image']
                                       for t in traj_data]
                    else:
                        traj_frames = [t["obs"]['image']
                                       for t in traj_data]

                    traj_height, traj_width, _ = traj_frames[0].shape
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    output_width = traj_width
                    output_height = traj_height
                    trj_number = find_number(
                        traj_file.split('/')[-1].split('.')[0])
                    out_path = f"traj_{trj_number}.mp4"
                    print(video_path)
                    print(out_path)
                    out = cv2.VideoWriter(os.path.join(
                        video_path, out_path), fourcc, 30, (output_width, output_height))

                    # create the string to put on each frame
                    if traj_result:
                        # res_string = f"Step {step} - Task {traj_result['variation_id']} - Reached {traj_result['reached']} - Picked {traj_result['picked']} - Success {traj_result['success']}"
                        res_string = f"Reached {traj_result['reached']} - Picked {traj_result['picked']} - Success {traj_result['success']}"
                    else:
                        res_string = f"Sample index {step}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.35
                    thickness = 1
                    predicted_slot = []
                    for i, traj_frame in enumerate(traj_frames):

                        output_frame = np.array(traj_frame[:, :, ::-1])

                        cv2.imwrite("frame.png", output_frame)
                        out.write(output_frame)

                    out.release()


def read_results(base_path="/", task_name="pick_place"):
    results_folder = f"results_{task_name}"
    # step_pattern = os.path.join(base_path, results_folder, "step-*")
    step_pattern = base_path
    avg_iou = 0
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

        success_cnt = 0
        reached_cnt = 0
        picked_cnt = 0
        file_cnt = 0
        mean_iou = 0.0
        tp_avg = 0.0
        fp_avg = 0.0
        number_frames = 0.0
        tp = 0.0
        fp = 0.0
        fp_pre_picking = 0.0
        fp_post_picking = 0.0
        fn = 0.0
        fn_pre_picking = 0.0
        fn_post_picking = 0.0
        OPEN_GRIPPER = np.array([-0.02, -0.25, -0.2, -0.02, -0.25, - 0.2])
        for context_file, traj_file in zip(context_files, traj_files):
            print(context_file, traj_file)
            with open(context_file, "rb") as f:
                context_data = pickle.load(f)
            with open(traj_file, "rb") as f:
                traj_data = pickle.load(f)

            try:
                for t in range(len(traj_data)):
                    if t != 0:
                        iou = traj_data.get(t)['obs']['iou']
                        number_frames += 1
                        if iou > 0.10:
                            tp += 1
                        else:
                            if traj_data.get(
                                    t)['obs'].get('predicted_bb') is not None:
                                fp += 1
                                # check for gripper open or close
                                gripper_state = traj_data.get(
                                    t)['obs'].get('gripper_qpos')
                                if np.array_equal(OPEN_GRIPPER, np.around(gripper_state, 2)):
                                    fp_pre_picking += 1
                                else:
                                    fp_post_picking += 1

                                bb = adjust_bb(traj_data.get(
                                    t)['obs']['predicted_bb'][0])
                                gt_bb_t = adjust_bb(
                                    traj_data.get(t)['obs']['gt_bb'][0])
                                traj_frame = np.array(traj_data.get(
                                    t)['obs']['camera_front_image'][:, :, ::-1])
                                traj_frame = np.array(cv2.rectangle(
                                    traj_frame,
                                    (int(bb[0]),
                                     int(bb[1])),
                                    (int(bb[2]),
                                        int(bb[3])),
                                    (0, 0, 255), 1))
                                traj_frame = np.array(cv2.rectangle(
                                    traj_frame,
                                    (int(gt_bb_t[0]),
                                     int(gt_bb_t[1])),
                                    (int(gt_bb_t[2]),
                                        int(gt_bb_t[3])),
                                    (0, 255, 0), 1))
                                cv2.imwrite(
                                    f"debug/{traj_file.split('/')[-1].split('.')[0]}_{t}.png", traj_frame)
                            else:
                                fn += 1
                                # check for gripper open or close
                                gripper_state = traj_data.get(
                                    t)['obs'].get('gripper_qpos')
                                if np.array_equal(OPEN_GRIPPER, np.around(gripper_state, 2)):
                                    fn_pre_picking += 1
                                else:
                                    fn_post_picking += 1
                                traj_frame = np.array(traj_data.get(
                                    t)['obs']['camera_front_image'][:, :, ::-1])
                                cv2.imwrite(
                                    f"debug/{traj_file.split('/')[-1].split('.')[0]}_{t}.png", traj_frame)
            except:
                pass
            # open json file
            traj_result = None
            try:
                json_file = traj_file.split('.')[-2]
                with open(f"{json_file}.json", "rb") as f:
                    traj_result = json.load(f)
                    file_cnt += 1
            except:
                pass

            if traj_result['success'] == 1:
                success_cnt += 1
            if traj_result['reached'] == 1:
                reached_cnt += 1
            if traj_result['picked'] == 1:
                picked_cnt += 1

            if 'avg_iou' in traj_result.keys():
                mean_iou += traj_result['avg_iou']
                tp_avg += traj_result['avg_tp']

                if traj_result['avg_iou'] < 0.10:
                    fp_avg += traj_result['avg_fp']
                else:
                    tp_avg += traj_result['avg_tp']

            try:
                avg_iou += traj_result['avg_iou']
            except:
                pass

        print(f"Success rate {success_cnt/file_cnt}")
        print(f"Reached rate {reached_cnt/file_cnt}")
        print(f"Picked rate {picked_cnt/file_cnt}")
        print(f"MeanIoU {mean_iou/file_cnt}")
        print(f"MeanFP {fp_avg/file_cnt}")
        print(f"Total Number frames {number_frames}")
        assert number_frames == (
            tp+fp+fn), "Number of frames must be equal to tp+fp+fn"
        print(f"Total Number tp {tp}")
        print(f"Total Number fp {fp}")
        print(f"Total Number fp-pre-picking {fp_pre_picking}")
        print(f"Total Number fp-post-picking {fp_post_picking}")
        print(f"Total Number fn {fn}")
        print(f"Total Number fn-pre-picking {fn_pre_picking}")
        print(f"Total Number fn-post-picking {fn_post_picking}")


if __name__ == '__main__':
    import argparse
    import debugpy
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default="/",
                        help="Path to checkpoint folder")
    parser.add_argument('--task', type=str,
                        default="pick_place", help="Task name")
    parser.add_argument('--metric', type=str,
                        default="results")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.debug:
        debugpy.listen(('0.0.0.0', 5679))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    if args.metric != "results":
        # 1. create video
        create_video_for_each_trj(
            base_path=args.base_path, task_name=args.task)
    else:
        import time
        read_results(base_path=args.base_path, task_name=args.task)
        time.sleep(3)
