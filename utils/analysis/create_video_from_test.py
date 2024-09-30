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
from torchvision.ops import box_iou
from utils import *
from torchvision.transforms import ToTensor

STATISTICS_CNTRS = {'reach_correct_obj': 0,
                    'reach_wrong_obj': 0,
                    'pick_correct_obj': 0,
                    'pick_wrong_obj': 0,
                    'pick_correct_obj_correct_place': 0,
                    'pick_correct_obj_wrong_place': 0,
                    'pick_wrong_obj_correct_place': 0,
                    'pick_wrong_obj_wrong_place': 0,
                    }


def pre_process(obs, crop_params, height, width):
    top, left = crop_params[0], crop_params[2]
    img_height, img_width = obs.shape[0], obs.shape[1]
    box_h, box_w = img_height - top - \
        crop_params[1], img_width - left - crop_params[3]

    # ---- Resized crop ----#
    obs = ToTensor()(obs)
    obs = resized_crop(obs, top=top, left=left, height=box_h,
                        width=box_w, size=(height, width))
    
    obs_np = obs.cpu().detach().numpy()
    obs_np = 255* np.transpose(obs_np, (1, 2, 0))
    return obs_np.astype(np.uint8)



def create_video_for_each_trj(base_path="/", task_name="pick_place"):
    from omegaconf import DictConfig, OmegaConf

    results_folder = f"results_{task_name}"

    # Load config
    if "gt_bb" in base_path:
        config_path = os.path.join(base_path, "../../../../config.yaml")
    else:
        config_path = os.path.join(base_path, "config.yaml") #../../

    # config_path = "/user/frosa/multi_task_lfd/checkpoint_save_folder/2Task-Pick-Place-Nut-Assembly-Mosaic-100-180-Target-Obj-Detector-BB-Batch50/config.yaml"
    config = OmegaConf.load(config_path)

    # step_pattern = os.path.join(base_path, results_folder, "step-*")
    step_pattern = base_path
    adjust = True #False if ("Real" in base_path) or ("REAL" in base_path) else True
    flip_channels = False if ("Real" in base_path) or (
        "REAL" in base_path) else True
    
    task_paths = glob.glob(os.path.join(args.base_path, 'task_*'))
        
    
    for step_path in task_paths:

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

                # open json file
                try:
                    json_file = traj_file.split('.')[-2]
                    with open(f"{json_file}.json", "rb") as f:
                        traj_result = json.load(f)
                except:
                    pass

                if traj_result.get('success', 0) == 1:
                    STATISTICS_CNTRS['pick_correct_obj_correct_place'] += 1
                if traj_result.get('reached', 0) == 1:
                    STATISTICS_CNTRS['reach_correct_obj'] += 1
                if traj_result.get('picked', 0) == 1:
                    STATISTICS_CNTRS['pick_correct_obj'] += 1

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
                activation_map = []
                predicted_conf_score = []
                iou = []
                predicted_bb = False
                for t, step in enumerate(traj_data):

                    # if reach(obs=step["obs"], task_name=task_name):
                    #     pass
                    # if pick(obs=step["obs"], task_name=task_name):
                    #     pass
                    # if place(obs=step["obs"], task_name=task_name):
                    #     pass
                    # if 'camera_front_image' in traj_data.get(t)["obs"].keys():
                    #     traj_frames.append(step["obs"]['camera_front_image'])
                    if "REAL" not in base_path and "Real" not in base_path:
                        if step["obs"].get('image', None) is None:
                            # traj_frames.append(pre_process(
                            #     obs=step["obs"]['camera_front_image'],
                            #     crop_params=config['tasks'][0]['crop'],
                            #     height=100,
                            #     width=180))
                            if step["obs"].get('camera_front_image_full_size', None) is not None:
                                traj_frames.append(cv2.imdecode(
                                    step["obs"]['camera_front_image_full_size'],
                                    cv2.IMREAD_COLOR))    
                            else:
                                traj_frames.append(step["obs"]['camera_front_image'])
                        else:
                            traj_frames.append(step["obs"]['image'])
                    else:
                        traj_frames.append(step["obs"]['camera_front_image'])

                    if 'predicted_bb' in traj_data.get(t)["obs"].keys():
                        predicted_bb = True
                        if isinstance(step["obs"]['predicted_bb'], np.ndarray):
                            bb_frames.append(
                                step["obs"]['predicted_bb'].tolist())
                        else:
                            bb_frames.append(step["obs"]['predicted_bb']['camera_front'])
                            
                        predicted_conf_score.append(
                            step["obs"].get('predicted_score', -1))

                        if 'gt_bb' in step["obs"].keys():
                            if isinstance(step["obs"]['gt_bb'], np.ndarray):
                                gt_bb.append(
                                    step["obs"]['gt_bb'].tolist())
                            else:
                                gt_bb.append(step["obs"]['gt_bb'])
                        # else:  
                        #     bbs = step["obs"]['obj_bb']['camera_front']
                            
                        # check if iou is saved
                        # if step["obs"].get('iou') is None:
                        #     if np.array(gt_bb[-1]).shape[0] == 1:
                        #         # compute iou
                        #         iou.append(
                        #             box_iou(boxes1=torch.from_numpy(np.array(bb_frames[-1])[None]), boxes2=torch.from_numpy(np.array(gt_bb[-1]))))
                        #     else:
                        #         # compute iou
                        #         iou.append(
                        #             box_iou(boxes1=torch.from_numpy(np.array(bb_frames[-1])[None]), boxes2=torch.from_numpy(np.array(gt_bb[-1])[None])))
                        # else:
                        #     iou.append(step["obs"]['iou'])

                    if 'activation_map' in traj_data.get(t)["obs"].keys():
                        activation_map.append(
                            traj_data.get(t)["obs"]['activation_map'])
                # get predicted slot
                # predicted_slot = []
                # if 'info' in traj_data.get(0):
                #     for i in range(1, len(traj_data)):
                #         predicted_slot.append(
                #             np.argmax(traj_data.get(i)['info']['target_pred']))

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
                if len(traj_data) >= 3:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    output_width = 2*traj_width
                    output_height = traj_height
                    # out_path = f"{task_name}_step_{step}_demo_{context_number}_traj_{trj_number}.mp4"
                    out_path = f"demo_{context_number}_traj_{trj_number}.mp4"
                    print(video_path)
                    print(out_path)
                    if "Real" in traj_file or "REAL" in traj_file:
                        frame_rate = 10
                    else:
                        frame_rate = 30
                    out = cv2.VideoWriter(os.path.join(
                        video_path, out_path), fourcc, frame_rate, (output_width, output_height))
                    if len(activation_map) != 0:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        output_width = 2*100
                        output_height = 180
                        out_path = f"demo_{context_number}_traj_{trj_number}_activation_map.mp4"
                        out_activation_map = cv2.VideoWriter(os.path.join(
                            video_path, out_path), fourcc, frame_rate, (output_width, output_height))
                else:
                    out_path = os.path.join(
                        video_path, f"demo_{context_number}_traj_{trj_number}.png")

                # create the string to put on each frame
                if traj_result:
                    # res_string = f"Task {traj_result['variation_id']} - Reached {traj_result['reached']} - Picked {traj_result['picked']} - Success {traj_result['success']}"
                    # res_string = f"Reached {traj_result['reached']} - Picked {traj_result['picked']} - Success {traj_result['success']}"
                    # res_string = ""
                    # if predicted_bb:
                    #     res_string = f"Step {step} - Task {traj_result['variation_id']}"
                    res_string = None
                else:
                    res_string = f"Sample index {step}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.35
                thickness = 1
                crop = config['tasks_cfgs'][task_name]['crop'] if config['tasks_cfgs'][task_name].get('crop', None) is not None else  config['tasks_cfgs'][task_name]['agent_crop']
                for i, traj_frame in enumerate(traj_frames):
                    if "Real" in traj_file or "REAL" in traj_file:
                        indx_t = i
                    else:
                        indx_t = i -1
                    # and len(bb_frames) >= i+1:
                    if len(bb_frames) != 0 and i > 0 and len(bb_frames) >= i+1:
                        if len(bb_frames[indx_t][0]) == 4:
                            for indx, _ in enumerate(bb_frames[indx_t]):
                                bb = adjust_bb(
                                    bb=bb_frames[indx_t][indx],
                                    crop_params=crop,
                                    img_size=(traj_height, traj_width)) if adjust else bb_frames[indx_t][indx]
                                # bb = bb_frames[indx_t][indx]
                                traj_frame = np.array(cv2.rectangle(
                                    traj_frame.copy(),
                                    (int(bb[0]),
                                     int(bb[1])),
                                    (int(bb[2]),
                                     int(bb[3])),
                                    (255, 0, 0), 1))
                        else:
                            bb = adjust_bb(
                                bb=bb_frames[indx_t][0],
                                crop_params=crop,
                                img_size=(traj_height, traj_width)) if adjust else bb_frames[indx_t][0]
                            gt_bb_t = adjust_bb(
                                bb=gt_bb[indx_t][0],
                                crop_params=crop,
                                img_size=(traj_height, traj_width)) if adjust else gt_bb[indx_t][0]
                    
                    if len(gt_bb) != 0 and i > 0 and len(gt_bb) >= i+1:
                        if len(gt_bb[indx_t][0]) == 4:
                            for indx, _ in enumerate(gt_bb[indx_t]):
                                gt_bb_t = adjust_bb(
                                    bb=gt_bb[indx_t][indx],
                                    crop_params=crop,
                                    img_size=(traj_height, traj_width)) if adjust else gt_bb[indx_t][indx]
                                # gt_bb_t = gt_bb[indx_t][indx]
                                if indx == 0:
                                    color = (0, 255, 0)
                                elif indx == 1:
                                    color = (0, 255, 0)
                                traj_frame = np.array(cv2.rectangle(
                                    traj_frame.copy(),
                                    (int(gt_bb_t[0]),
                                     int(gt_bb_t[1])),
                                    (int(gt_bb_t[2]),
                                     int(gt_bb_t[3])),
                                    color, 1))
                        else:
                            for indx, _ in enumerate(gt_bb[indx_t]):
                                # = gt_bb[indx_t][indx][0]
                                # gt_bb_t = adjust_bb(
                                #     bb=gt_bb[indx_t][indx][0],
                                #     crop_params=crop) if adjust else gt_bb[indx_t][indx][0]
                                gt_bb_t = gt_bb[indx_t][indx][0]
                                if indx == 0:
                                    color = (0, 255, 0)
                                elif indx == 1:
                                    color == (0, 0, 255)
                                traj_frame = np.array(cv2.rectangle(
                                    traj_frame.copy(),
                                    (int(gt_bb_t[0]),
                                     int(gt_bb_t[1])),
                                    (int(gt_bb_t[2]),
                                     int(gt_bb_t[3])),
                                    color, 1))

                    #     if len(activation_map) != 0:
                    #         activation_map_t = activation_map[i-1]

                    #     # if i != len(traj_frames)-1 and len(predicted_slot) != 0:
                    #     #     cv2.putText(traj_frame,  res_string, (0, 80), font,
                    #     #                 font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                    #     #     cv2.putText(traj_frame,  f"Predicted slot {predicted_slot[i]}", (
                    #     #         0, 99), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                    #     # if predicted_bb:
                    #     #     cv2.putText(traj_frame,
                    #     #                 f"Conf-Score {round(float(predicted_conf_score[i-1]), 2)} - IoU {round(float(iou[i-1]), 2)}",
                    #     #                 (100, 180),
                    #     #                 font,
                    #     #                 font_scale,
                    #     #                 (0, 0, 255),
                    #     #                 thickness,
                    #     #                 cv2.LINE_AA)

                    #     # else:
                    #     #     cv2.putText(output_frame,  res_string, (0, 99), font,
                    #     #                 font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

                    if flip_channels:
                        traj_frame = traj_frame #[:, :, ::-1]

                    output_frame = cv2.hconcat(
                        [new_image, traj_frame])
                    cv2.putText(output_frame,  res_string, (0, 99), font,
                                font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
                    cv2.imwrite("frame.png", output_frame)
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

                        output_frame = np.array(traj_frame) #[:, :, ::-1]

                        cv2.imwrite("frame.png", output_frame)
                        out.write(output_frame)

                    out.release()


def read_results(base_path="/", task_name="pick_place", place=True):
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
        tp = [0.0, 0.0]
        fp = [0.0, 0.0]
        fp_pre_picking = [0.0, 0.0]
        fp_post_picking = [0.0, 0.0]
        fn = [0.0, 0.0]
        fn_pre_picking = [0.0, 0.0]
        fn_post_picking = [0.0, 0.0]
        iou_acc = [0.0, 0.0]
        OPEN_GRIPPER = np.array([-0.02, -0.25, -0.2, -0.02, -0.25, - 0.2])
        threshold = 0.5
        for context_file, traj_file in zip(context_files, traj_files):
            print(context_file, traj_file)
            with open(context_file, "rb") as f:
                context_data = pickle.load(f)
            with open(traj_file, "rb") as f:
                traj_data = pickle.load(f)

            
            print(len(traj_data))
            # number_frames += len(traj_data)
            for t in range(len(traj_data)):
                if t != 0:
                    # print(number_frames)
                    number_frames += 1
                    traj_frame = np.array(traj_data.get(
                                    t)['obs']['camera_front_image'])  # [:, :, ::-1])

                    ious = traj_data.get(t)['obs']['iou']

                    if ious.shape == ():
                        ious = ious[None]
                    for indx, iou in enumerate(ious):
                        iou_acc[indx] += iou
                        if iou > threshold:
                            tp[indx] += 1
                        else:
                            if traj_data.get(
                                    t)['obs'].get('predicted_bb') is not None:
                                fp[indx] += 1
                                # check for gripper open or close
                                gripper_state = traj_data.get(
                                    t)['obs'].get('gripper_qpos')
                                if np.array_equal(OPEN_GRIPPER, np.around(gripper_state, 2)):
                                    fp_pre_picking[indx] += 1
                                else:
                                    fp_post_picking[indx] += 1

                                # bb = adjust_bb(
                                #     bb=traj_data.get(
                                #         t)['obs']['predicted_bb'][indx],
                                #     crop_params=None)
                                # gt_bb_t = adjust_bb(
                                #     bb=traj_data.get(t)['obs']['gt_bb'][indx],
                                #     crop_params=None)
                                # traj_frame = np.array(cv2.rectangle(
                                #     traj_frame,
                                #     (int(bb[0]),
                                #     int(bb[1])),
                                #     (int(bb[2]),
                                #     int(bb[3])),
                                #     (0, 0, 255), 1))
                                # traj_frame = np.array(cv2.rectangle(
                                #     traj_frame,
                                #     (int(gt_bb_t[0]),
                                #     int(gt_bb_t[1])),
                                #     (int(gt_bb_t[2]),
                                #     int(gt_bb_t[3])),
                                #     (0, 255, 0), 1))
                                # cv2.imwrite(
                                #     f"debug/{traj_file.split('/')[-1].split('.')[0]}_{t}.png", traj_frame)
                            else:
                                fn[indx] += 1
                                # check for gripper open or close
                                gripper_state = traj_data.get(
                                    t)['obs'].get('gripper_qpos')
                                if np.array_equal(OPEN_GRIPPER, np.around(gripper_state, 2)):
                                    fn_pre_picking[indx] += 1
                                else:
                                    fn_post_picking[indx] += 1
                                # traj_frame = np.array(traj_data.get(
                                #     t)['obs']['camera_front_image'][:, :, ::-1])
                                # cv2.imwrite(
                                #     f"debug/{traj_file.split('/')[-1].split('.')[0]}_{t}.png", traj_frame)

            traj_result = None
            try:
                json_file = traj_file.split('.')[-2]
                with open(f"{json_file}.json", "rb") as f:
                    traj_result = json.load(f)
                    file_cnt += 1
            except:
                pass

            if traj_result.get('success', 1) == 1:
                success_cnt += 1
            if traj_result.get('reached', 1) == 1:
                reached_cnt += 1
            if traj_result.get('picked', 1) == 1:
                picked_cnt += 1

            if 'avg_iou' in traj_result.keys():
                mean_iou += traj_result['avg_iou']
                tp_avg += traj_result['avg_tp']

                if traj_result['avg_iou'] < threshold:
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
            tp[0]+fp[0]+fn[0]), "Number of frames must be equal to tp+fp+fn"
        
        res = dict()
        if not place:
            print(f"Total Number tp {tp[0]}")
            print(f"Total Number fp {fp[0]}")
            print(f"Total Number fp-pre-picking {fp_pre_picking[0]}")
            print(f"Total Number fp-post-picking {fp_post_picking[0]}")
            print(f"Total Number fn {fn[0]}")
            print(f"Total Number fn-pre-picking {fn_pre_picking[0]}")
            print(f"Total Number fn-post-picking {fn_post_picking[0]}")
            pre = round(tp[0]/(tp[0]+fp[0]),3)
            rec = round(tp[0]/(tp[0]+fn[0]),3)
            print(f"Precision {pre} - Recall {rec}")

            res = dict()
            res['tp'] = tp[0]
            res['fp'] = fp[0]
            res['fp_pre_picking'] = fp_pre_picking[0]
            res['fp_post_picking'] = fp_post_picking[0]
            res['fn'] = fn[0]
            res['fn_pre_picking'] = fn_pre_picking[0]
            res['fn_post_picking'] = fn_post_picking[0]
            res['prec'] = pre
            res['rec'] = rec
        
        else:
            print(f"\n---- Target ----")
            print(f"Target: Total Number tp {tp[0]}")
            print(f"Target: Total Number fp {fp[0]}")
            print(f"Target: Total Number fp-pre-picking {fp_pre_picking[0]}")
            print(f"Target: Total Number fp-post-picking {fp_post_picking[0]}")
            print(f"Target: Total Number fn {fn[0]}")
            print(f"Target: Total Number fn-pre-picking {fn_pre_picking[0]}")
            print(f"Target: Total Number fn-post-picking {fn_post_picking[0]}")
            print(f"Target: Mean iou {iou_acc[0]/number_frames}")
            pre_target = round(tp[0]/(tp[0]+fp[0]),3)
            rec_target = round(tp[0]/(tp[0]+fn[0]),3)
            print(f"Target: Precision {pre_target} - Recall {rec_target}")
            
            
            res['target'] = dict()
            res['target']['tp'] = tp[0]
            res['target']['fp'] = fp[0]
            res['target']['fp_pre_picking'] = fp_pre_picking[0]
            res['target']['fp_post_picking'] = fp_post_picking[0]
            res['target']['fn'] = fn[0]
            res['target']['fn_pre_picking'] = fn_pre_picking[0]
            res['target']['fn_post_picking'] = fn_post_picking[0]
            res['target']['prec'] = pre_target
            res['target']['rec'] = rec_target
            res['target']['mean_iou'] = round(iou_acc[0]/number_frames, 3)
            
            
            print(f"\n---- Placing ----")
            print(f"Placing: Total Number tp {tp[1]}")
            print(f"Placing: Total Number fp {fp[1]}")
            print(f"Placing: Total Number fp-pre-picking {fp_pre_picking[1]}")
            print(f"Placing: Total Number fp-post-picking {fp_post_picking[1]}")
            print(f"Placing: Total Number fn {fn[1]}")
            print(f"Placing: Total Number fn-pre-picking {fn_pre_picking[1]}")
            print(f"Placing: Total Number fn-post-picking {fn_post_picking[1]}")
            pre_placing = round(tp[1]/(tp[1]+fp[1]),3)
            rec_placing = round(tp[1]/(tp[1]+fn[1]),3)
            print(f"Placing: Precision {pre_placing} - Recall {rec_placing}")
            
            res['placing'] = dict()
            res['placing']['tp'] = tp[1]
            res['placing']['fp'] = fp[1]
            res['placing']['fp_pre_picking'] = fp_pre_picking[1]
            res['placing']['fp_post_picking'] = fp_post_picking[1]
            res['placing']['fn'] = fn[1]
            res['placing']['fn_pre_picking'] = fn_pre_picking[1]
            res['placing']['fn_post_picking'] = fn_post_picking[1]
            res['placing']['prec'] = pre_placing
            res['placing']['rec'] = rec_placing
            res['placing']['mean_iou'] = round(iou_acc[1]/number_frames, 3)
            
            
            print(f"\n---- Global ----")
            global_tp = tp[0] + tp[1]
            global_fp = fp[0] + fp[1]
            global_fp_pre_picking = fp_pre_picking[0] + fp_pre_picking[1]
            global_fp_post_picking = fp_post_picking[0] + fp_post_picking[1]
            global_fn = fn[0] + fn[1]
            global_fn_pre_picking = fn_pre_picking[0] + fn_pre_picking[1]
            global_fn_post_picking = fn_post_picking[0] + fn_post_picking[1]
            print(f"Total Number tp {global_tp}")
            print(f"Total Number fp {global_fp}")
            print(f"Total Number fp-pre-picking {global_fp_pre_picking}")
            print(f"Total Number fp-post-picking {global_fp_post_picking}")
            print(f"Total Number fn {global_fp_post_picking}")
            print(f"Total Number fn-pre-picking {global_fn_pre_picking}")
            print(f"Total Number fn-post-picking {global_fn_post_picking}")
            global_pre = round(global_tp/(global_tp+global_fp),3)
            global_rec = round(global_tp/(global_tp+global_fn),3)
            print(f"Precision {global_pre} - Recall {global_rec}")
            
            res['global'] = dict()
            res['global']['tp'] = global_tp
            res['global']['fp'] = global_fp
            res['global']['fp_pre_picking'] = global_fp_pre_picking
            res['global']['fp_post_picking'] = global_fp_post_picking
            res['global']['fn'] = global_fn
            res['global']['fn_pre_picking'] = global_fn_pre_picking
            res['global']['fn_post_picking'] = global_fn_post_picking
            res['global']['prec'] = global_pre
            res['global']['rec'] = global_rec
            res['global']['mean_iou'] = round((iou_acc[0]+iou_acc[1])/(2*number_frames), 3)

        with open(os.path.join(base_path, f'results_{threshold}.json'), 'w') as json_file:
            json.dump(res, json_file, indent=4)
        
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
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    if args.metric != "results":
        # 1. create video
        create_video_for_each_trj(
            base_path=args.base_path, task_name=args.task)
    else:
        import time
        read_results(base_path=args.base_path, task_name=args.task, place='KP' in args.base_path )
        time.sleep(3)
