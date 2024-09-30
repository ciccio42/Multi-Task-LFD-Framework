import torch
import re
import numpy as np


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

ENV_OBJECTS = {}


def find_obj_names(obs, task_name):
    ENV_OBJECTS[task_name] = dict()

    for obs_name in obs.keys():
        pass


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


def adjust_bb(bb, crop_params=[20, 25, 80, 75], img_size=(200, 360)):

    x1_old, y1_old, x2_old, y2_old = bb
    x1_old = int(x1_old)
    y1_old = int(y1_old)
    x2_old = int(x2_old)
    y2_old = int(y2_old)

    top, left = crop_params[0], crop_params[2]
    img_height, img_width = img_size[0], img_size[1]
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


def reach(obs, task_name):
    # get target obj-id
    if task_name == "pick_place":
        target_object = obs['target-object']
        target_box_id = obs['target-box-id']
    else:
        assert task_name != "nut_assembly", "Not implemented error"

    object_name = ENV_OBJECTS[task_name]['obj_names'][target_object]

    print(obs["eef_pos"][2])
    if obs["eef_pos"][2] < 0.77:
        # reaching phase
        if np.linalg.norm(obs[f"{object_name}_to_robot0_eef_pos"], 2) < 0.01:
            return True
        else:
            return False
    return False


def pick(obs, task_name):
    # get target obj-id
    if task_name == "pick_place":
        target_box_id = obs['target-object']
    else:
        assert task_name != "nut_assembly", "Not implemented error"

    object_name = ENV_OBJECTS[task_name]['obj_names'][target_box_id]
    print(f"{obs['gripper_qpos']}")
    if obs["eef_pos"][2] < 0.77 and np.linalg.norm(obs[f"{object_name}_to_robot0_eef_pos"], 2) < 0.01:
        pass
    return False


def place(obs, task_name):
    # get target obj-id
    if task_name == "pick_place":
        target_object = obs['target-object']
        target_box_id = obs['target-box-id']
    else:
        assert task_name != "nut_assembly", "Not implemented error"

    object_name = ENV_OBJECTS[task_name]['obj_names'][target_object]

    return False
