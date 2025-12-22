import numpy as np
import json
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
import torch

LABEL_MAP = {
    'Car' : 0, 
    'Pedestrian': 1, 
    'Cyclist': 2
}

LABEL_MAP_LIST = ['Car', 'Pedestrian', 'Cyclist']

def rotx(t):
    """ 3D Rotation about the x-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def transform_inv(T):
    T_inv = np.eye(4)
    T_inv[:3,:3] = T[:3,:3].T
    T_inv[:3,3] = -1.0 * ( T_inv[:3,:3] @ T[:3,3] )
    return T_inv

def cart_to_hom(pts):
    """
    :param pts: (N, 3 or 2)
    :return pts_hom: (N, 4 or 3)
    """
    pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
    return pts_hom

def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask

def xtreme2pcd_gt(result_path, file_name):
    json_res_path = result_path / file_name
    if json_res_path.exists():
        with open(json_res_path, 'r') as file:
            data = json.load(file)[0]
        boxes = []
        boxes_name = []
        for box in data['objects']:
            box_item = []
            box_name = box['className']
            boxes_name.append(box_name)
            contour = box['contour']
            size_3d = contour['size3D']
            center_3d = contour['center3D']
            rotation_3d = contour['rotation3D']
            size_3d = [val for _, val in size_3d.items()]
            center_3d = [val for _, val in center_3d.items()]
            rotation_3d = [val for _, val in rotation_3d.items()]
            box_item = center_3d + size_3d + [rotation_3d[-1]]
            boxes.append(box_item)
    
        return np.array(boxes), np.array(boxes_name)
    
def save_single_frame_gtdatabase(root_path, save_path, points, bboxes, labels, file_name, replace=False):
    all_db_infos = {}
    num_obj = bboxes.shape[0]
    point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
        torch.from_numpy(points[:, 0:3]), torch.from_numpy(bboxes[:,:7])
    ).numpy()  # (nboxes, npoints)
    names = [LABEL_MAP_LIST[label] for label in labels]
    for i in range(num_obj):
        filename = '%s_%s_%d.bin' % (file_name, names[i], i)
        filepath = save_path / filename
        gt_points = points[point_indices[i] > 0]
        gt_points[:, :3] -= bboxes[i, :3]
        gt_points = gt_points.astype(np.float32)
        if filepath.exists():
            if replace:
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)
        else:
            with open(filepath, 'w') as f:
                gt_points.tofile(f)         

        db_path = str(filepath.relative_to(root_path))
        db_info = {'name': names[i], 'path': db_path, 'frame_idx': file_name, 'gt_idx': i,
                    'box3d_lidar': bboxes[i,:7], 'num_points_in_gt': gt_points.shape[0],}

        if names[i] in all_db_infos:
            all_db_infos[names[i]].append(db_info)
        else:
            all_db_infos[names[i]] = [db_info]

    return all_db_infos

def group_samples(N, seq_name, M):
    num_groups = (N + M - 1) // M
    group_indices = []
    seq_name_to_len = []
    sample_idx_list = []
    last_num = N - num_groups*M

    for i in range(N):
        group_idx = i // M 
        sample_idx = i - M*group_idx
        sqquence_name = f'm3ed_{seq_name}_{int(M/10)}s_sequence_' + str(group_idx)
        group_indices.append(sqquence_name)
        sample_idx_list.append(sample_idx)
        if group_idx < num_groups-1:
            seq_name_to_len.append(M)
        else:
            seq_name_to_len.append(last_num)


    return group_indices, seq_name_to_len, sample_idx_list