
import cv2
import numpy as np
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file

def load_img_w_box3d(img_path, item_info=None):
    img = cv2.imread(img_path)
    # box_3d = np.array(item_info['ground_info'][0]['bbox_3d'])[None, :]
    # camera extrinsic and intrinsic and distortion
    # corners_2d, _ = conver_box2d(box_3d, img.shape, item_info)
    # img = m3ed_util.draw_projected_box3d(img, corners_2d[0], color=(0, 255, 0))
    return img

def load_cfg_from_yaml(cfg_file_path):
    cfg = EasyDict()
    cfg_from_yaml_file(cfg_file_path, cfg)

    return cfg
