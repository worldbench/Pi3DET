import os
import io
import pickle
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import copy
from pathlib import Path
from ..dataset import DatasetTemplate
from ...utils import box_utils
from . import pi3det_utils

class PI3DET_Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.image_shape = np.array([1280, 800])
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.data_infos = []
        self.include_pi3det_data(self.mode)

    def include_pi3det_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Pi3Det dataset')
        data_infos = []
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = Path(self.root_path) / info_path
            if not info_path.exists():
                raise FileNotFoundError(f'Pi3Det info file {info_path} does not exist.')
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                data_infos.extend(infos)
        self.data_infos.extend(data_infos)

        if self.logger is not None:
            self.logger.info('Total samples for Pi3Det dataset: %d' % (len(data_infos)))

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.data_infos) * self.total_epochs

        return len(self.data_infos)
    
    def remove_ego_points(self, points, center_radius=1.0):
        mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
        return points[mask]  

    def get_lidar(self, file_path):
        lidar_file = os.path.join(self.root_path, file_path)
        assert os.path.exists(lidar_file), f'Lidar file {lidar_file} does not exist.'
        points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        # points = self.remove_ego_points(points, center_radius=0.5)
        points = np.hstack([points, np.zeros((points.shape[0], 1)).astype(points.dtype)])      
        return points

    def get_image(self, file_path):

        img_file = self.root_path / file_path
        assert img_file.exists()
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_fov_flag(self, points, calib):
        extristric = calib['E']
        K = calib['I']
        D = calib['D']
        extend_points = pi3det_utils.cart_to_hom(points[:,:3])
        points_cam = extend_points @ extristric.T
        rvecs = np.zeros((3,1))
        tvecs = np.zeros((3,1))

        pts_img, _ = cv2.projectPoints(points_cam[:,:3].astype(np.float32), rvecs, tvecs,
                K, D)
        imgpts = pts_img[:,0,:]
        depth = points_cam[:,2]

        imgpts = np.round(imgpts)
        kept1 = (imgpts[:, 1] >= 0) & (imgpts[:, 1] < self.image_shape[1]) & \
                    (imgpts[:, 0] >= 0) & (imgpts[:, 0] < self.image_shape[0]) & \
                    (depth > 0.0)
        return kept1

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.data_infos)

        info = copy.deepcopy(self.data_infos[index])
        calib = info['calib']
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])
        input_dict = {
            'frame_id': info['frame_id'],
            'calib': calib,
            'platform': info['platform']
        }

        if 'annos' in info:
            annos = info['annos']
            gt_names = annos['gt_names']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            if self.dataset_cfg.get('SHIFT_COOR', None):
                gt_boxes_lidar[:, 0:3] += self.dataset_cfg.SHIFT_COOR
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if self.dataset_cfg.get('REMOVE_ORIGIN_GTS', None) and self.training:
                input_dict['points'] = box_utils.remove_points_in_boxes3d(input_dict['points'], input_dict['gt_boxes'])
                mask = np.zeros(gt_boxes_lidar.shape[0], dtype=np.bool_)
                input_dict['gt_boxes'] = input_dict['gt_boxes'][mask]
                input_dict['gt_names'] = input_dict['gt_names'][mask]

            if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
                input_dict['gt_boxes'] = None

        if "points" in get_item_list:
            point_clout_path = info['point_clout_path']
            points = self.get_lidar(point_clout_path)
            if self.dataset_cfg.FOV_POINTS_ONLY:
                fov_flag = self.get_fov_flag(points, calib)
                points = points[fov_flag]
            if self.dataset_cfg.get('SHIFT_COOR', None):
                points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
            input_dict['points'] = points

        if "images" in get_item_list:
            image_path = info['image_path']
            input_dict['images'] = self.get_image(image_path)

        # load saved pseudo label for unlabel data
        if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
            self.fill_pseudo_labels(input_dict)
            
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict
    
    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'pred_labels': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7]),
                'platform': np.zeros(num_samples),
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]

            if self.dataset_cfg.get('SHIFT_COOR', None):
                #print ("*******WARNING FOR SHIFT_COOR:", self.dataset_cfg.SHIFT_COOR)
                pred_boxes[:, 0:3] -= self.dataset_cfg.SHIFT_COOR

            # BOX FILTER
            if self.dataset_cfg.get('TEST', None) and self.dataset_cfg.TEST.BOX_FILTER['FOV_FILTER']:
                box_preds_lidar_center = pred_boxes[:, 0:3]
                fov_flag = self.get_fov_flag(box_preds_lidar_center, calib)
                pred_boxes = pred_boxes[fov_flag]
                pred_labels = pred_labels[fov_flag]
                pred_scores = pred_scores[fov_flag]

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]
            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            single_pred_dict['platform'] = batch_dict['platform'][index]
            annos.append(single_pred_dict)
        return annos
    
    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.data_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            map_name_to_kitti = {
                'Vehicle': 'Car',
                'Pedestrian': 'Pedestrian',
                'Cyclist': 'Cyclist',
                'Sign': 'Sign',
                'Car': 'Car'
            }
            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict
        
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.data_infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict