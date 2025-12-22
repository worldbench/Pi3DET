import pickle
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import copy
from ..dataset import DatasetTemplate
from . import m3ed_utils
from ...utils import box_utils

class M3ED_CP_Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        '''
        M3ED dataset for cross platform
        '''
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

        platform = dataset_cfg.PLATFORM_PATH[self.mode]
        self.root_split_path = self.root_path / platform / ('training' if self.mode != 'test' else 'testing')

        self.include_m3ed_data()

    def include_m3ed_data(self):
        anno_dict_path = self.root_split_path / 'labels.pkl'
        calib_dict_path = self.root_split_path / 'calibs.pkl'
        info_path = self.root_split_path.parent / 'trainvalSets' / ('training.txt' if self.mode != 'test' else 'testing.txt')
        with open(anno_dict_path, 'rb') as f:
            self.anno_dict = pickle.load(f)
        with open(calib_dict_path, 'rb') as f:
            self.calib_dict = pickle.load(f)
        self.image_shape = np.array([1280, 800])
        self.data_list_info = self.load_saved_frame_txt(info_path)
        if self.logger is not None:
            self.logger.info('Total samples for M3ED Cross Platcorm dataset: %d' % (len(self.calib_dict)))

    def __len__(self):
        return len(self.data_list_info)

    def load_saved_frame_txt(self, txt_path):

        save_frame_info = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                label = line.strip().split(' ')
                frame_info = dict(
                    platform = label[0],
                    terrian = label[1],
                    sequence = label[2],
                    frame_index = label[3]
                )
                save_frame_info.append(frame_info)
        f.close()
        return save_frame_info

    def get_lidar(self, frame_id):
        point_cloud_path = self.root_split_path / 'point_cloud' / f'{frame_id}.bin'
        points = np.fromfile(point_cloud_path, dtype=np.float64).reshape(-1, 4)
        return points
    
    def get_calib(self, frame_id):
        calib = self.calib_dict[frame_id]
        return calib

    def get_fov_flag(self, points, calib):
        extristric = calib['extristric']
        K = calib['K']
        D = calib['D']
        extend_points = m3ed_utils.cart_to_hom(points[:,:3])
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

    def get_virtual_pose(self, pose, drone=False):
        # n --> virtual
        Ln_T_L0 = m3ed_utils.transform_inv(pose)
        r = Rotation.from_matrix(Ln_T_L0[:3,:3])
        Rn_T_R0_x, Rn_T_R0_y, _ = r.as_euler('xyz')
        R_x = m3ed_utils.rotx(Rn_T_R0_x)
        R_y = m3ed_utils.roty(Rn_T_R0_y)
        virtual_r_matrix = R_y @ R_x
        virtual_pose = np.eye(4)
        virtual_pose[:3, :3] = virtual_r_matrix

        if drone:
            position_n = Ln_T_L0[:3,3]
            change_z = position_n[2] - 1.7
            virtual_translate = np.eye(4)
            virtual_translate[2,3] = change_z
            virtual_pose = virtual_translate @ virtual_pose

        return virtual_pose

    def convert_boxes_from_vir_to_n(self, pose, boxes, drone=False):
        # vir pose
        virtual_pose = self.get_virtual_pose(pose, drone)
        pose = m3ed_utils.transform_inv(virtual_pose)
        if not drone:
            boxes[:,2] += 1.3
        r = Rotation.from_matrix(pose[:3,:3])
        ego2world_yaw = r.as_euler('xyz')[-1]
        boxes_global = boxes.copy()
        expand_centroids = np.concatenate([boxes[:, :3], 
                                           np.ones((boxes.shape[0], 1))], 
                                           axis=-1)
        centroids_global = np.dot(expand_centroids, pose.T)[:,:3]        
        boxes_global[:,:3] = centroids_global
        boxes_global[:,6] += ego2world_yaw
        return boxes_global

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
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            if self.dataset_cfg.get('SHIFT_COOR', None):
                pred_boxes[:, 0:3] -= self.dataset_cfg.SHIFT_COOR

            # BOX FILTER
            # if self.dataset_cfg.get('TEST', None) and self.dataset_cfg.TEST.BOX_FILTER['FOV_FILTER']:
            #     box_preds_lidar_center = pred_boxes[:, 0:3]
            #     fov_flag = getattr(self, f'seq_{0}').get_fov_flag(box_preds_lidar_center)
            #     pred_boxes = pred_boxes[fov_flag]
            #     pred_labels = pred_labels[fov_flag]
            #     pred_scores = pred_scores[fov_flag]

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            annos.append(single_pred_dict)

        return annos

    def get_anno_info(self, index, calib):
        frame_id = str(index).zfill(5)
        anno_dict = copy.deepcopy(self.anno_dict[frame_id])
        frame_info = self.data_list_info[index]
        if frame_info['platform'] != 'car':
            converet_box = anno_dict['gt_boxes']
            if frame_info['platform'] == 'spot':
                drone = False
            else:
                drone = True
            if frame_info['sequence'] != 'srt_under_bridge_2':
                converet_box = self.convert_boxes_from_vir_to_n(calib['pose'], converet_box, drone)
            else:
                converet_box[:,2] += 1.3
            anno_dict['gt_boxes'] = converet_box
        return anno_dict

    def kitti_eval(self, eval_det_annos, eval_gt_annos, class_names):
        from ..kitti.kitti_object_eval_python import eval as kitti_eval

        map_name_to_kitti = {
            'car': 'Car',
            'Vehicle': 'Car',
            'Pedestrian': 'Pedestrian'
        }

        def transform_to_kitti_format(annos, info_with_fakelidar=False, is_gt=False):
            for anno in annos:
                if 'name' not in anno:
                    anno['name'] = anno['gt_names']
                    anno.pop('gt_names')

                for k in range(anno['name'].shape[0]):
                    if anno['name'][k] in map_name_to_kitti:
                        anno['name'][k] = map_name_to_kitti[anno['name'][k]]
                    else:
                        anno['name'][k] = 'Person_sitting'

                if 'boxes_lidar' in anno:
                    gt_boxes_lidar = anno['boxes_lidar'].copy()
                else:
                    gt_boxes_lidar = anno['gt_boxes'].copy()

                # filter by range
                if self.dataset_cfg.get('GT_FILTER', None) and \
                        self.dataset_cfg.GT_FILTER.RANGE_FILTER:
                    if self.dataset_cfg.GT_FILTER.get('RANGE', None):
                        point_cloud_range = self.dataset_cfg.GT_FILTER.RANGE
                    else:
                        point_cloud_range = self.point_cloud_range
                        point_cloud_range[2] = -10
                        point_cloud_range[5] = 10

                    mask = box_utils.mask_boxes_outside_range_numpy(gt_boxes_lidar,
                                                                    point_cloud_range,
                                                                    min_num_corners=1)
                    gt_boxes_lidar = gt_boxes_lidar[mask]
                    anno['name'] = anno['name'][mask]
                    if not is_gt:
                        anno['score'] = anno['score'][mask]
                        anno['pred_labels'] = anno['pred_labels'][mask]

                # filter by fov
                if is_gt and self.dataset_cfg.get('GT_FILTER', None):
                    if self.dataset_cfg.GT_FILTER.get('FOV_FILTER', None):
                        fov_gt_flag = self.extract_fov_gt(
                            gt_boxes_lidar, self.dataset_cfg['FOV_DEGREE'], self.dataset_cfg['FOV_ANGLE']
                        )
                        gt_boxes_lidar = gt_boxes_lidar[fov_gt_flag]
                        anno['name'] = anno['name'][fov_gt_flag]

                anno['bbox'] = np.zeros((len(anno['name']), 4))
                anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
                anno['truncated'] = np.zeros(len(anno['name']))
                anno['occluded'] = np.zeros(len(anno['name']))

                if len(gt_boxes_lidar) > 0:
                    if info_with_fakelidar:
                        gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes_lidar)

                    gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
                    anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
                    anno['location'][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
                    anno['location'][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
                    anno['location'][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
                    dxdydz = gt_boxes_lidar[:, 3:6]
                    anno['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
                    anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
                    anno['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + anno['rotation_y']
                else:
                    anno['location'] = anno['dimensions'] = np.zeros((0, 3))
                    anno['rotation_y'] = anno['alpha'] = np.zeros(0)

        # self.filter_det_results(eval_det_annos, self.oracle_infos)
        transform_to_kitti_format(eval_det_annos)
        transform_to_kitti_format(eval_gt_annos, info_with_fakelidar=False, is_gt=True)

        kitti_class_names = []
        for x in class_names:
            if x in map_name_to_kitti:
                kitti_class_names.append(map_name_to_kitti[x])
            else:
                kitti_class_names.append('Person_sitting')
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
        )
        return ap_result_str, ap_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if kwargs['eval_metric'] == 'kitti':
            eval_det_annos = copy.deepcopy(det_annos)
            eval_gt_annos = copy.deepcopy(self.anno_dict2_list())
            return self.kitti_eval(eval_det_annos, eval_gt_annos, class_names)
        else:
            raise NotImplementedError

    def anno_dict2_list(self):
        anno_list = []
        for frame_index in range(len(self.data_list_info)):
            single_anno_info = {}
            frame_anno_dict = self.get_anno_info(frame_index, self.get_calib(str(frame_index).zfill(5)))
            gt_boxes_3d = frame_anno_dict['gt_boxes']
            gt_names = frame_anno_dict['gt_names']
            single_anno_info.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_3d[:,:7]
            })
            anno_list.append(single_anno_info)
        return anno_list

    def __getitem__(self, index):
        frame_id = str(index).zfill(5)
        points = self.get_lidar(frame_id)
        calib = self.get_calib(frame_id)

        if self.dataset_cfg.FOV_POINTS_ONLY:
            fov_flag = self.get_fov_flag(points, calib)
            points = points[fov_flag]
        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
        
        input_dict = {
            'points': points,
            'frame_id': frame_id,
            'calib': calib,
            'image_shape': self.image_shape
        }
        anno_dict = self.get_anno_info(index, calib)
        input_dict.update({
            'gt_names': anno_dict['gt_names'],
            'gt_boxes': anno_dict['gt_boxes']
        })

        if self.dataset_cfg.get('SHIFT_COOR', None):
            input_dict['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR

        if self.dataset_cfg.get('REMOVE_ORIGIN_GTS', None) and self.training:
            input_dict['points'] = box_utils.remove_points_in_boxes3d(input_dict['points'], input_dict['gt_boxes'])
            mask = np.zeros(anno_dict['gt_boxes'].shape[0], dtype=np.bool_)
            input_dict['gt_boxes'] = input_dict['gt_boxes'][mask]
            input_dict['gt_names'] = input_dict['gt_names'][mask]

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict