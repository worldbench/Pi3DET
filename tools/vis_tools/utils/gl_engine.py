import pyqtgraph.opengl as gl
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QVector3D, QFont
from PyQt5.QtWidgets import QFrame

import cv2
import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
from PIL import Image
import logging

# from . import m3ed_util as utils
import supervision as sv

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

PRED_COLORS = [
    (  0, 255,   0, 255),  # Crimson
    (233, 150,  70, 255),  # Darksalmon
    (220,  20,  60, 255),  # Crimson
    (255,  61,  99, 255),  # Red
    (0,     0, 230, 255),  # Blue
    (255,  61,  99, 255),  # Red
    (0,     0, 230, 255),  # Blue
    (47,   79,  79, 255),  # Darkslategrey
    (112,  128, 144, 255),  # Slategrey
    (  0, 255,   0, 255),  # cars in green
    (255,   0,   0, 255),  # pedestrian in red
    (255, 255,   0, 255),  # cyclists in yellow
    (255, 127,  80, 255),  # Coral
    (233, 150,  70, 255),  # Darksalmon
        ]

class AL_viewer(gl.GLViewWidget):
    
    def __init__(self):
        super().__init__()

        self.noRepeatKeys = [Qt.Key.Key_W, Qt.Key.Key_S, Qt.Key.Key_A, Qt.Key.Key_D, Qt.Key.Key_Q, Qt.Key.Key_E,
            Qt.Key.Key_Right, Qt.Key.Key_Left, Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_PageUp, Qt.Key.Key_PageDown]
        
        self.speed = 1
        # self.setBackgroundColor('w')
        
    def evalKeyState(self):
        vel_speed = 10 * self.speed 
        if len(self.keysPressed) > 0:
            for key in self.keysPressed:
                if key == Qt.Key.Key_Right:
                    self.orbit(azim=-self.speed, elev=0)
                elif key == Qt.Key.Key_Left:
                    self.orbit(azim=self.speed, elev=0)
                elif key == Qt.Key.Key_Up:
                    self.orbit(azim=0, elev=-self.speed)
                elif key == Qt.Key.Key_Down:
                    self.orbit(azim=0, elev=self.speed)
                elif key == Qt.Key.Key_A:
                    self.pan(vel_speed * self.speed, 0, 0, 'view-upright')
                elif key == Qt.Key.Key_D:
                    self.pan(-vel_speed, 0, 0, 'view-upright')
                elif key == Qt.Key.Key_W:
                    self.pan(0, vel_speed, 0, 'view-upright')
                elif key == Qt.Key.Key_S:
                    self.pan(0, -vel_speed, 0, 'view-upright')
                elif key == Qt.Key.Key_Q:
                    self.pan(0, 0, vel_speed, 'view-upright')
                elif key == Qt.Key.Key_E:
                    self.pan(0, 0, -vel_speed, 'view-upright')
                elif key == Qt.Key.Key_PageUp:
                    pass
                elif key == Qt.Key.Key_PageDown:
                    pass
                self.keyTimer.start(16)
        else:
            self.keyTimer.stop()


def get_camera_position(view):

    d = view.opts['distance']
    az = view.opts['azimuth']
    el = view.opts['elevation']
    print(d, az, el)

class QHLine(QFrame):

    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)

def ndarray_to_pixmap(ndarray):
 
    if len(ndarray.shape) == 2:
        height, width = ndarray.shape
        bytes_per_line = width
        qimage = QImage(ndarray.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    elif len(ndarray.shape) == 3:
        height, width, channels = ndarray.shape
        bytes_per_line = 3 * width
        qimage = QImage(ndarray.data, width, height, bytes_per_line, QImage.Format_BGR888)
    
    elif len(ndarray.shape) == 4:
        height, width, channels = ndarray.shape
        bytes_per_line = 4 * width
        qimage = QImage(ndarray.data, width, height, bytes_per_line, QImage.Format_BGRA8888)
    
    else:
        raise ValueError("ndarray must be 3D or 2D ndarry")
    
    pixmap = QPixmap.fromImage(qimage)
    return pixmap

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, m, centroid

def normalize_feature(feature):
    """
    Normalize the feature to the range [0, 1].
    """
    feature = feature - np.min(feature)
    feature = feature / (np.max(feature) - np.min(feature))
    return feature

def get_points_mesh(points, size, colors = None):

    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    if colors is None:
        feature = normalize_feature(points[:,2])
        feature = points[:,2]
        norm = mpl.colors.Normalize(vmin=-2.5, vmax=1.5)
        # norm = mpl.colors.Normalize(vmin=feature.min()+0.5, vmax=feature.max()-0.5)
        cmap = cm.jet 
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        colors = m.to_rgba(feature)
        colors[:, [2, 1, 0, 3]] = colors[:, [0, 1, 2, 3]]
        colors[:, 3] = 0.5
        # colors = np.zeros([points.shape[0], 4])
        # colors[:,0] = 0.5
        # colors[:,1] = 0.5
        # colors[:,2] = 0.5
        # colors[:,3] = 255

    else:
        if isinstance(colors, torch.Tensor):
            colors = colors.detach().cpu().numpy()

    mesh = gl.GLScatterPlotItem(pos=np.asarray(points[:, 0:3]), size=size, color=colors)

    return mesh

def resize_img(img: np.ndarray, target_width, target_height):
    image = Image.fromarray(img)
    image_resized = image.resize((target_width, target_height))
    return np.array(image_resized)

def save_temp_ndarry(ndarry):
    cv2.imwrite('../vis_tools/qt_windows/tmp/temp.jpg', ndarry*255)

def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False
    return logger


def create_boxes(bboxes_3d, scores = None, colors = None):
    boxes = {}
    box_items = []
    l1_items = []
    l2_items = []
    score_items = []

    # create annotation boxes
    for i in range(bboxes_3d.shape[0]):

        annotation = bboxes_3d[i]
        if annotation.shape[0] == 8:
            x, y, z, w, l, h, rotation, category = annotation
        else:
            x, y, z, w, l, h, rotation = annotation[:7]
            category = annotation[-1]

        rotation = np.rad2deg(rotation) + 90
        if colors is None:
            try:
                color = PRED_COLORS[int(category)]
            except IndexError:
                color = (255, 255, 255, 255)
        else:
            color = colors[i]

        box = gl.GLBoxItem(QVector3D(1, 1, 1), color=color)
        box.setSize(l, w, h)
        box.translate(-l / 2, -w / 2, -h / 2)
        box.rotate(angle=rotation, x=0, y=0, z=1)
        box.translate(x, y, z)
        box_items.append(box)

        #################
        # heading lines #
        #################

        p1 = [-l / 2, -w / 2, -h / 2]
        p2 = [l / 2, -w / 2, h / 2]

        pts = np.array([p1, p2])

        l1 = gl.GLLinePlotItem(pos=pts, width=2, color=color, antialias=True, mode='lines')
        l1.rotate(angle=rotation, x=0, y=0, z=1)
        l1.translate(x, y, z)

        l1_items.append(l1)

        p3 = [-l / 2, -w / 2, h / 2]
        p4 = [l / 2, -w / 2, -h / 2]

        pts = np.array([p3, p4])

        l2 = gl.GLLinePlotItem(pos=pts, width=2, color=color, antialias=True, mode='lines')
        l2.rotate(angle=rotation, x=0, y=0, z=1)
        l2.translate(x, y, z)

        l2_items.append(l2)

        distance = np.linalg.norm([x, y, z], axis=0)
        boxes[distance] = (box, l1, l2)

        if scores is not None:
            round_score = np.round(scores[i],2)
            text_item = gl.GLTextItem(pos=[0,0,0], text=str(round_score), color=color, font=QFont('Helvetica', 8))
            text_item.translate(x, y, z)
            score_items.append(text_item)

    box_info = {
        'boxes' : boxes,
        'box_items' : box_items,
        'l1_items' : l1_items,
        'l2_items' : l2_items,
        'score_items': score_items
    }

    return box_info


def create_boxes_new(bboxes_3d, scores = None, colors = None, line_width = 4):

    box_items = []
    boxes_vec_points = np.zeros([bboxes_3d.shape[0], 3, 8])
    l,w,h = bboxes_3d[:,3], bboxes_3d[:,4], bboxes_3d[:,5]
    c_xyz = bboxes_3d[:,:3][:,:, None]

    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    boxes_vec_points[:, 0, :] = np.transpose(np.stack(x_corners))
    boxes_vec_points[:, 1, :] = np.transpose(np.stack(y_corners))
    boxes_vec_points[:, 2, :] = np.transpose(np.stack(z_corners))

    rotzs = []
    for box in bboxes_3d:
        rotzs.append(rotz(box[6]))
    rotzs = np.stack(rotzs)

    corners_3d = rotzs @ boxes_vec_points # N, 3, 8
    corners_3d += c_xyz

    for box_id in range(bboxes_3d.shape[0]):
        single_boxes_vec_points = corners_3d[box_id]
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            pts = np.array([single_boxes_vec_points[:, i], single_boxes_vec_points[:, j]])
            l2 = gl.GLLinePlotItem(pos=pts, width=line_width, color=(0,250,0,255), antialias=True, mode='lines')
            box_items.append(l2)
            i, j = k + 4, (k + 1) % 4 + 4
            pts = np.array([single_boxes_vec_points[:, i], single_boxes_vec_points[:, j]])
            l2 = gl.GLLinePlotItem(pos=pts, width=line_width, color=(0,250,0,255), antialias=True, mode='lines')
            box_items.append(l2)

            i, j = k, k + 4
            pts = np.array([single_boxes_vec_points[:, i], single_boxes_vec_points[:, j]])
            l2 = gl.GLLinePlotItem(pos=pts, width=line_width, color=(0,250,0,255), antialias=True, mode='lines')
            box_items.append(l2)

        # pts = np.array([single_boxes_vec_points[:, 0], single_boxes_vec_points[:, -3]])
        # l1 = gl.GLLinePlotItem(pos=pts, width=3, color=(0,0,1,255), antialias=True, mode='lines')
        # box_items.append(l1)

        # pts = np.array([single_boxes_vec_points[:, -4], single_boxes_vec_points[:, 1]])
        # l1 = gl.GLLinePlotItem(pos=pts, width=3, color=(0,0,1,255), antialias=True, mode='lines')
        # box_items.append(l1)

    return box_items

class M3ED_BOX:
    def __init__(self, numpy_box):
        self.classIdx = numpy_box[-1]
        self.w = numpy_box[4]
        self.l = numpy_box[3]
        self.h = numpy_box[5]
        self.ry = numpy_box[-2]
        self.t = (numpy_box[0], numpy_box[1], numpy_box[2])

def show_image_with_boxes(img, objects, K, D, Rt):

    for obj in objects:
        obj = M3ED_BOX(obj)
        box3d_pts_2d, _ = utils.compute_box_3d(obj, K, D, Rt)
        if box3d_pts_2d is None:
            print("something wrong in the 3D box.")
            continue
        if obj.classIdx == 0:
            img = utils.draw_projected_box3d(img, box3d_pts_2d, color=(0, 255, 0))
        elif obj.classIdx == 1:
            img = utils.draw_projected_box3d(img, box3d_pts_2d, color=(255, 255, 0))
        elif obj.classIdx == 2:
            img = utils.draw_projected_box3d(img, box3d_pts_2d, color=(0, 255, 255))

    return img

def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def cart_to_hom(pts):
    """
    :param pts: (N, 3 or 2)
    :return pts_hom: (N, 4 or 3)
    """
    pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
    return pts_hom

def get_fov_flag(points, extristric, K, D, image_shape):
    extend_points = cart_to_hom(points[:,:3])
    points_cam = extend_points @ extristric.T
    # points_mask = points_cam[:,2] > 0
    # points_cam = points_cam[points_mask]

    rvecs = np.zeros((3,1))
    tvecs = np.zeros((3,1))

    pts_img, _ = cv2.projectPoints(points_cam[:,:3].astype(np.float32), rvecs, tvecs,
            K, D)

    imgpts = pts_img[:,0,:]

    imgpts[:, 1] = np.clip(imgpts[:, 1], 0, image_shape[1] - 1)
    imgpts[:, 0] = np.clip(imgpts[:, 0], 0, image_shape[0] - 1)
    return imgpts

def convert_box_3d_2d(img, boxes_3d, ex, K, D, label=None):
    cv2.imwrite('tools/ALTest/temp/temp.jpg', np.array(img))
    num_box = boxes_3d.shape[0]
    image_shape = (img.shape[1], img.shape[0])
    boxes_vec_points = np.zeros([num_box, 3, 8])
    c_xyz = boxes_3d[:,:3][:,:, None]
    l,w,h = boxes_3d[:,3], boxes_3d[:,4], boxes_3d[:,5]
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    boxes_vec_points[:, 0, :] = np.transpose(np.stack(x_corners))
    boxes_vec_points[:, 1, :] = np.transpose(np.stack(y_corners))
    boxes_vec_points[:, 2, :] = np.transpose(np.stack(z_corners))

    rotzs = []
    for box in boxes_3d:
        rotzs.append(rotz(box[6]))
    rotzs = np.stack(rotzs)

    corners_3d = rotzs @ boxes_vec_points # N, 3, 8
    corners_3d += c_xyz
    corners_3d = np.transpose(corners_3d, (0,2,1)).reshape(-1, 3)
    corners_2d = get_fov_flag(corners_3d, ex, K, D , image_shape).reshape(-1, 8, 2)
    corners_2d_x1 = np.min(corners_2d[...,0], axis=1)
    corners_2d_x2 = np.max(corners_2d[...,0], axis=1)
    corners_2d_y1 = np.min(corners_2d[...,1], axis=1)
    corners_2d_y2 = np.max(corners_2d[...,1], axis=1)

    corners_2d = np.stack([corners_2d_x1, corners_2d_x2, corners_2d_y1, corners_2d_y2])
    corners_2d = np.transpose(corners_2d)
    detections = sv.Detections(
        xyxy=corners_2d[:,[0,2,1,3]],
        class_id=corners_2d[:,0].astype(np.int64)
    )
    try:
        if label is None:
            box_idxes = [
                f"{box_id}" for box_id in range(num_box)]

            svimage = bounding_box_annotator.annotate(img, detections)
        else:
            svimage = bounding_box_annotator.annotate(img, detections, label)
        return np.array(svimage), corners_2d
    except:
        return None, corners_2d


