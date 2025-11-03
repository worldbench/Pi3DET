import os
import pickle
import numpy as np
from loguru import logger
from utils import common
from utils import gl_engine as gl
from pcdet.datasets import PI3DET_Dataset
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QFileDialog,\
                            QLabel, QLineEdit, QListWidget

class CustomListWidget(QListWidget):

    deleteKeyPressed = pyqtSignal()
    upKeyPressed = pyqtSignal()
    downKeyPressed = pyqtSignal()

    def __init__(self, parent=None):
        super(CustomListWidget, self).__init__(parent)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.deleteKeyPressed.emit()
        elif event.key() == Qt.Key_Up:
            self.upKeyPressed.emit()
        elif event.key() == Qt.Key_Down:
            self.downKeyPressed.emit()
        else:
            super(CustomListWidget, self).keyPressEvent(event)

class MainWindow(QWidget):
    def __init__(self) -> None:

        super(MainWindow, self).__init__()
        self.setWindowTitle("RoboSense Track 5")
        self.setGeometry(100, 100, 800, 600)
        self.sample_index = 0
        self.logger = logger
        self.init_window()

    def init_window(self):
        main_layout = QHBoxLayout()
        self.init_dataload_window()
        self.init_display_window()
        main_layout.addLayout(self.dataload_layout)
        main_layout.addLayout(self.display_layout)
        main_layout.setStretch(0, 2)
        main_layout.setStretch(1, 8)
        self.setLayout(main_layout)
        self.current_pixmap = None

    def init_dataload_window(self):
        self.dataload_layout = QVBoxLayout()
        self.split_select_cbox = QComboBox()
        self.split_select_cbox.activated.connect(self.split_selected)
        self.dataload_layout.addWidget(self.split_select_cbox)
        self.split_select_cbox.addItems([
            'training', 'validation'
        ])

        # load dataset button
        self.load_dataset_button = QPushButton('Load Dataset')
        self.dataload_layout.addWidget(self.load_dataset_button)
        self.load_dataset_button.clicked.connect(self.load_dataset)
        self.load_pred_pkl_button = QPushButton('Load Anno')
        self.dataload_layout.addWidget(self.load_pred_pkl_button)
        self.load_pred_pkl_button.clicked.connect(self.load_pred_pkl)
        # frame list
        self.frame_list = CustomListWidget()
        self.frame_list.itemClicked.connect(self.on_frame_selected)
        self.dataload_layout.addWidget(self.frame_list)
        self.frame_list.upKeyPressed.connect(self.frame_decrement_index)
        self.frame_list.downKeyPressed.connect(self.frame_increment_index)

    def init_display_window(self):
        self.display_layout = QVBoxLayout()
        temp_layout = QHBoxLayout()
        # image
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        temp_layout.addWidget(self.image_label)
        # lidar
        self.viewer = gl.AL_viewer()
        temp_layout.addWidget(self.viewer)
        temp_layout.setStretch(0, 5)
        temp_layout.setStretch(1, 5)
        self.display_layout.addLayout(temp_layout)
        # "Qlabel"
        h_layout = QHBoxLayout()
        self.sample_index_info = QLabel("")
        self.sample_index_info.setAlignment(Qt.AlignCenter)
        h_layout.addWidget(self.sample_index_info)
        self.goto_sample_index_box = QLineEdit(self)
        h_layout.addWidget(self.goto_sample_index_box)
        self.goto_sample_index_box.setPlaceholderText('')

        self.goto_sample_index_button = QPushButton('GoTo')
        h_layout.addWidget(self.goto_sample_index_button)
        self.goto_sample_index_button.clicked.connect(self.goto_sample_index)

        self.display_layout.addLayout(h_layout)
        h_layout = QHBoxLayout()

        # << button
        self.prev_view_button = QPushButton('<<<')
        h_layout.addWidget(self.prev_view_button)
        self.prev_view_button.clicked.connect(self.decrement_index)
        # >> button
        self.next_view_button = QPushButton('>>>')
        self.next_view_button.clicked.connect(self.increment_index)
        h_layout.addWidget(self.next_view_button)
        self.display_layout.addLayout(h_layout)

    def split_selected(self, index):
        item = self.split_select_cbox.itemText(index)
        self.dataset_split = item
        self.logger.info(f'Split selected: {item}')

    def update_frame_list(self):
        self.frame_list.clear()
        self.frame_list.addItems([f'frame_{i:07d}' for i in range(self.frames_count)])

    def load_dataset(self):
        pi3det_cfg_path = 'cfgs/dataset_configs/pi3det/vis_pi3det_dataset.yaml'
        dataset_cfg = common.load_cfg_from_yaml(pi3det_cfg_path)
        self.dataset = PI3DET_Dataset(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian'],
            training=self.dataset_split == 'training',
            root_path=dataset_cfg.DATA_PATH
        )
        self.frames_count = len(self.dataset)
        self.logger.info(f'Dataset loaded with {self.frames_count} samples.')
        self.update_frame_list()
        self.show_sample()

    def reset_viewer(self):
        self.sample_index_info.setText("")
        self.viewer.items = []

    def check_index_overflow(self) -> None:

        if self.sample_index == -1:
            self.sample_index = self.frames_count - 1

        if self.sample_index >= self.frames_count:
            self.sample_index = 0

        self.frame_list.setCurrentRow(self.sample_index)

    def increment_index(self):
        self.sample_index += 1
        self.check_index_overflow()
        self.show_sample()

    def decrement_index(self):
        self.sample_index -= 1
        self.check_index_overflow()
        self.show_sample()

    def frame_increment_index(self):
        self.increment_index()

    def frame_decrement_index(self):
        self.decrement_index()

    def goto_sample_index(self):
        try:
            index = int(self.goto_sample_index_box.text())
            self.sample_index = index
            self.check_index_overflow()
            self.show_sample()
        except:
            raise NotImplementedError('Check the index.')

    def on_frame_selected(self, item):
        self.sample_index = self.frame_list.row(item)
        self.show_sample()

    def extrac_pi3det_item(self):
        self.data_dict = self.dataset.__getitem__(self.sample_index)

    def show_image(self):
        image_path = os.path.join(self.dataset.root_path, self.dataset.data_infos[self.sample_index]['image_path'])
        img_w_box_2d = common.load_img_w_box3d(image_path)
        self.current_pixmap = gl.ndarray_to_pixmap(img_w_box_2d)
        self.update_image()

    def show_lidar(self):
        points = self.data_dict['points']
        mesh = gl.get_points_mesh(points, size=5.0)
        self.viewer.addItem(mesh)

    def show_boxes(self):
        if 'gt_boxes' in self.data_dict:
            boxes_3d = self.data_dict['gt_boxes']
            box_info = gl.create_boxes(bboxes_3d=boxes_3d)
            for box_item, l1_item, l2_item in zip(box_info['box_items'], box_info['l1_items'],\
                                                            box_info['l2_items']):
                self.viewer.addItem(box_item)
                self.viewer.addItem(l1_item)
                self.viewer.addItem(l2_item)

        if hasattr(self, 'predict_infos'):
            boxes_3d = self.predict_infos[self.sample_index]['boxes_lidar']
            labels = self.predict_infos[self.sample_index]['pred_labels'][:, None] + 2
            boxes_3d = np.hstack([boxes_3d, labels])
            box_info = gl.create_boxes(bboxes_3d=boxes_3d)
            for box_item, l1_item, l2_item in zip(box_info['box_items'], box_info['l1_items'],\
                                                            box_info['l2_items']):
                self.viewer.addItem(box_item)
                self.viewer.addItem(l1_item)
                self.viewer.addItem(l2_item)

    def resizeEvent(self, event):
        if self.current_pixmap:
            self.update_image()

    def update_image(self):
        if self.current_pixmap:
            pixmap = self.current_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)

    def load_pred_pkl(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "Select Anno File", "../", "All Files (*)", options=options)
        self.logger.info(f'Selected file: {file}')
        f = open(file, 'rb')
        anno_list = pickle.load(f)
        self.predict_infos = anno_list

    def show_sample(self):
        self.reset_viewer()
        self.extrac_pi3det_item()
        self.sample_index_info.setText(f'{self.sample_index} / {self.frames_count - 1}')
        # show image
        self.show_image()
        # show lidar
        self.show_lidar()
        # show boxes
        self.show_boxes()
