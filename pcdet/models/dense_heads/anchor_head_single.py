import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .anchor_head_template import AnchorHeadTemplate

class AlignmentModule(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(AlignmentModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.branch3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
                        if in_channels != out_channels else None
        self.fc1 = nn.Linear(out_channels, out_channels // reduction)
        self.fc2 = nn.Linear(out_channels // reduction, out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out1 = self.relu(self.branch1(x))
        out2 = self.relu(self.branch3(x))
        out3 = self.relu(self.branch5(x))
        out = out1 + out2 + out3
        w = F.adaptive_avg_pool2d(out, output_size=1)
        w = w.view(w.size(0), -1)                     
        w = self.relu(self.fc1(w))                    
        w = torch.sigmoid(self.fc2(w))                
        w = w.view(w.size(0), w.size(1), 1, 1)        
        out = out * w                                
        if self.shortcut is not None:
            residual = self.shortcut(x)
        else:
            residual = x
        out = out + residual
        return self.relu(out)

class PositionEncodingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats), nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding

class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.norm = nn.BatchNorm1d(out_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.norm(x)
        x = self.activation(self.fc2(x))
        return x

class MLPUNet(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(MLPUNet, self).__init__()
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        prev_dim = input_dim
        for dim in hidden_dims:
            self.encoders.append(MLPBlock(prev_dim, dim))
            prev_dim = dim
        
        self.bottleneck = MLPBlock(hidden_dims[-1], hidden_dims[-1])
        
        for dim in reversed(hidden_dims[:-1]):
            self.decoders.append(MLPBlock(prev_dim * 2, dim))
            prev_dim = dim
        
        self.final_layer = nn.Linear(prev_dim * 2, input_dim)
    
    def forward(self, x):
        skip_connections = []
        
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
        
        x = self.bottleneck(x)
        for decoder in self.decoders:
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        x = torch.cat([x, skip_connections.pop()], dim=1)
        x = self.final_layer(x)
        return x

# class AlignmentModule(nn.Module):
#     def __init__(self, model_cfg, grid_size, input_channels):
#         super(AlignmentModule, self).__init__()
#         self.model_cfg = model_cfg
#         self.adp_max_pool = nn.AdaptiveMaxPool2d((1,1))
#         target_assigner_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
#         feature_map_size = grid_size[:2] // target_assigner_cfg[0]['feature_map_stride']
#         self.bev_pos = self.create_2D_grid(feature_map_size[0], feature_map_size[1])
#         self.self_posembed = PositionEncodingLearned(input_channel=2, num_pos_feats=input_channels)
#         self.pose_est_model = MLPUNet(input_channels, hidden_dims=[128,64,32])

#     def create_2D_grid(self, x_size, y_size):
#         meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
#         # NOTE: modified
#         batch_x, batch_y = torch.meshgrid(
#             *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
#         batch_x = batch_x + 0.5
#         batch_y = batch_y + 0.5
#         coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
#         coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
#         return coord_base
    
#     def forward(self, x):
#         B,C,H,W = x.shape
#         # scene geom feat
#         scene_gemo_feat = self.adp_max_pool(x)
#         scene_gemo_feat = self.pose_est_model(scene_gemo_feat.squeeze(-1).squeeze(-1))

#         pose_latent_feature = scene_gemo_feat.view(B,C,1,1).repeat(1,1,H,W)
#         bev_pos = self.bev_pos.repeat(B, 1, 1).to(x.device)
#         bev_position = self.self_posembed(bev_pos)
#         bev_position = bev_position.reshape(B,C,W,H).permute(0,1,3,2)
#         x = x + pose_latent_feature + bev_position
#         return x

class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

        # ALIGNMENT MODULE
        if self.model_cfg.get('ALIGNMENT', False):
            self.alignment_model = AlignmentModule(input_channels,input_channels,8)

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def get_loss(self, weights=None):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        cls_weight = box_weight = 1.0
        if weights is not None:
            cls_weight = weights[0]
            box_weight = weights[1]
        rpn_loss = cls_weight * cls_loss + box_weight * box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        if hasattr(self, 'alignment_model'):
            if data_dict['batch_mode'] == 'target':
                spatial_features_2d = self.alignment_model(spatial_features_2d)
        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict


class ActiveAnchorHeadSingle1(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        data_dict['bev_score'] = cls_preds.max(dim=1)[0].view(-1, 1, *cls_preds.shape[2:])
        data_dict['bev_map'] = cls_preds

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict


class AnchorHeadSingle_TQS(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)
        self.input_channels = input_channels
        self.margin_scale = self.model_cfg.get('MARGIN_SCALE', None)

        self._init_cls_layers()
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def _init_cls_layers(self):
        self.conv_cls = nn.Conv2d(
            self.input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_cls1 = nn.Conv2d(
            self.input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_cls2 = nn.Conv2d(
            self.input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.xavier_normal_(self.conv_cls1.weight)
        nn.init.xavier_uniform_(self.conv_cls2.weight)
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)


    def get_active_loss(self, mode=None):
        cls_loss, tb_dict = self.get_multi_cls_layer_loss()
        cls_loss_1, tb_dict_1 = self.get_multi_cls_layer_loss(head='cls_preds_1')
        tb_dict.update(tb_dict_1)
        cls_loss_2, tb_dict_2 = self.get_multi_cls_layer_loss(head='cls_preds_2')
        tb_dict.update(tb_dict_2)
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        if mode is 'train_detector':
            rpn_loss = cls_loss + box_loss + cls_loss_1 + cls_loss_2
        elif mode == 'train_mul_cls':
            rpn_loss = cls_loss_1 + cls_loss_2
        return rpn_loss, tb_dict

    def get_multi_cls_layer_loss(self, head=None):
        head = 'cls_preds' if head is None else head
        cls_preds = self.forward_ret_dict[head]
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        loss_name = 'rpn_loss_cls' + head.split('_')[-1] if head is not None else 'rpn_loss_cls'
        tb_dict = {
            loss_name: cls_loss.item()
        }
        return cls_loss, tb_dict

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        # multi-classifier
        cls_preds_1 = self.conv_cls1(spatial_features_2d)
        cls_preds_2 = self.conv_cls2(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        # multi-classifier
        cls_preds_1 = cls_preds_1.permute(0, 2, 3, 1).contiguous()
        cls_preds_2 = cls_preds_2.permute(0, 2, 3, 1).contiguous()

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        self.forward_ret_dict['cls_preds_1'] = cls_preds_1
        self.forward_ret_dict['cls_preds_2'] = cls_preds_2

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

    def committee_evaluate(self, data_dict):
        batch_size = self.forward_ret_dict['cls_preds_1'].shape[0]
        cls_preds_1 = self.forward_ret_dict['cls_preds_1']
        cls_preds_2 = self.forward_ret_dict['cls_preds_2']
        cls_preds_1 = cls_preds_1.view(batch_size, -1, self.num_class)  # (B, num_anchor, num_class)
        cls_preds_2 = cls_preds_2.view(batch_size, -1, self.num_class)  # (B, num_anchor, num_class)
        distances = torch.zeros((batch_size, 1))
        for i in range(batch_size):
            reweight_cls_1 = cls_preds_1[i]
            reweight_cls_2 = cls_preds_2[i]
            dis = (F.sigmoid(reweight_cls_1) - F.sigmoid(reweight_cls_2)).pow(2)  # (num_pos_anchor, num_class)
            dis = dis.mean(dim=-1).mean()
            distances[i] = dis
        self.forward_ret_dict['committee_evaluate'] = distances
        data_dict['committee_evaluate'] = distances
        return data_dict

    def uncertainty_evaluate(self, data_dict):
        batch_size = self.forward_ret_dict['cls_preds_1'].shape[0]
        cls_preds_1 = self.forward_ret_dict['cls_preds_1'].view(batch_size, -1, self.num_class)
        cls_preds_2 = self.forward_ret_dict['cls_preds_2'].view(batch_size, -1, self.num_class)
        uncertainty = torch.zeros((batch_size, 1))
        for i in range(batch_size):
            reweight_cls_1 = cls_preds_1[i].view(-1, 1)
            reweight_cls_2 = cls_preds_2[i].view(-1, 1)
            reweight_cls_1 = F.sigmoid(reweight_cls_1)
            reweight_cls_2 = F.sigmoid(reweight_cls_2)
            uncertainty_cls_1 = torch.min(torch.cat([torch.ones_like(reweight_cls_1) - reweight_cls_1, reweight_cls_1 - torch.zeros_like(reweight_cls_1)], dim=1)).view(-1).mean()
            uncertainty_cls_2 = torch.min(torch.cat([torch.ones_like(reweight_cls_2) - reweight_cls_2, reweight_cls_2 - torch.zeros_like(reweight_cls_2)], dim=1)).view(-1).mean()
            uncertainty[i] = (uncertainty_cls_1 + uncertainty_cls_2) / 2
        data_dict['uncertainty_evaluate'] = uncertainty
        return data_dict
