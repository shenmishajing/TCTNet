#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
SparseRCNN Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math
from typing import Optional, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.modeling.poolers import ROIPooler, cat
from detectron2.structures import Boxes

from .util.box_ops import box_cxcywh_to_xyxy

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class DynamicHead(nn.Module):

    def __init__(self, cfg, roi_input_shape):
        super().__init__()

        # Build RoI.
        box_pooler = self._init_box_pooler(cfg, roi_input_shape)
        self.box_pooler = box_pooler

        # Build heads.
        num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES
        d_model = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        dim_feedforward = cfg.MODEL.SparseRCNN.DIM_FEEDFORWARD
        nhead = cfg.MODEL.SparseRCNN.NHEADS
        dropout = cfg.MODEL.SparseRCNN.DROPOUT
        activation = cfg.MODEL.SparseRCNN.ACTIVATION
        num_heads = cfg.MODEL.SparseRCNN.NUM_HEADS
        rcnn_head = RCNNHead(cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation)
        self.head_series = _get_clones(rcnn_head, num_heads)
        self.return_intermediate = cfg.MODEL.SparseRCNN.DEEP_SUPERVISION

        # Init parameters.
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL
        self.num_classes = num_classes
        if self.use_focal:
            prior_prob = cfg.MODEL.SparseRCNN.PRIOR_PROB
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss.
            if self.use_focal:
                if p.shape[-1] == self.num_classes:
                    nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(cfg, input_shape):

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size = pooler_resolution,
            scales = pooler_scales,
            sampling_ratio = sampling_ratio,
            pooler_type = pooler_type,
        )
        return box_pooler

    def forward(self, features, init_bboxes, init_features):

        inter_class_logits = []
        inter_pred_bboxes = []

        bboxes = init_bboxes
        proposal_features = init_features

        for rcnn_head in self.head_series:
            class_logits, pred_bboxes, proposal_features = rcnn_head(features, bboxes, proposal_features, self.box_pooler)

            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()

        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes)

        return class_logits[None], pred_bboxes[None]


class RCNNHead(nn.Module):

    def __init__(self, cfg, d_model, num_classes, dim_feedforward = 2048, nhead = 8, dropout = 0.1, activation = "relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights = (2.0, 2.0, 1.0, 1.0)):
        super().__init__()

        self.d_model = d_model

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.relation_matrix = nn.Parameter(torch.ones(d_model, d_model))
        self.linear_relation = nn.Linear(2 * d_model, d_model)
        self.linear_interact = nn.Linear(2 * d_model, d_model)

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout = dropout)
        self.inst_interact = DynamicConv(cfg)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # cls.
        num_cls = cfg.MODEL.SparseRCNN.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace = True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = cfg.MODEL.SparseRCNN.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace = True))
        self.reg_module = nn.ModuleList(reg_module)

        # pred.
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL
        if self.use_focal:
            self.class_logits = nn.Linear(d_model, num_classes)
        else:
            self.class_logits = nn.Linear(d_model, num_classes + 1)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self, features, bboxes, pro_features, pooler):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        """

        N, nr_boxes = bboxes.shape[:2]

        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(bboxes[b]))
        roi_features = pooler(features, proposal_boxes)

        relation_feature = self.pool(roi_features).squeeze(dim = -1).squeeze(dim = -1)
        relation_feature2 = relation_feature.mm(self.relation_matrix).mm(relation_feature.T).mm(relation_feature)
        relation_feature = torch.cat((relation_feature, relation_feature2), dim = 1)
        relation_feature = self.dropout(self.linear_relation(relation_feature))

        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1)

        # self_att.
        pro_features = pro_features.permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value = pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.permute(1, 0, 2).reshape(N * nr_boxes, self.d_model)
        pro_features = self.dropout(
            self.linear_interact(torch.cat((pro_features, relation_feature), dim = 1)).reshape(1, N * nr_boxes, self.d_model))
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features).squeeze(dim = 0)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        cls_feature = obj_features.clone()
        reg_feature = obj_features.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))

        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features.reshape(N, nr_boxes, -1)

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max = self.scale_clamp)
        dh = torch.clamp(dh, max = self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes


class DynamicConv(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.SparseRCNN.DIM_DYNAMIC
        self.num_dynamic = cfg.MODEL.SparseRCNN.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace = True)

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (N * nr_boxes, self.d_model)
        roi_features: (N * nr_boxes, self.d_model, 49)
        '''
        features = roi_features.permute(0, 2, 1)
        parameters = self.dynamic_layer(pro_features)

        param1 = parameters[:, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


class ProposalHead(nn.Module):

    def __init__(self, cfg, roi_input_shape):
        super().__init__()

        self.k = cfg.MODEL.SparseRCNN.PROPOSAL.K
        self.hidden_dim = cfg.MODEL.SparseRCNN.HIDDEN_DIM

        # Build WH Proposals.
        self.init_proposal_boxes_wh = []
        for ratio in cfg.MODEL.SparseRCNN.PROPOSAL.WH_RATIO:
            for scale in cfg.MODEL.SparseRCNN.PROPOSAL.WH_SCALE:
                m = scale / math.sqrt(ratio)
                if abs(ratio - 1) < 1e-7:
                    self.init_proposal_boxes_wh.append([m, m])
                else:
                    self.init_proposal_boxes_wh.append([m, ratio * m])
                    self.init_proposal_boxes_wh.append([ratio * m, m])
        self.init_proposal_features = nn.Parameter(torch.Tensor(len(self.init_proposal_boxes_wh), self.hidden_dim))
        self.init_proposal_boxes_wh = nn.Parameter(torch.clamp(torch.tensor(self.init_proposal_boxes_wh), 0, 1))

        # Build heads.
        roi_in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        roi_feature_to_index = {feature: i for i, feature in enumerate(roi_in_features)}
        proposal_in_features = cfg.MODEL.SparseRCNN.PROPOSAL.IN_FEATURES
        proposal_index = [roi_feature_to_index[feature] for feature in proposal_in_features]
        self.in_feature = list(zip(proposal_index, proposal_in_features))
        self.objectness_heads = nn.ModuleDict(
            {in_feature: ObjectnessHead(roi_input_shape[in_feature].channels, self.k) for in_feature in proposal_in_features})

        # Init parameters.
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features, images_whwh):
        """
        :param features: List of Tensor(N, C, H, W) output of FPN
        :param images_whwh: Tensor(N, 4) output of FPN
        """

        xy = []
        proposal_objectness = []
        for index, in_feature in self.in_feature:
            p = self.objectness_heads[in_feature](features[index])
            proposal_objectness.append(p.squeeze(dim = 1))
            xy.append(self.objectness_heads[in_feature].get_xy(p))
        xy = torch.cat(xy, dim = 1)
        xy = xy[:, :, None, :].expand(-1, -1, self.init_proposal_boxes_wh.shape[0], -1)
        init_proposal_boxes_wh = self.init_proposal_boxes_wh
        init_proposal_boxes_wh = init_proposal_boxes_wh[None, None, :, :].expand(*xy.shape[:2], -1, -1)
        init_proposal_features = self.init_proposal_features[None, None, :, :].expand(*xy.shape[:2], -1, -1)
        proposal_boxes = torch.cat((xy, init_proposal_boxes_wh), dim = 3)
        N = proposal_boxes.shape[0]
        proposal_boxes = proposal_boxes.reshape((-1, 4))
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes.reshape((N, -1, 4))
        proposal_boxes *= images_whwh[:, None, :]
        return proposal_boxes, init_proposal_features.reshape((N, -1, self.hidden_dim)), proposal_objectness


class ObjectnessHead(nn.Module):

    def __init__(self, channel, k):
        super().__init__()

        self.channel = channel
        self.k = k

        # layers.
        self.conv = nn.Conv2d(self.channel, 1, 3, padding = 1)
        self.norm = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def get_xy(self, p: torch.Tensor):
        """
        :param p: (N, 1, H, W) or (N, H, W) output of self.forward
        :return xy: (N, k, 2) k xys
        """
        p = p.squeeze(dim = 1)
        N, h, w = p.shape
        p = p.reshape((N, -1))
        values, indices = p.topk(self.k, 1)
        xy = torch.stack((indices // w / float(h), indices % w / float(w)), dim = -1)
        return xy

    def forward(self, feature):
        """
        :param feature: (N, C, H, W) one scale feature of FPN output
        :return p: (N, 1, H, W) probability of object
        """

        feature = self.conv(feature)
        feature = self.norm(feature)
        p = self.sigmoid(feature)

        return p


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
