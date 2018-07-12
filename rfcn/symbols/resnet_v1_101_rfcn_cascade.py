# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yuwen Xiong, Xizhou Zhu
# --------------------------------------------------------

import cPickle
import mxnet as mx
from utils.symbol import Symbol
from operator_py.proposal import *
from operator_py.proposal_target import *
from operator_py.box_annotator_ohem import *
from operator_py.decode_bbox import *


class resnet_v1_101_rfcn_cascade(Symbol):

    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 512
        self.num_layers = 50       

    def residual_unit(self, data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):

        if bottle_neck:
            # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
            conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                    no_bias=True, workspace=workspace, name=name + '_conv1')
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
            act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
            conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                    no_bias=True, workspace=workspace, name=name + '_conv2')
            bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
            act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
            conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                    workspace=workspace, name=name + '_conv3')
            if dim_match:
                shortcut = data
            else:
                shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                                workspace=workspace, name=name+'_sc')
            if memonger:
                shortcut._set_attr(mirror_stage='True')
            return conv3 + shortcut
        else:
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
            conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                        no_bias=True, workspace=workspace, name=name + '_conv1')
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
            act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
            conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                        no_bias=True, workspace=workspace, name=name + '_conv2')
            if dim_match:
                shortcut = data
            else:
                shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                                workspace=workspace, name=name+'_sc')
            if memonger:
                shortcut._set_attr(mirror_stage='True')
            return conv2 + shortcut

    def resnet(self, data, units, num_stages, filter_list, bottle_neck=True, bn_mom=0.9, workspace=256, dtype='float32', memonger=False):

        num_unit = len(units)
        assert(num_unit == num_stages)
        if dtype == 'float32':
            data = mx.sym.identity(data=data, name='id')
        else:
            if dtype == 'float16':
                data = mx.sym.Cast(data=data, dtype=np.float16)
        data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')

        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                    no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

        stages = []
        for i in range(num_stages):
            body = self.residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                                name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                                memonger=memonger)
            for j in range(units[i]-1):
                body = self.residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                    bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
            stages.append(body)
        
        return stages[-2], stages[-1]

    def get_rpn(self, conv_feat, num_anchors):
        rpn_conv = mx.sym.Convolution(
            data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
        rpn_relu = mx.sym.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")
        return rpn_cls_score, rpn_bbox_pred

    def get_symbol(self, cfg, is_train=True):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        # input init
        if is_train:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")
            gt_boxes = mx.sym.Variable(name="gt_boxes")
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')
        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")

        if self.num_layers == 50:
            units = [3, 4, 6, 3]
        elif self.num_layers == 101:
            units = [3, 4, 23, 3]
        filter_list = [64, 256, 512, 1024, 2048]
        num_stages = 4
        bottle_neck = True

        # shared convolutional layers, res5
        conv_feat, relu1 = self.resnet(data = data,
                                units       = units,
                                num_stages  = num_stages,
                                filter_list = filter_list,
                                bottle_neck = bottle_neck,
                                workspace   = self.workspace,
                                dtype       = 'float32')

        rpn_cls_score, rpn_bbox_pred = self.get_rpn(conv_feat, num_anchors)

        if is_train:
            # prepare rpn data
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

            # classification
            rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                                   normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")
            # bounding box regression
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)

            # ROI proposal
            rpn_cls_act = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
            rpn_cls_act_reshape = mx.sym.Reshape(
                data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
            rois = mx.contrib.sym.Proposal(
                cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois_rpn',
                feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
        else:
            # ROI Proposal
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
            rpn_cls_prob = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
            rpn_cls_prob_reshape = mx.sym.Reshape(
                data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
            rois = mx.contrib.sym.Proposal(
                cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois_rpn',
                feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)

        gt_boxes = gt_boxes if is_train else None
        group = [rpn_cls_prob, rpn_bbox_loss] if is_train else []
        bbox_delta = None

        for stage in range(1):
            output = self.get_rfcn_head(stage+1, relu1, rois, gt_boxes, bbox_delta, im_info, 
                                        num_reg_classes, num_classes, is_train, cfg)
            rois = output[0]
            bbox_delta = output[1]
            if(is_train):
                group += output[2:]
            else:
                group += output
        if not is_train:
            rois_output = mx.sym.concat(*group[0::3], dim=0, name='rois')
            bbox_pred_output = mx.sym.concat(*group[1::3], dim=1, name='bbox_pred_reshape')
            cls_prob_output = mx.sym.concat(*group[2::3], dim=1, name='cls_prob_reshape')
            group += [rois_output, cls_prob_output, bbox_pred_output]

        group = mx.sym.Group(group)
        self.sym = group
        return group

    def get_rfcn_head(self, stage, relu1, rois, gt_boxes, bbox_delta, im_info,
                        num_reg_classes, num_classes, is_train, cfg):
        # cascade rfcn head

        assert stage in [1, 2, 3], 'Only support 1-3 heads.'

        with mx.name.Prefix('cascade%d_' % stage):
            if stage > 1:
                rois = mx.sym.Custom(rois=rois, bbox_pred=bbox_delta, im_info=im_info, name='rois', 
                                    num_reg_classes=num_reg_classes, op_type='decode_bbox')
            if is_train:
                # ROI proposal target
                gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
                rois, label, bbox_target, bbox_weight = mx.sym.Custom(rois=rois, gt_boxes=gt_boxes_reshape,
                                                                    op_type='proposal_target',
                                                                    num_classes=num_reg_classes,
                                                                    batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                    batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                    cfg=cPickle.dumps(cfg),
                                                                    fg_fraction=cfg.TRAIN.FG_FRACTION,
                                                                    cascade=stage)

            # conv_new_1
            conv_new_1 = mx.sym.Convolution(data=relu1, kernel=(1, 1), num_filter=1024, name="rfcn_conv_new", lr_mult=3.0)
            relu_new_1 = mx.sym.Activation(data=conv_new_1, act_type='relu', name='relu1')

            # rfcn_cls/rfcn_bbox
            rfcn_cls = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7*7*num_classes, name="rfcn_cls")
            rfcn_bbox = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7*7*4*num_reg_classes, name="rfcn_bbox")
            # ps roi pooling
            psroipooled_cls_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois, group_size=7, pooled_size=7,
                                                    output_dim=num_classes, spatial_scale=0.0625)
            psroipooled_loc_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois, group_size=7, pooled_size=7,
                                                    output_dim=8, spatial_scale=0.0625)
            cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
            bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
            cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
            bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))

            if is_train:
                cls_loss = mx.sym.SoftmaxOutput(name='cls_loss', data=cls_score, label=label, normalization='valid')
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)
                rcnn_label = label

                # reshape output
                rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
                cls_loss = mx.sym.Reshape(data=cls_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_loss_reshape')
                bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_loss_reshape')
                bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_delta_reshape')
                output = [rois, bbox_pred, cls_loss, bbox_loss, mx.sym.BlockGrad(rcnn_label)]
            else:
                cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
                cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                        name='cls_prob_reshape')
                bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_pred_reshape')
                output = [rois, bbox_pred, cls_prob]
        return output
            
    def init_weight_rpn(self, cfg, arg_params, aux_params):
        initializer = mx.initializer.Xavier(rnd_type='gaussian', magnitude=2)
        for name, shape in self.arg_shape_dict.items():
            if 'rpn' in name:
                arg_params[name] = mx.nd.zeros(shape=shape)
                initializer(mx.init.InitDesc(name), arg_params[name])

    def init_weight_rfcn(self, cfg, arg_params, aux_params):
        initializer = mx.initializer.Xavier(rnd_type='gaussian', magnitude=2)
        for name, shape in self.arg_shape_dict.items():
            if 'rfcn' in name:
                arg_params[name] = mx.nd.zeros(shape=shape)
                initializer(mx.init.InitDesc(name), arg_params[name])

    def init_weight(self, cfg, arg_params, aux_params):
        self.init_weight_rpn(cfg, arg_params, aux_params)
        self.init_weight_rfcn(cfg, arg_params, aux_params)
