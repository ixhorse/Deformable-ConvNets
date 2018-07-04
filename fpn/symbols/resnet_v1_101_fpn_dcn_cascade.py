# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Haozhi Qi
# --------------------------------------------------------

import cPickle
import mxnet as mx
from utils.symbol import Symbol
from operator_py.pyramid_proposal import *
from operator_py.proposal_target import *
from operator_py.fpn_roi_pooling import *
from operator_py.box_annotator_ohem import *
from operator_py.decode_bbox import *


class resnet_v1_101_fpn_dcn_cascade(Symbol):
    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        shared_param_dict = ['cascade%d_offset_p%d' %(c+1, x) for x in range(2,5) for c in range(2)]
        # self.shared_param_list = ['offset_p2', 'offset_p3', 'offset_p4', 'offset_p5',
        #                           'rpn_conv', 'rpn_cls_score', 'rpn_bbox_pred']
        self.shared_param_list = shared_param_dict + ['rpn_conv', 'rpn_cls_score', 'rpn_bbox_pred']
        self.shared_param_dict = {}
        for name in self.shared_param_list:
            self.shared_param_dict[name + '_weight'] = mx.sym.Variable(name + '_weight')
            self.shared_param_dict[name + '_bias'] = mx.sym.Variable(name + '_bias')

    def residual_unit(self, data, num_filter, stride, dim_match, name, bottle_neck=True, num_group=32, bn_mom=0.9, workspace=256, 
                        memonger=False, with_dpyramid=False):

        if bottle_neck:
            # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
            conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                    no_bias=True, workspace=workspace, name=name + '_conv1')
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
            act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
            if with_dpyramid:
                conv2_offset = mx.sym.Convolution(data=act1, num_filter=72, pad=(1,1), kernel=(3,3), stride=stride, name=name + '_conv2_offset')
                conv2 = mx.contrib.symbol.DeformableConvolution(data=act2, offset=conv2_offset, num_filter=int(num_filter*0.25), 
                                                                num_deformable_group=4, pad=(1, 1), kernel=(3, 3), stride=stride, no_bias=True, 
                                                                name=name + '_conv2')
            else:
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
            
    def get_resnet_backbone(self, data, units, num_stages, filter_list, num_group, bottle_neck=True, bn_mom=0.9, 
                            workspace=256, dtype='float32', memonger=False):
        
        num_unit = len(units)
        assert(num_unit == num_stages)
        data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')

        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                    no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

        stages = []
        for i in range(num_stages):
            body = self.residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                                name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, num_group=num_group,
                                bn_mom=bn_mom, workspace=workspace, memonger=memonger, with_dpyramid=False)
            for j in range(units[i]-1):
                if(i == 0):
                    with_dpyramid = False
                elif(j == units[i]-2):
                    with_dpyramid = True
                else:
                    with_dpyramid = False
                body = self.residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                    bottle_neck=bottle_neck, num_group=num_group, bn_mom=bn_mom, workspace=workspace, 
                                    memonger=memonger, with_dpyramid=with_dpyramid)
            stages.append(body)
        
        return stages

    def get_fpn_feature(self, c2, c3, c4, feature_dim=256):

        # lateral connection
        # fpn_p5_1x1 = mx.symbol.Convolution(data=c5, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=feature_dim, name='fpn_p5_1x1')
        fpn_p4_1x1 = mx.symbol.Convolution(data=c4, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=feature_dim, name='fpn_p4_1x1')
        fpn_p3_1x1 = mx.symbol.Convolution(data=c3, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=feature_dim, name='fpn_p3_1x1')
        fpn_p2_1x1 = mx.symbol.Convolution(data=c2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=feature_dim, name='fpn_p2_1x1')
        # top-down connection
        # fpn_p5_upsample = mx.symbol.UpSampling(fpn_p5_1x1, scale=2, sample_type='bilinear', num_filter=feature_dim, name='upsampling_fpn_p5')
        # fpn_p4_plus = mx.sym.ElementWiseSum(*[fpn_p5_upsample, fpn_p4_1x1], name='fpn_p4_sum')
        fpn_p4_upsample = mx.symbol.UpSampling(fpn_p4_1x1, scale=2, sample_type='bilinear', num_filter=feature_dim, name='upsampling_fpn_p4')
        fpn_p3_plus = mx.sym.ElementWiseSum(*[fpn_p4_upsample, fpn_p3_1x1], name='fpn_p3_sum')
        fpn_p3_upsample = mx.symbol.UpSampling(fpn_p3_plus, scale=2, sample_type='bilinear', num_filter=feature_dim, name='upsampling_fpn_p3')
        fpn_p2_plus = mx.sym.ElementWiseSum(*[fpn_p3_upsample, fpn_p2_1x1], name='fpn_p2_sum')
        # FPN feature
        # fpn_p6 = mx.sym.Convolution(data=c5, kernel=(3, 3), pad=(1, 1), stride=(2, 2), num_filter=feature_dim, name='fpn_p6')
        # fpn_p5 = mx.symbol.Convolution(data=fpn_p5_1x1, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=feature_dim, name='fpn_p5')
        fpn_p4 = mx.symbol.Convolution(data=fpn_p4_1x1, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=feature_dim, name='fpn_p4')
        fpn_p3 = mx.symbol.Convolution(data=fpn_p3_plus, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=feature_dim, name='fpn_p3')
        fpn_p2 = mx.symbol.Convolution(data=fpn_p2_plus, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=feature_dim, name='fpn_p2')
        return fpn_p2, fpn_p3, fpn_p4

    def get_rpn_subnet(self, data, num_anchors, suffix):
        rpn_conv = mx.sym.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=512, name='rpn_conv_' + suffix,
                                      weight=self.shared_param_dict['rpn_conv_weight'], bias=self.shared_param_dict['rpn_conv_bias'])
        rpn_relu = mx.sym.Activation(data=rpn_conv, act_type='relu', name='rpn_relu_' + suffix)
        rpn_cls_score = mx.sym.Convolution(data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name='rpn_cls_score_' + suffix,
                                           weight=self.shared_param_dict['rpn_cls_score_weight'], bias=self.shared_param_dict['rpn_cls_score_bias'])
        rpn_bbox_pred = mx.sym.Convolution(data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name='rpn_bbox_pred_' + suffix,
                                           weight=self.shared_param_dict['rpn_bbox_pred_weight'], bias=self.shared_param_dict['rpn_bbox_pred_bias'])

        # n x (2*A) x H x W => n x 2 x (A*H*W)
        rpn_cls_score_t1 = mx.sym.Reshape(data=rpn_cls_score, shape=(0, 2, -1, 0), name='rpn_cls_score_t1_' + suffix)
        rpn_cls_score_t2 = mx.sym.Reshape(data=rpn_cls_score_t1, shape=(0, 2, -1), name='rpn_cls_score_t2_' + suffix)
        rpn_cls_prob = mx.sym.SoftmaxActivation(data=rpn_cls_score_t1, mode='channel', name='rpn_cls_prob_' + suffix)
        rpn_cls_prob_t = mx.sym.Reshape(data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_t_' + suffix)
        rpn_bbox_pred_t = mx.sym.Reshape(data=rpn_bbox_pred, shape=(0, 0, -1), name='rpn_bbox_pred_t_' + suffix)
        return rpn_cls_score_t2, rpn_cls_prob_t, rpn_bbox_pred_t, rpn_bbox_pred

    def get_symbol(self, cfg, is_train=True):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)

        data = mx.sym.Variable(name="data")
        im_info = mx.sym.Variable(name="im_info")

        # shared convolutional layers
        num_group=1
        filter_list = [64, 256, 512, 1024]
        bottle_neck = True
        num_stages = 3
        units = [3, 4, 23]
        res2, res3, res4 = self.get_resnet_backbone(data        = data,
                                                    units       = units,
                                                    num_stages  = num_stages,
                                                    filter_list = filter_list,
                                                    num_group   = num_group,
                                                    bottle_neck = bottle_neck,
                                                    memonger    = True)
        fpn_p2, fpn_p3, fpn_p4 = self.get_fpn_feature(res2, res3, res4)

        rpn_cls_score_p2, rpn_prob_p2, rpn_bbox_loss_p2, rpn_bbox_pred_p2 = self.get_rpn_subnet(fpn_p2, cfg.network.NUM_ANCHORS, 'p2')
        rpn_cls_score_p3, rpn_prob_p3, rpn_bbox_loss_p3, rpn_bbox_pred_p3 = self.get_rpn_subnet(fpn_p3, cfg.network.NUM_ANCHORS, 'p3')
        rpn_cls_score_p4, rpn_prob_p4, rpn_bbox_loss_p4, rpn_bbox_pred_p4 = self.get_rpn_subnet(fpn_p4, cfg.network.NUM_ANCHORS, 'p4')
        # rpn_cls_score_p5, rpn_prob_p5, rpn_bbox_loss_p5, rpn_bbox_pred_p5 = self.get_rpn_subnet(fpn_p5, cfg.network.NUM_ANCHORS, 'p5')
        # rpn_cls_score_p6, rpn_prob_p6, rpn_bbox_loss_p6, rpn_bbox_pred_p6 = self.get_rpn_subnet(fpn_p6, cfg.network.NUM_ANCHORS, 'p6')

        rpn_cls_prob_dict = {
            # 'rpn_cls_prob_stride64': rpn_prob_p6,
            # 'rpn_cls_prob_stride32': rpn_prob_p5,
            'rpn_cls_prob_stride16': rpn_prob_p4,
            'rpn_cls_prob_stride8': rpn_prob_p3,
            'rpn_cls_prob_stride4': rpn_prob_p2,
        }
        rpn_bbox_pred_dict = {
            # 'rpn_bbox_pred_stride64': rpn_bbox_pred_p6,
            # 'rpn_bbox_pred_stride32': rpn_bbox_pred_p5,
            'rpn_bbox_pred_stride16': rpn_bbox_pred_p4,
            'rpn_bbox_pred_stride8': rpn_bbox_pred_p3,
            'rpn_bbox_pred_stride4': rpn_bbox_pred_p2,
        }
        arg_dict = dict(rpn_cls_prob_dict.items() + rpn_bbox_pred_dict.items())

        if is_train:
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')
            gt_boxes = mx.sym.Variable(name="gt_boxes")

            rpn_cls_score = mx.sym.Concat(rpn_cls_score_p2, rpn_cls_score_p3, rpn_cls_score_p4, dim=2)
            rpn_bbox_loss = mx.sym.Concat(rpn_bbox_loss_p2, rpn_bbox_loss_p3, rpn_bbox_loss_p4, dim=2)
            # RPN classification loss
            rpn_cls_output = mx.sym.SoftmaxOutput(data=rpn_cls_score, label=rpn_label, multi_output=True, normalization='valid',
                                                  use_ignore=True, ignore_label=-1, name='rpn_cls_prob')
            # bounding box regression
            rpn_bbox_loss = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_l1', scalar=3.0, data=(rpn_bbox_loss - rpn_bbox_target))
            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss, grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)

            aux_dict = {
                'op_type': 'pyramid_proposal', 'name': 'rois',
                'im_info': im_info, 'feat_stride': tuple(cfg.network.RPN_FEAT_STRIDE),
                'scales': tuple(cfg.network.ANCHOR_SCALES), 'ratios': tuple(cfg.network.ANCHOR_RATIOS),
                'rpn_pre_nms_top_n': cfg.TRAIN.RPN_PRE_NMS_TOP_N, 'rpn_post_nms_top_n': cfg.TRAIN.RPN_POST_NMS_TOP_N,
                'threshold': cfg.TRAIN.RPN_NMS_THRESH, 'rpn_min_size': cfg.TRAIN.RPN_MIN_SIZE
            }

            # ROI proposal
            rois = mx.sym.Custom(**dict(arg_dict.items() + aux_dict.items()))
        else:
            aux_dict = {
                'op_type': 'pyramid_proposal', 'name': 'rois',
                'im_info': im_info, 'feat_stride': tuple(cfg.network.RPN_FEAT_STRIDE),
                'scales': tuple(cfg.network.ANCHOR_SCALES), 'ratios': tuple(cfg.network.ANCHOR_RATIOS),
                'rpn_pre_nms_top_n': cfg.TEST.RPN_PRE_NMS_TOP_N, 'rpn_post_nms_top_n': cfg.TEST.RPN_POST_NMS_TOP_N,
                'threshold': cfg.TEST.RPN_NMS_THRESH, 'rpn_min_size': cfg.TEST.RPN_MIN_SIZE
            }
            # ROI proposal
            rois = mx.sym.Custom(**dict(arg_dict.items() + aux_dict.items()))

        gt_boxes = gt_boxes if is_train else None
        group = [rpn_cls_output, rpn_bbox_loss] if is_train else []
        bbox_delta = None

        for stage in range(cfg.network.cascade_stage):
            output = self.get_rcnn_head(stage+1, rois, [fpn_p2, fpn_p3, fpn_p4], gt_boxes, bbox_delta, im_info, 
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

    def get_rcnn_head(self, stage, rois, fpn_feat, gt_boxes, bbox_delta, im_info,
                        num_reg_classes, num_classes, is_train, cfg):
        # cascade rcnn head

        assert stage in [1, 2, 3], 'Only support 1-3 heads.'

        with mx.name.Prefix('cascade%d_' % stage):
            if stage > 1:
                rois = mx.sym.Custom(rois=rois, bbox_pred=bbox_delta, im_info=im_info, name='rois', 
                                    num_reg_classes=num_reg_classes, op_type='decode_bbox')
            if is_train:
                # ROI proposal target
                gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
                rois, label, bbox_target, bbox_weight = mx.sym.Custom(rois=rois, gt_boxes=gt_boxes_reshape, 
                                                                    op_type='proposal_target', num_classes=num_reg_classes, 
                                                                    batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                    batch_rois=cfg.TRAIN.BATCH_ROIS, 
                                                                    cfg=cPickle.dumps(cfg), 
                                                                    fg_fraction=cfg.TRAIN.FG_FRACTION,
                                                                    cascade=stage)
            # deformable roi pooling
            prefix = 'cascade%d_' % stage
            offset_p2_weight = mx.sym.Variable(name=prefix+'offset_p2_weight', dtype=np.float32, lr_mult=0.01)
            offset_p3_weight = mx.sym.Variable(name=prefix+'offset_p3_weight', dtype=np.float32, lr_mult=0.01)
            offset_p4_weight = mx.sym.Variable(name=prefix+'offset_p4_weight', dtype=np.float32, lr_mult=0.01)
            offset_p5_weight = mx.sym.Variable(name=prefix+'offset_p5_weight', dtype=np.float32, lr_mult=0.01)
            offset_p2_bias = mx.sym.Variable(name=prefix+'offset_p2_bias', dtype=np.float32, lr_mult=0.01)
            offset_p3_bias = mx.sym.Variable(name=prefix+'offset_p3_bias', dtype=np.float32, lr_mult=0.01)
            offset_p4_bias = mx.sym.Variable(name=prefix+'offset_p4_bias', dtype=np.float32, lr_mult=0.01)
            offset_p5_bias = mx.sym.Variable(name=prefix+'offset_p5_bias', dtype=np.float32, lr_mult=0.01)

            roi_pool = mx.symbol.Custom(data_p2=fpn_feat[0], data_p3=fpn_feat[1], data_p4=fpn_feat[2],
                                        offset_weight_p2=offset_p2_weight, offset_bias_p2=offset_p2_bias,
                                        offset_weight_p3=offset_p3_weight, offset_bias_p3=offset_p3_bias,
                                        offset_weight_p4=offset_p4_weight, offset_bias_p4=offset_p4_bias,
                                        # offset_weight_p5=offset_p5_weight, offset_bias_p5=offset_p5_bias,
                                        rois=rois, op_type='fpn_roi_pooling', name='fpn_roi_pooling', 
                                        feat_strides=(4,8,16), with_deformable=True)
            # 2 fc
            fc_new_1 = mx.symbol.FullyConnected(name='fc_new_1', data=roi_pool, num_hidden=1024)
            fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu')

            fc_new_2 = mx.symbol.FullyConnected(name='fc_new_2', data=fc_new_1_relu, num_hidden=1024)
            fc_new_2_relu = mx.sym.Activation(data=fc_new_2, act_type='relu', name='fc_new_2_relu')

            # cls_score/bbox_pred
            cls_score = mx.symbol.FullyConnected(name='cls_score', data=fc_new_2_relu, num_hidden=num_classes)
            bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=fc_new_2_relu, num_hidden=num_reg_classes * 4)

            if is_train:
                if cfg.TRAIN.ENABLE_OHEM:
                    labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                                num_reg_classes=num_reg_classes, roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                                cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                                bbox_targets=bbox_target, bbox_weights=bbox_weight)
                    cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem, normalization='valid', use_ignore=True, ignore_label=-1)
                    bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
                    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
                    rcnn_label = labels_ohem
                else:
                    cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid')
                    bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
                    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)
                    rcnn_label = label

                # reshape output
                rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
                cls_loss = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
                bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_loss_reshape')
                bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_delta_reshape')
                output = [rois, bbox_pred, cls_loss, bbox_loss, mx.sym.BlockGrad(rcnn_label)]
            else:
                cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
                cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
                bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_pred_reshape')
                output = [rois, bbox_pred, cls_prob]
        return output

    def init_weight_rcnn(self, cfg, arg_params, aux_params):
        for i in range(cfg.network.cascade_stage):
            prefix = 'cascade%d_' % (i+1)
            arg_params[prefix+'fc_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[prefix+'fc_new_1_weight'])
            arg_params[prefix+'fc_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict[prefix+'fc_new_1_bias'])
            arg_params[prefix+'fc_new_2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[prefix+'fc_new_2_weight'])
            arg_params[prefix+'fc_new_2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict[prefix+'fc_new_2_bias'])
            arg_params[prefix+'cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[prefix+'cls_score_weight'])
            arg_params[prefix+'cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict[prefix+'cls_score_bias'])
            arg_params[prefix+'bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[prefix+'bbox_pred_weight'])
            arg_params[prefix+'bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict[prefix+'bbox_pred_bias'])

    def init_deformable_convnet(self, cfg, arg_params, aux_params):
        arg_params['stage2_unit4_conv2_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage2_unit4_conv2_offset_weight'])
        arg_params['stage2_unit4_conv2_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage2_unit4_conv2_offset_bias'])
        arg_params['stage3_unit23_conv2_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage3_unit23_conv2_offset_weight'])
        arg_params['stage3_unit23_conv2_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage3_unit23_conv2_offset_bias'])

    def init_weight_fpn(self, cfg, arg_params, aux_params):
        initializer = mx.initializer.Xavier(rnd_type='gaussian', magnitude=2)
        for name, shape in self.arg_shape_dict.items():
            if 'fpn' in name:
                arg_params[name] = mx.nd.zeros(shape=shape)
                initializer(name, arg_params[name])
        for name, shape in self.aux_shape_dict.items():
            if 'fpn' in name:
                aux_params[name] = mx.nd.zeros(shape=shape)
                initializer(name, aux_params[name])

    def init_weight(self, cfg, arg_params, aux_params):
        for name in self.shared_param_list:
            if 'offset' in name:
                arg_params[name + '_weight'] = mx.nd.zeros(shape=self.arg_shape_dict[name + '_weight'])
            else:
                arg_params[name + '_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[name + '_weight'])
            arg_params[name + '_bias'] = mx.nd.zeros(shape=self.arg_shape_dict[name + '_bias'])
        self.init_deformable_convnet(cfg, arg_params, aux_params)
        self.init_weight_rcnn(cfg, arg_params, aux_params)
        self.init_weight_fpn(cfg, arg_params, aux_params)