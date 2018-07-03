"""
DecodeBBox Operator transform anchor coordinates into ROI coordinates with prediction results on
classification probability and bounding box prediction results by rcnn head, 
and image size and scale information.

Cascade RCNN implementation
"""

import mxnet as mx
import numpy as np
import numpy.random as npr
from distutils.util import strtobool

from bbox.bbox_transform import bbox_pred, clip_boxes
from rpn.generate_anchor import generate_anchors
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

DEBUG = False

class DecodeBBoxOperator(mx.operator.CustomOp):
    def __init__(self):
        super(DecodeBBoxOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        batch_size = in_data[1].shape[0]
        if batch_size > 1:
            raise ValueError("Sorry, multiple images each device is not implemented")

        # the first set of anchors are background probabilities
        # keep the second part
        rois = in_data[0].asnumpy()[:, 1:]
        bbox_deltas = in_data[1].asnumpy()[0, :, 4:8]
        im_info = in_data[2].asnumpy()[0, :]


        # Convert anchors into proposals via bbox transformations
        proposals = bbox_pred(rois, bbox_deltas)

        # clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        # Output rois array
        # implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        self.assign(out_data[0], req[0], blob)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)

@mx.operator.register("decode_bbox")
class ProposalProp(mx.operator.CustomOpProp):
    def __init__(self, num_reg_classes):
        super(ProposalProp, self).__init__(need_top_grad=False)
        self._num_reg_classes = int(num_reg_classes)
        assert self._num_reg_classes == 2, 'only support CLASS_AGNOSTIC'

    def list_arguments(self):
        return ['rois', 'bbox_pred', 'im_info']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        rois_shape = in_shape[0]
        bbox_pred_shape = in_shape[1]
        assert rois_shape[0] == bbox_pred_shape[1], 'ROI number does not equal in rois, cls and reg'

        batch_size = bbox_pred_shape[0]
        im_info_shape = (batch_size, 3)
        output_shape = (rois_shape[0], 5)

        return [rois_shape, bbox_pred_shape, im_info_shape], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return DecodeBBoxOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
