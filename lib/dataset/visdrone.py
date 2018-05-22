# --------------------------------------------------------
# Deformable Convolutional Networks
# chenyf
# 2018/5/8
# --------------------------------------------------------

"""
VisDrone database
This class loads ground truth notations from anoations
and transform them into IMDB format. 
"""

import cPickle
import cv2
import os
import numpy as np
import PIL
from imdb import IMDB

class VisDrone(IMDB):
    def __init__(self, image_set, root_path, devkit_path, result_path=None, mask_size=-1, binary_thresh=None):
        """
        fill basic information to initialize imdb
        :param image_set: 2007_trainval, 2007_test, etc
        :param root_path: 'cache'
        :param devkit_path: data and results
        :return: imdb object
        """
        super(VisDrone, self).__init__('visdrone', image_set, root_path, devkit_path, result_path)  # set self.name

        self.root_path = root_path
        self.devkit_path = devkit_path
        self.data_path = os.path.join(devkit_path, 'VisDrone2018-DET-' + image_set)

        self.classes = ['__background__',  # always index 0
                        'pedestrian', 'people', 'bicycle', 'car',
                        'van', 'truck', 'tricycle', 'awning-tricycle',
                        'bus', 'motor']
        self.num_classes = len(self.classes)
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        print 'num_images', self.num_images
        self.mask_size = mask_size
        self.binary_thresh = binary_thresh

        self.config = {'min_size': 2}

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self.data_path, 'index' + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.data_path, 'images', index + '.jpg')
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self.load_visdrone_annotation(index) for index in self.image_set_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def load_visdrone_annotation(self, index):
        """
        for a given index, load image and bounding boxes info from XML file
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)

        filename = os.path.join(self.data_path, 'annotations', index + '.txt')
        im = PIL.Image.open(roi_rec['image'])
        w, h = im.size
        roi_rec['height'] = float(h)
        roi_rec['width'] = float(w)
        #im_size = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION).shape
        #assert im_size[0] == roi_rec['height'] and im_size[1] == roi_rec['width']
        with open(filename, 'r') as f:
            objs = [x.strip().split(',') for x in f.readlines()]
        
        # ignored class and others class
        filter_list = []
        for i in range(len(objs)):
            obj = objs[i]
            if(obj[5] == '0' or obj[5] == '11'):
                filter_list.append(i)
        for i in range(len(filter_list)-1, -1, -1):
            objs.pop(filter_list[i])
        
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        truncation = np.zeros((num_objs), dtype=np.int32)
        occlusion = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            x1 = float(obj[0])
            y1 = float(obj[1])
            x2 = x1 + float(obj[2])
            y2 = y1 + float(obj[3])
            cls = int(obj[5])
            # cls = 0 if cls == 11 else cls
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            truncation[ix] = int(obj[6])
            occlusion[ix] = int(obj[7])

        roi_rec.update({'boxes': boxes,
                        'gt_classes': gt_classes,
                        'gt_overlaps': overlaps,
                        'max_classes': overlaps.argmax(axis=1),
                        'max_overlaps': overlaps.max(axis=1),
                        'truncation':truncation,
                        'occlusion':occlusion,
                        'flipped': False})
        return roi_rec

    def write_result(self, detections):
        result_dir = os.path.join(self.result_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        
        for img_ind, index in enumerate(self.image_set_index):
            with open(os.path.join(result_dir, index+'.txt'), 'w') as f:
                for cls, cls_result in enumerate(detections):
                    if len(cls_result[img_ind]) == 0:
                        continue
                    for box in cls_result[img_ind]:
                        box[2] = box[2] - box[0]
                        box[3] = box[3] - box[1]
                        f.write('{:d},{:d},{:d},{:d},{:.4f},{:d},-1,-1\n'.
                                format(int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4], cls))
        print '...write result files success.\n'