# coding: utf-8
import sys

import mxnet
import os
import numpy
import cv2
import time
from math import floor
import numpy as np
from collections import namedtuple
import random
from PIL import Image, ImageDraw, ImageFont
import config as cfg
from util import AlgorithmBase, PointsRecorder


import datetime
import uuid
from tensor_stream import TensorStreamConverter, FourCC
import queue
import threading

sys.path.insert(0, '/home/hc/packages/mxnet/python')


class Persondec(object):
    def __init__(self):

        self.symbolFile = cfg.phone_symbol_1
        self.modelFile = cfg.phone_model_1
        self.RF = [71.0, 111.0, 143.0, 159.0]
        self.map = [83, 41, 41, 41]
        self.stride = [4, 8, 8, 8]
        self.bbox_small = [15, 30, 60, 95]
        self.bbox_large = [30, 60, 95, 140]
        self.center_area_start = [3, 7, 7, 7]
        self.num_scale = 4
        self.constant = [i/2.0 for i in self.RF]
        self.ratio = 4

        self.ctx = mxnet.gpu(0)
        self.score_threshold = 0.2
        self.top_k = 100
        self.NMS_threshold = 0.25
        self.interval = None
        # load symbol and parameters
        if not os.path.exists(self.symbolFile):
            #print('The symbol1 file does not exist!!!!')
            quit()
        if not os.path.exists(self.modelFile) :
            #print('The model file does not exist!!!!')
            quit()
        self.symbolNet = mxnet.symbol.load(self.symbolFile)
        self.arg_name_arrays = {}
        self.aux_name_arrays = {}
        saved_dict = mxnet.nd.load(self.modelFile)
        for k, v in saved_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                self.arg_name_arrays.update({name: v.as_in_context(self.ctx)})
            if tp == 'aux':
                self.aux_name_arrays.update({name: v.as_in_context(self.ctx)})

        self.w, self.h = 0, 0
        self.opencv_version = int(cv2.__version__.split('.')[0])

    def NMS_fast(self, boxes, overlapThresh):
        if  boxes.shape[0]== 0:#len 可以 返回矩阵的行数
            return numpy.array([])

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype != numpy.float32:
            boxes = boxes.astype(numpy.float32)

        # initialize the list of picked indexes
        pick = []
        # grab the coordinates of the bounding boxes
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        sc = boxes[:,4]
        widths = x2-x1
        heights = y2-y1


        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = heights * widths
        idxs = numpy.argsort(sc)# 从小到大排序


        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # compare secend highest score boxes
            xx1 = numpy.maximum(x1[i], x1[idxs[:last]])
            yy1 = numpy.maximum(y1[i], y1[idxs[:last]])
            xx2 = numpy.minimum(x2[i], x2[idxs[:last]])
            yy2 = numpy.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bo（ box
            w = numpy.maximum(0, xx2 - xx1 + 1)
            h = numpy.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have
            idxs =  numpy.delete(idxs, numpy.concatenate(([last], numpy.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick]

    def _bindModel(self, height, width):
        if self.h != height or self.w != width:
            self.arg_name_arrays['data'] = mxnet.nd.zeros((1, 3, height, width), self.ctx)
            self.executor = self.symbolNet.bind(ctx=self.ctx, \
                                                args=self.arg_name_arrays, \
                                                grad_req='null', \
                                                aux_states=self.aux_name_arrays)
            self.h,self.w = height, width

    # 这个函数用于某个像素范围内的人脸，需要使用这提前估计出单个人脸在输入图像中的像素大小范围
    # interval: [bound_l, bound_r] 例如：[40, 100] 表示希望检测出输入图像中像素范围在40-100之间的人脸
    def detect_interval(self, image):

        if image.ndim != 3 or image.shape[2] != 3:
            #print('Only RGB images are supported.')
            return None

        # 如果不指定interval参数，默认使用网络可以检测的范围
        if self.interval is None:
            self.interval = [min(self.bbox_small), max(self.bbox_large)]

        if self.interval[0]<15 or self.interval[1]<self.interval[0]:
            #print('The algorithm only detects faces larger than 20 pixels.')
            return None

        default_interval = [min(self.bbox_small), max(self.bbox_large)]
        bbox_collection = []
        is_first_loop = True
        while True:
            # 如果图片是压缩得到的，我们认为压缩后的尺寸应尽量较少不必要的压缩，否则可能由过压缩而带来误检
            if is_first_loop or float(default_interval[1])/self.interval[1]*self.interval[0] <default_interval[0]:
                scale = float(default_interval[0])/self.interval[0]
                is_first_loop = False
            else:
                scale = float(default_interval[1]) / self.interval[1]
            self.interval[0] = int(self.interval[0]*scale)
            self.interval[1] = int(self.interval[1]*scale)


            input_image = cv2.resize(image, (0,0), fx=scale, fy=scale)

            pad = 0
            if input_image.shape[0] < 83 or input_image.shape[1] < 83:
                short_side = min(input_image.shape[:2])
                pad = int((83-short_side)/2 + 1)
                pad_image = numpy.zeros((input_image.shape[0]+2*pad, input_image.shape[1]+2*pad, input_image.shape[2]), dtype=numpy.uint8)
                pad_image[pad:pad+input_image.shape[0], pad:pad+input_image.shape[1], :] = input_image
                input_image = pad_image
            # cv2.imshow('im',input_image)
            # cv2.waitKey(0)

            # prepare input image
            input_height, input_width = input_image.shape[:2]
            h = int(floor(input_height / 8) * 8 +7)
            w = int((floor(input_width / 8) * 8 + 7))
            input_image = cv2.resize(input_image,(w,h))
            self._bindModel(h, w) # bind first

            input_image = input_image[:, :, :, numpy.newaxis]
            input_image = input_image.transpose([3, 2, 0, 1])

            # feedforward
            start = time.time()
            self.executor.copy_params_from({'data': mxnet.nd.array(input_image, self.ctx)})
            self.executor.forward(is_train=False)

            outputs = []
            for output in self.executor.outputs:
                outputs.append(output.asnumpy())
            # #print 'Forward time: %f ms' % ((time.time()-start)*1000)

            for idx, output in enumerate(outputs):
                output = numpy.squeeze(output)
                score_map = output[0, :, :]
                # _show_score = ((numpy.maximum(0,score_map))*255).astype(numpy.uint8)
                # _show = cv2.resize(_show_score,None,fx=4,fy=4)
                # cv2.imshow('show_%d' %idx,_show)
                # cv2.waitKey(0)
                reg_map = output[1:, :, :]

                score_map_mask = numpy.zeros(score_map.shape, dtype=numpy.uint8)
                score_map_mask[score_map > self.score_threshold] = 255
                if self.opencv_version == 4:
                    contours, hierarchy = cv2.findContours(score_map_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                else:
                    _, contours, hierarchy = cv2.findContours(score_map_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    # (x,y),radius = cv2.minEnclosingCircle(contour)
                    # x = int(x)
                    # y = int(y)
                    contour = numpy.squeeze(contour, axis=1)
                    coord_mean = numpy.mean(contour, axis=0)
                    x = int(coord_mean[0])
                    y = int(coord_mean[1])
                    score = score_map[y, x]
                    RF_center_x = self.center_area_start[idx] + self.stride[idx] * x
                    RF_center_y = self.center_area_start[idx] + self.stride[idx] * y
                    x_lt = RF_center_x - reg_map[0, y, x] * self.constant[idx]
                    y_lt = RF_center_y - reg_map[1, y, x] * self.constant[idx]
                    x_rb = RF_center_x - reg_map[2, y, x] * self.constant[idx]
                    y_rb = RF_center_y - reg_map[3, y, x] * self.constant[idx]

                    x_lt = max((x_lt - pad) / scale, 0)
                    y_lt = max((y_lt - pad) / scale, 0)
                    x_rb = min((x_rb - pad) / scale, image.shape[1] - 1)
                    y_rb = min((y_rb - pad) / scale, image.shape[0] - 1)

                    bbox_collection.append((score, x_lt, y_lt, x_rb, y_rb))
            if self.interval[1] <= default_interval[1]:
                break
            else:
                self.interval[0] = default_interval[1]

        # NMS
        bbox_collection = sorted(bbox_collection, key=lambda item:item[0], reverse=1)
        ##print(bbox_collection)
        if len(bbox_collection) > self.top_k:
            bbox_collection = bbox_collection[0:self.top_k]
        bbox_collection_numpy = numpy.empty((len(bbox_collection), 5), dtype=numpy.float32)
        for i in range(len(bbox_collection)):
            bbox_collection_numpy[i, 0]=bbox_collection[i][1]
            bbox_collection_numpy[i, 1]=bbox_collection[i][2]
            bbox_collection_numpy[i, 2]=bbox_collection[i][3]
            bbox_collection_numpy[i, 3]=bbox_collection[i][4]
            bbox_collection_numpy[i, 4]=bbox_collection[i][0]

        final_bboxes = self.NMS_fast(bbox_collection_numpy, self.NMS_threshold)
        final_bboxes_=[]
        for i in range(final_bboxes.shape[0]):
            final_bboxes_.append([final_bboxes[i, 0], final_bboxes[i, 1], final_bboxes[i, 2], final_bboxes[i, 3]])
        return final_bboxes_

    def work(self,img):
        ratio_ = 1 / float(self.ratio)
        h, w, _ = img.shape
        _img = cv2.resize(img, (0, 0), fx=ratio_, fy=ratio_)
        bboxes = self.detect_interval(_img)
        lperson = []
        for _bb in bboxes:
            bb = [int(item * self.ratio) for item in _bb]
            w1 = bb[2] - bb[0]
            h1 = bb[3] - bb[1]
            x1 = max(int(bb[0] - 0.2 * w1), 0)
            x2 = min(int(bb[2] + 0.2 * w1), w)
            y1 = max(int(bb[1]- 0.1 * h1), 0)
            y2 = min(int(bb[3] + 0.1 * h1), h)
            # lperson.append([bb[0], bb[1], bb[2], bb[3]])
            lperson.append([x1, y1, x2, y2])
        return lperson

class Headdec(object):
    def __init__(self):
        # ASSD param
        self.RF = [71.0, 111.0, 143.0, 159.0]
        self.map = [83, 41, 41, 41]
        self.stride = [4, 8, 8, 8]
        self.bbox_small = [15, 30, 60, 95]
        self.bbox_large = [30, 60, 95, 140]
        self.center_area_start = [3, 7, 7, 7]
        self.num_scale = 4
        #self.center_area_ratio = [1, 0.8, 0.6, 0.5]
        self.constant = [i/2.0 for i in self.RF]
        self.symbolFile = cfg.phone_symbol_2
        self.modelFile = cfg.phone_model_2
        self.ctx = mxnet.gpu(0)
        self.score_threshold = 0.3
        self.top_k = 100
        self.NMS_threshold = 0.2
        self.ratio = 3
        self.interval = [15,140]
        # load symbol and parameters
        if not os.path.exists(self.symbolFile):
            #print('The symbol file does not exist!!!!')
            quit()
        if not os.path.exists(self.modelFile) :
            #print('The model file does not exist!!!!')
            quit()
        self.symbolNet = mxnet.symbol.load(self.symbolFile)
        self.arg_name_arrays = {}
        self.aux_name_arrays = {}
        saved_dict = mxnet.nd.load(self.modelFile)
        for k, v in saved_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                self.arg_name_arrays.update({name: v.as_in_context(self.ctx)})
            if tp == 'aux':
                self.aux_name_arrays.update({name: v.as_in_context(self.ctx)})

        self.w, self.h = 0, 0
        self.opencv_version = int(cv2.__version__.split('.')[0])

    def NMS_fast(self, boxes, overlapThresh):
        if  boxes.shape[0]== 0:#len 可以 返回矩阵的行数
            return numpy.array([])

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype != numpy.float32:
            boxes = boxes.astype(numpy.float32)

        # initialize the list of picked indexes
        pick = []
        # grab the coordinates of the bounding boxes
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        sc = boxes[:,4]
        widths = x2-x1
        heights = y2-y1


        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = heights * widths
        idxs = numpy.argsort(sc)# 从小到大排序


        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # compare secend highest score boxes
            xx1 = numpy.maximum(x1[i], x1[idxs[:last]])
            yy1 = numpy.maximum(y1[i], y1[idxs[:last]])
            xx2 = numpy.minimum(x2[i], x2[idxs[:last]])
            yy2 = numpy.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bo（ box
            w = numpy.maximum(0, xx2 - xx1 + 1)
            h = numpy.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have
            idxs =  numpy.delete(idxs, numpy.concatenate(([last], numpy.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick]

    def _bindModel(self, height, width):
        if self.h != height or self.w != width:
            self.arg_name_arrays['data'] = mxnet.nd.zeros((1, 3, height, width), self.ctx)
            self.executor = self.symbolNet.bind(ctx=self.ctx, \
                                                args=self.arg_name_arrays, \
                                                grad_req='null', \
                                                aux_states=self.aux_name_arrays)
            self.h,self.w = height, width

    # 这个函数用于某个像素范围内的人脸，需要使用这提前估计出单个人脸在输入图像中的像素大小范围
    # interval: [bound_l, bound_r] 例如：[40, 100] 表示希望检测出输入图像中像素范围在40-100之间的人脸
    def detect_interval(self, image):

        if image.ndim != 3 or image.shape[2] != 3:
            #print('Only RGB images are supported.')
            return None

        # 如果不指定interval参数，默认使用网络可以检测的范围
        if self.interval is None:
            self.interval = [min(self.bbox_small), max(self.bbox_large)]

        if self.interval[0]<15 or self.interval[1]<self.interval[0]:
            #print('The algorithm only detects faces larger than 20 pixels.')
            return None

        default_interval = [min(self.bbox_small), max(self.bbox_large)]
        bbox_collection = []
        is_first_loop = True
        while True:
            # 如果图片是压缩得到的，我们认为压缩后的尺寸应尽量较少不必要的压缩，否则可能由过压缩而带来误检
            if is_first_loop or float(default_interval[1])/self.interval[1]*self.interval[0] <default_interval[0]:
                scale = float(default_interval[0])/self.interval[0]
                is_first_loop = False
            else:
                scale = float(default_interval[1]) / self.interval[1]
            self.interval[0] = int(self.interval[0]*scale)
            self.interval[1] = int(self.interval[1]*scale)

            input_image = cv2.resize(image, (0,0), fx=scale, fy=scale)

            pad = 0
            if input_image.shape[0] < 83 or input_image.shape[1] < 83:
                short_side = min(input_image.shape[:2])
                pad = int((83-short_side)/2 + 1)
                pad_image = numpy.zeros((int(input_image.shape[0]+2*pad), int(input_image.shape[1]+2*pad), input_image.shape[2]), dtype=numpy.uint8)
                pad_image[pad:pad+input_image.shape[0], pad:pad+input_image.shape[1], :] = input_image
                input_image = pad_image
            # cv2.imshow('im',input_image)
            # cv2.waitKey(0)

            # prepare input image
            input_height, input_width = input_image.shape[:2]
            h = int(floor(input_height / 8) * 8 +7)
            w = int((floor(input_width / 8) * 8 + 7))
            input_image = cv2.resize(input_image,(w,h))
            self._bindModel(h, w) # bind first

            input_image = input_image[:, :, :, numpy.newaxis]
            input_image = input_image.transpose([3, 2, 0, 1])

            # feedforward
            start = time.time()
            self.executor.copy_params_from({'data': mxnet.nd.array(input_image, self.ctx)})
            self.executor.forward(is_train=False)

            outputs = []
            for output in self.executor.outputs:
                outputs.append(output.asnumpy())
            # #print 'Forward time: %f ms' % ((time.time()-start)*1000)

            for idx, output in enumerate(outputs):
                output = numpy.squeeze(output)
                score_map = output[0, :, :]
                # _show_score = ((numpy.maximum(0,score_map))*255).astype(numpy.uint8)
                # _show = cv2.resize(_show_score,None,fx=4,fy=4)
                # cv2.imshow('show_%d' %idx,_show)
                # cv2.waitKey(0)
                reg_map = output[1:, :, :]

                score_map_mask = numpy.zeros(score_map.shape, dtype=numpy.uint8)
                score_map_mask[score_map > self.score_threshold] = 255
                if self.opencv_version == 4:
                    contours, hierarchy = cv2.findContours(score_map_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                else:
                    _, contours, hierarchy = cv2.findContours(score_map_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    # (x,y),radius = cv2.minEnclosingCircle(contour)
                    # x = int(x)
                    # y = int(y)
                    contour = numpy.squeeze(contour, axis=1)
                    coord_mean = numpy.mean(contour, axis=0)
                    x = int(coord_mean[0])
                    y = int(coord_mean[1])
                    score = score_map[y, x]
                    RF_center_x = self.center_area_start[idx] + self.stride[idx] * x
                    RF_center_y = self.center_area_start[idx] + self.stride[idx] * y
                    x_lt = RF_center_x - reg_map[0, y, x] * self.constant[idx]
                    y_lt = RF_center_y - reg_map[1, y, x] * self.constant[idx]
                    x_rb = RF_center_x - reg_map[2, y, x] * self.constant[idx]
                    y_rb = RF_center_y - reg_map[3, y, x] * self.constant[idx]

                    x_lt = max((x_lt - pad) / scale, 0)
                    y_lt = max((y_lt - pad) / scale, 0)
                    x_rb = min((x_rb - pad) / scale, image.shape[1] - 1)
                    y_rb = min((y_rb - pad) / scale, image.shape[0] - 1)

                    bbox_collection.append((score, x_lt, y_lt, x_rb, y_rb))
            if self.interval[1] <= default_interval[1]:
                break
            else:
                self.interval[0] = default_interval[1]

        # NMS
        bbox_collection = sorted(bbox_collection, key=lambda item:item[0], reverse=1)
        ##print(bbox_collection)
        if len(bbox_collection) > self.top_k:
            bbox_collection = bbox_collection[0:self.top_k]
        bbox_collection_numpy = numpy.empty((len(bbox_collection), 5), dtype=numpy.float32)
        for i in range(len(bbox_collection)):
            bbox_collection_numpy[i, 0]=bbox_collection[i][1]
            bbox_collection_numpy[i, 1]=bbox_collection[i][2]
            bbox_collection_numpy[i, 2]=bbox_collection[i][3]
            bbox_collection_numpy[i, 3]=bbox_collection[i][4]
            bbox_collection_numpy[i, 4]=bbox_collection[i][0]

        final_bboxes = self.NMS_fast(bbox_collection_numpy, self.NMS_threshold)
        final_bboxes_=[]
        for i in range(final_bboxes.shape[0]):
            final_bboxes_.append([final_bboxes[i, 0], final_bboxes[i, 1], final_bboxes[i, 2], final_bboxes[i, 3]])

        return final_bboxes_

    def work(self,img):
        ratio_ = 1 / float(self.ratio)
        h, w, _ = img.shape
        _img = cv2.resize(img, (0, 0), fx=ratio_, fy=ratio_)
        bboxes = self.detect_interval(_img)
        lhead = []
        n = 0
        for _bb in bboxes:
            bb = [int(item * self.ratio) for item in _bb]
            w1 = bb[2] - bb[0]
            h1 = bb[3] - bb[1]
            x1 = max(int(bb[0] - 0.5*w1), 0)
            x2 = min(int(bb[2] + 0.5*w1), w)
            y1 = max(int(bb[1] - 0.1 * h1), 0)
            y2 = min(int(bb[3] + 0.7*h1), h)
            if y1 < y2 and x1 < x2 and y2 <= h and x2 <= w:
                lhead.append([bb[0], bb[1], bb[2], bb[3], x1, y1, x2, y2])
        return lhead

def get_center_rec(img):
    H,W = img.shape[0], img.shape[1]
    if H>W:
        offset = int((H-W)/2)
        _img = img[offset:H-offset,:,:]
    else:
        offset = int((W-H)/2)
        _img = img[ :,offset:W - offset, :]
    return _img

class Hand_class(object):
    def __init__(self):
        self.ctx = mxnet.gpu(0)
        self.net_file = cfg.phone_model
        sym, arg_params, aux_params = mxnet.model.load_checkpoint(self.net_file,53)
        self.mod = mxnet.mod.Module(symbol=sym, context=self.ctx, label_names=None)
        self.mod.bind(for_training=False, data_shapes=[('data', (1, 3, 112, 112))],
                      label_shapes=self.mod._label_shapes)
        self.mod.set_params(arg_params, aux_params, allow_missing=True)
        self.Batch = namedtuple('Batch', ['data'])

    def work(self, img_BGR):
        img_BGR = get_center_rec(img_BGR)
        img_BGR = cv2.resize(img_BGR, (128, 128))
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        img = img - 127.5
        input_image = img[:, :, :, np.newaxis]
        input_image = input_image.transpose([3, 2, 0, 1])
        self.mod.forward(self.Batch([mxnet.nd.array(input_image, self.ctx)]))
        prob = self.mod.get_outputs()[0].asnumpy()
        prob = np.squeeze(prob)
        max_id = np.argmax(prob)
        max_score = prob[max_id]
        return max_id, max_score

class Manager(AlgorithmBase):

    def __init__(self, save_flag,params1):

        self.save_flag = save_flag
        self.state = True
        self.output_img = None

        self.fourcc = 828601953
        self.input_Q = queue.Queue(cfg.queue_lenth)
        self.output_img = None

        self.remember_json = {}
        self.alarm_json = {}
        self.resize = (960, 540)
        self.output_Q = params1
        self.record = PointsRecorder(record_len=10, num_thr=6)




    def loop(self):
        n = 0
        Persons = Persondec()
        Head = Headdec()
        HC = Hand_class()
        # camera = cv2.VideoCapture(cfg.phone_video)
        #
        # fps = camera.get(5)
        # if fps > 50:
        #     fps = 30
        # fps = int(fps)
        # size = (int(camera.get(3)), int(camera.get(4)))
        # fourcc = cv2.VideoWriter_fourcc(*'MP42')  # 最小压缩格式
        # # if not os.path.exists('./output_video'):
        # #     os.makedirs('./output_video')
        # save_path = 'result/output_video/cellphone.avi'
        # for i in range(1000):
        #     if not os.path.exists('result/output_video/cellphone.avi'):
        #         save_path = 'result/output_video/cellphone.avi'
        #         break
        # if self.save_flag:
        #     saveVideo = cv2.VideoWriter(save_path, fourcc, fps, size)

        push_get = threading.Thread(target=PUSHER(self.input_Q, cfg.phone_video, ).Image_pusher)
        push_get.setDaemon(True)
        push_get.start()

        # time.sleep(2)

        while True:
            # time.sleep(0.3)

            img = self.input_Q.get()
            if self.state == True:
                # print('吸烟打电话检测正在运行')
                try:
                    insk = []
                    persons = Persons.work(img)
                    for pe in persons:
                        pecrop = img[pe[1]:pe[3], pe[0]:pe[2]]
                        heads = Head.work(pecrop)
                        for he in heads:
                            hecrop = pecrop[he[5]:he[7], he[4]:he[6]]
                            cl, sc = HC.work(hecrop)
                            if cl == 1:
                                self.record.update(1)
                                if self.record.warning_flag1():
                                    if 1 not in insk:
                                        insk.append(1)
                                    cv2.rectangle(img, (pe[0], pe[1]), (pe[2], pe[3]), (0, 0, 255), thickness=3)

                            elif cl == 2:
                                self.record.update(1)
                                if self.record.warning_flag1():
                                    if 2 not in insk:
                                        insk.append(2)
                                    cv2.rectangle(img, (pe[0], pe[1]), (pe[2], pe[3]), (255, 0, 0), thickness=3)
                            else:
                                self.record.update(0)

                    # print(time.ctime() + '吸烟打电话标志++++++++++++++++++++++++', str(insk))
                    # _show = img.copy()
                    if img is not None:
                        self.output_img = img.copy()
                    if (1 not in insk) and (2 not in insk):
                        if 'ALARM' in self.alarm_json:
                            self.alarm_json['ALARM'][0].write(cv2.resize(img, self.resize))
                            if (datetime.datetime.now() - self.alarm_json['ALARM'][1]).seconds > 10:
                                self.alarm_json['ALARM'][0].release()
                                self.output_Q.put(
                                    {
                                        'alarm_jpg': self.alarm_json['ALARM'][2],
                                        'alarm_video': self.alarm_json['ALARM'][3],
                                        'alarm_time': self.alarm_json['ALARM'][1],
                                        'alarm_index': str(uuid.uuid1()),
                                        'types': 'phone_smoke',
                                        'type_flag': str(self.alarm_json['ALARM'][4]),
                                    }
                                )
                                self.alarm_json.pop('ALARM')


                    else:
                        if 'ALARM' not in self.alarm_json:
                            '''Control warnings for every miniute.'''
                            curr_time = time.time()

                            if 'ING' not in self.remember_json:
                                self.remember_json['ING'] = curr_time
                            elif curr_time - self.remember_json['ING'] > 18:
                                self.remember_json['ING'] = curr_time
                            else:
                                continue

                            alarm_package = os.path.join(
                                os.environ['HOME'] + "/AIResult"  + '/SmokePhoneResult/' + str(time.ctime()) + str(uuid.uuid1()) + '/')
                            if not os.path.exists(alarm_package):
                                os.makedirs(alarm_package)
                            alarm_jpg = str(uuid.uuid1()) + '.jpg'
                            alarm_video = str(uuid.uuid1()) + '.mp4'
                            jpg_path = os.path.join(alarm_package + alarm_jpg)
                            video_path = os.path.join(alarm_package + alarm_video)
                            type_flag = ''
                            if (1 in insk) and (2 not in insk):
                                type_flag = 1
                            elif (2 in insk) and (1 not in insk):
                                type_flag = 2
                            elif (1 in insk) and (2 in insk):
                                type_flag = 3
                            # print("==============\n", type_flag)
                            if type_flag != "":
                                print("检测到吸烟打电话行为！！！", type_flag)
                            cv2.imwrite(jpg_path, img)
                            video_writer = cv2.VideoWriter(video_path, self.fourcc, 5, self.resize)
                            self.alarm_json['ALARM'] = [video_writer, datetime.datetime.now(), jpg_path, video_path, type_flag]
                            # img =
                            self.alarm_json['ALARM'][0].write(cv2.resize(img, self.resize))

                        else:
                            # img =
                            self.alarm_json['ALARM'][0].write(cv2.resize(img, self.resize))
                            if (datetime.datetime.now() - self.alarm_json['ALARM'][1]).seconds > 10:
                                self.alarm_json['ALARM'][0].release()
                                self.output_Q.put(
                                    {
                                        'alarm_jpg': self.alarm_json['ALARM'][2],
                                        'alarm_video': self.alarm_json['ALARM'][3],
                                        'alarm_time': self.alarm_json['ALARM'][1],
                                        'alarm_index': str(uuid.uuid1()),
                                        'types': 'phone_smoke',
                                        'type_flag': str(self.alarm_json['ALARM'][4]),
                                    }
                                )
                                self.alarm_json.pop('ALARM')
                except Exception as E4R:
                    #print(('吸烟打电话报错'+time.ctime() + str(E4R) + '\n') * 10)
                    time.sleep(0.5)
                    continue


            # if _show is not None:
            #     self.output_img = _show.copy()

    def get_output(self):
        return self.output_img

    def set_state(self, state):
        self.state = state

class PUSHER(object):
    def __init__(self,Q,video_stream):
        self.Q = Q
        self.video_stream = video_stream

    def Image_pusher(self,):
        # 创建流媒体并初始化
        # #print(video_stream,'****************')
        #print(time.ctime()+'video come in:',self.video_stream)
        reader = TensorStreamConverter(self.video_stream, repeat_number=2)

        try:
            reader.initialize()
            reader.start()
            #print('Start Stream Successfully!')
            img_size = reader.getFrameSize()
        except RuntimeError:
            #print('Start channel Wrong happened in %s' % self.video_stream)
            reader.stop()
            reader.release()

        _watchQ = True
        _watchCurrentTime = time.time()
        wrong = 0
        while True:
            time.sleep(0.3)
            try:
                img = reader.readFrame(pixel_format=FourCC.BGR24, width=img_size[0], height=img_size[1])
                self.Q.put(img)
                #print(time.ctime()+'吸烟打电话frame队列长度===>',self.Q.qsize())
                if self.Q.qsize() >= 99:
                    self.Q.queue.clear()

                # assert img is not None
            except:
                wrong += 1
                if wrong > 5:
                    #print('--------------------------------------------CameraID:%s stream fialed.' % self.video_stream)
                    self.running = False
                    break
                else:
                    continue


if __name__ == '__main__':
    M = Manager(False,'')
    M.start()
