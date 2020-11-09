# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import time
import mxnet


class ASSD_base(object):
    def __init__(self, symbolFile, modelFile, ctx):
        self.symbolFile = symbolFile
        self.modelFile = modelFile
        self.ctx = ctx
        # ASSD-model param
        self.num_scale = 5
        self.stride = [4, 8, 8, 8, 16]
        self.bbox_small = [20, 30, 60, 95, 140]
        self.bbox_large = [30, 60, 95, 140, 210]
        self.center_area_start = [3, 7, 7, 7, 15]
        self.map = [83, 41, 41, 41, 20]
        # self.center_area_ratio = [1, 0.8, 0.6, 0.5]
        self.RF = [71.0, 111.0, 143.0, 159.0, 239.0]
        self.constant = [i / 2.0 for i in self.RF]
        self.H, self.W = 0, 0

        self.select_scale = {}
        for i in range(max(self.bbox_large) + 1):
            for scale_idx, j in enumerate(self.bbox_large):
                if i <= j:
                    self.select_scale[i] = scale_idx
                    break
        # load symbol and parameters
        if not os.path.exists(self.symbolFile):
            print('The symbol file does not exist!!!!')

            quit()
        if not os.path.exists(self.modelFile):
            print('The model file does not exist!!!!')

            quit()
        self.symbolNet = mxnet.symbol.load(self.symbolFile)
        self.arg_name_arrays = {}
        self.aux_name_arrays = {}
        self.executor = None
        saved_dict = mxnet.nd.load(self.modelFile)
        for k, v in saved_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                self.arg_name_arrays.update({name: v.as_in_context(self.ctx)})
            if tp == 'aux':
                self.aux_name_arrays.update({name: v.as_in_context(self.ctx)})

    def NMS_fast(self, boxes, overlapThresh):
        if boxes.shape[0] == 0:  # len 可以 返回矩阵的行数
            return np.array([])

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype != np.float32:
            boxes = boxes.astype(np.float32)

        # initialize the list of picked indexes
        pick = []
        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        sc = boxes[:, 4]
        widths = x2 - x1
        heights = y2 - y1

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = heights * widths
        idxs = np.argsort(sc)  # 从小到大排序

        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # compare secend highest score boxes
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bo（ box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick]

    def _bindModel(self, height, width):
        if self.H != height or self.W != width:
            self.arg_name_arrays['data'] = mxnet.nd.zeros((1, 3, height, width), self.ctx)
            self.executor = self.symbolNet.bind(ctx=self.ctx, \
                                                args=self.arg_name_arrays, \
                                                grad_req='null', \
                                                aux_states=self.aux_name_arrays)
            self.H, self.W = height, width

    # 这个函数用于某个像素范围内的人脸，需要使用这提前估计出单个人脸在输入图像中的像素大小范围
    # interval: [bound_l, bound_r] 例如：[40, 100] 表示希望检测出输入图像中像素范围在40-100之间的人脸
    def detect_interval(self, image, interval=None, score_threshold=0.8, top_k=100,
                        NMS_threshold=0.3, bboxes_rescale=1):

        if image.ndim != 3 or image.shape[2] != 3:
            print('Only RGB images are supported.')
            return None

        # 如果不指定interval参数，默认使用网络可以检测的范围
        if interval is None:
            interval = [min(self.bbox_small), max(self.bbox_large)]

        if interval[0] < 20 or interval[1] < interval[0]:
            print('The algorithm only detects faces larger than 20 pixels.')
            return None

        default_interval = [min(self.bbox_small), max(self.bbox_large)]
        bbox_collection = []
        is_first_loop = True
        while True:
            # 如果图片是压缩得到的，我们认为压缩后的尺寸应尽量较少不必要的压缩，否则可能由过压缩而带来误检
            if is_first_loop or float(default_interval[1]) / interval[1] * interval[0] < default_interval[0]:
                scale = float(default_interval[0]) / interval[0]
                is_first_loop = False
            else:
                scale = float(default_interval[1]) / interval[1]
            interval[0] = int(interval[0] * scale)
            interval[1] = int(interval[1] * scale)

            input_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

            pad = 0
            if input_image.shape[0] < 83 or input_image.shape[1] < 83:
                short_side = min(input_image.shape[:2])
                pad = (83 - short_side) / 2 + 1
                pad_image = np.zeros(
                    (input_image.shape[0] + 2 * pad, input_image.shape[1] + 2 * pad, input_image.shape[2]),
                    dtype=np.uint8)
                pad_image[pad:pad + input_image.shape[0], pad:pad + input_image.shape[1], :] = input_image
                input_image = pad_image
            # prepare input image
            input_height, input_width = input_image.shape[:2]
            self._bindModel(input_height, input_width)  # bind first

            input_image = input_image[:, :, :, np.newaxis]
            input_image = input_image.transpose([3, 2, 0, 1])

            # feedforward
            start = time.time()
            self.executor.copy_params_from({'data': mxnet.nd.array(input_image, self.ctx)})
            self.executor.forward(is_train=False)

            outputs = []
            for output in self.executor.outputs:
                outputs.append(output.asnumpy())
            # print 'Forward time: %f ms' % ((time.time()-start)*1000)

            for idx, output in enumerate(outputs):
                output = np.squeeze(output)
                score_map = output[0, :, :]
                reg_map = output[1:, :, :]

                score_map_mask = np.zeros(score_map.shape, dtype=np.uint8)
                score_map_mask[score_map > score_threshold] = 255
                _, contours, hierarchy = cv2.findContours(score_map_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    contour = np.squeeze(contour, axis=1)
                    coord_mean = np.mean(contour, axis=0)
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
            if interval[1] <= default_interval[1]:
                break
            else:
                interval[0] = default_interval[1]

        # NMS
        bbox_collection = sorted(bbox_collection, key=lambda item: item[0], reverse=1)
        # print(bbox_collection)
        if len(bbox_collection) > top_k:
            bbox_collection = bbox_collection[0:top_k]
        bbox_collection_np = np.empty((len(bbox_collection), 5), dtype=np.float32)
        for i in range(len(bbox_collection)):
            bbox_collection_np[i, 0] = bbox_collection[i][1]
            bbox_collection_np[i, 1] = bbox_collection[i][2]
            bbox_collection_np[i, 2] = bbox_collection[i][3]
            bbox_collection_np[i, 3] = bbox_collection[i][4]
            bbox_collection_np[i, 4] = bbox_collection[i][0]

        final_bboxes = self.NMS_fast(bbox_collection_np, NMS_threshold)
        final_bboxes = final_bboxes * bboxes_rescale
        final_bboxes_ = []
        for i in range(final_bboxes.shape[0]):
            final_bboxes_.append((final_bboxes[i, 0], final_bboxes[i, 1], final_bboxes[i, 2], final_bboxes[i, 3]))

        return final_bboxes_


class ASSD_predictor():
    def __init__(self):
        net_file = 'resources/face-recognition-models/ASSD-model/deep_ASSD_5scale_deeperlosses_32_64_64_deploy.json'
        param_file = \
            'resources/face-recognition-models/ASSD-model/ASSD_face_32_64_64_small18_deeperlosses_iter_850000_model.params'
        self.my_predictor = ASSD_base(net_file, param_file, mxnet.gpu(0))

    def get_face_part(self, img, min_h=0, scale=1.0, num_max=5, target_face_len=112):
        img_temp = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        '''获得人脸图像'''
        bboxes = self.my_predictor.detect_interval(img_temp, interval=[20, 210], score_threshold=0.7, top_k=10000,
                                                   NMS_threshold=0.3)
        '''Sorted faces by height'''
        bboxes = sorted(bboxes, key=lambda item: (item[2] - item[0]), reverse=1)[:num_max]

        '''按照原始尺度获取人脸图片区域'''
        bboxes = [[int(item / scale) for item in bb] for bb in bboxes]
        face_parts = []
        for bbox in bboxes:
            if bbox[3] - bbox[1] < min_h:
                continue
            W, H = img.shape[1], img.shape[0]
            # crop = [max(int(bbox[0] - (bbox[2] - bbox[0]) * extra), 0),
            #         max(int(bbox[1] - (bbox[3] - bbox[1]) * extra), 0),
            #         min(int(bbox[2] + (bbox[2] - bbox[0]) * extra), W),
            #         min(int(bbox[3] + (bbox[3] - bbox[1]) * extra), H)]
            # cv2.rectangle(img, (crop[0], crop[1]), (crop[2], crop[3]), (0, 255, 255), thickness=2)
            part = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            # part = cv2.resize(part,(target_face_len,target_face_len))

            face_parts.append(part)
        # scale = min(float(200) / float(part.shape[0]), 1)
        # part = cv2.resize(part, (0, 0), fx=scale, fy=scale)
        return face_parts

    def get_ori_face(self, src_path, dst_path_prefix):
        target_scale = 500
        success = False
        if not os.path.isfile(src_path):
            return '人脸注册文件 %s 未找到.' % src_path, success, None
        img = cv2.imread(src_path)
        if img is None:
            return '文件 %s 不是照片.' % src_path, success, None
        H, W = img.shape[0], img.shape[1]
        scale = min(float(720) / float(img.shape[0]), 1)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        bboxes = self.my_predictor.detect_interval(img, interval=[20, img.shape[0]], score_threshold=0.7, top_k=10000,
                                                   NMS_threshold=0.3)

        # Choose the largest bbox detected
        if len(bboxes) < 1:
            return '在文件 %s 中未找到人脸.' % src_path, success, ''
        elif len(bboxes) > 1:
            # 在多张人脸中寻找最大的一个作为输出
            max_len = 0
            for bb in bboxes:
                if max(bb[2] - bb[0], bb[3] - bb[1]) > max_len:
                    bbox, max_len = bb, max(bb[2] - bb[0], bb[3] - bb[1])
        else:
            bbox = bboxes[0]

        # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), thickness=2)
        # Enlarge 1.4x and crop
        crop = [max(int(bbox[0] - (bbox[2] - bbox[0]) * 0.2), 0), max(int(bbox[1] - (bbox[3] - bbox[1]) * 0.2), 0),
                min(int(bbox[2] + (bbox[2] - bbox[0]) * 0.2), W), min(int(bbox[3] + (bbox[3] - bbox[1]) * 0.2), H)]
        # cv2.rectangle(img, (crop[0], crop[1]), (crop[2], crop[3]), (0, 255, 255), thickness=2)
        part = img[crop[1]:crop[3], crop[0]:crop[2], :]
        scale = min(1, target_scale / float(part.shape[0]))
        try:
            part = cv2.resize(part, (0, 0), fx=scale, fy=scale)
        except:
            print('ERROR! Resize failed for img is None.')
        # cv2.imshow('part',part)
        # cv2.waitKey()
        self.update_save_index()
        save_path = os.path.join(dst_path_prefix, 'face_%d.JPG' % self.save_index)
        cv2.imwrite(save_path, part)
        success = True
        return '人脸 %s 抠图成功!' % src_path, success, save_path

    def get_face_tobecompared(self, img, dst_path_prefix):
        start = time.time()
        save_path_list = []
        # if not os.path.isfile(src_path):
        #     return save_path_list,'人脸比对文件 %s 未找到.' % src_path
        # img = cv2.imread(src_path)
        # show = cv2.resize(img,(0,0),fx=0.2,fy=0.2)
        # cv2.imshow('im',show)
        # cv2.waitKey(0)
        if img is None:
            return save_path_list
        print
        'Read image costs %f s.' % (time.time() - start)
        start = time.time()
        target_dst_H = 500  # 存储图片高
        target_src_H = 1080  # 输入图片高
        if target_src_H < img.shape[0]:
            re_scale = target_src_H / float(img.shape[0])
            input_im = cv2.resize(img, (0, 0), fx=re_scale, fy=re_scale)
        else:
            re_scale = 1.0
            input_im = img
        bboxes = self.my_predictor.detect_interval(input_im, interval=[20, img.shape[0]], score_threshold=0.7,
                                                   top_k=3, NMS_threshold=0.3, bboxes_rescale=1 / re_scale)
        H, W = img.shape[0], img.shape[1]
        for idx, bbox in enumerate(bboxes):
            self.update_save_index()
            save_path = os.path.join(dst_path_prefix, 'face_%d.JPG' % self.save_index)
            save_path_list.append(save_path)
            crop = [int(max(bbox[0] - (bbox[2] - bbox[0]) * .2, 0)),
                    int(max(bbox[1] - (bbox[3] - bbox[1]) * .2, 0)),
                    int(min(bbox[2] + (bbox[2] - bbox[0]) * .2, W)),
                    int(min(bbox[3] + (bbox[3] - bbox[1]) * .2, H))]
            part = img[crop[1]:crop[3], crop[0]:crop[2], :]

            target_scale = target_dst_H / float(crop[3] - crop[1])
            if target_scale < 1:
                part = cv2.resize(part, (0, 0), fx=target_scale, fy=target_scale)
            # cv2.imshow('part',part)
            # cv2.waitKey(0)
            cv2.imwrite(save_path, part)
        print('Post-processing include detection costs %f s.' % (time.time() - start))

        return save_path_list  # , '已完成 %s 中的人脸抠图!' % src_path
