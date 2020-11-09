# coding: utf-8

import os
import cv2
import time
import mxnet as mx
import numpy as np
from util import AlgorithmBase
from math import floor
from collections import namedtuple
from PIL import Image, ImageDraw, ImageFont
import uuid

import config as cfg
import datetime
from tensor_stream import TensorStreamConverter, FourCC
import queue
import threading

class Classify(object, ):
    def __init__(self, ctx):
        self.ctx = ctx
        self.People_Classify_model = cfg.People_Classify_model
        self.People_Classify_num = cfg.People_Classify_num
        self.net_file = os.path.join(os.path.dirname(__file__),
                                     '../resources/people-detect-model/%s' % self.People_Classify_model)
        sym, arg_params, aux_params = mx.model.load_checkpoint(self.net_file, self.People_Classify_num)
        self.mod = mx.mod.Module(symbol=sym, context=self.ctx, label_names=None)
        self.mod.bind(for_training=False, data_shapes=[('data', (1, 3, 256, 120))],
                      label_shapes=self.mod._label_shapes)
        self.mod.set_params(arg_params, aux_params, allow_missing=True)
        self.Batch = namedtuple('Batch', ['data'])

    def Work(self, img_BGR):
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        start = time.time()
        img = cv2.resize(img, (120, 256))
        img = img - 127.5
        input_image = img[:, :, :, np.newaxis]
        input_image = input_image.transpose([3, 2, 0, 1])
        self.mod.forward(self.Batch([mx.nd.array(input_image, self.ctx)]))
        prob = self.mod.get_outputs()[0].asnumpy()
        prob = np.squeeze(prob)
        max_id = np.argmax(prob)  # (A:0 B:1 C:2)
        # #print("max_id: ", max_id)
        # #print('It costs %f ms.' % ((time.time() - start) * 1000.0))
        return max_id


class MyPredict(object):
    def __init__(self, ctx):
        self.symbolFile = os.path.join(os.path.dirname(__file__),
                                       '../resources/people-detect-model/%s' % cfg.symbolFile)
        self.modelFile = os.path.join(os.path.dirname(__file__), '../resources/people-detect-model/%s' % cfg.modelFile)
        self.ctx = ctx

        # ASSD-model param
        self.RF = [71.0, 111.0, 143.0, 159.0]
        self.map = [83, 41, 41, 41]
        self.stride = [4, 8, 8, 8]
        self.bbox_small = [15, 30, 60, 95]
        self.bbox_large = [30, 60, 95, 140]
        self.center_area_start = [3, 7, 7, 7]
        self.num_scale = 4
        # self.center_area_ratio = [1, 0.8, 0.6, 0.5]
        self.constant = [i / 2.0 for i in self.RF]

        # load symbol and parameters
        if not os.path.exists(self.symbolFile):
            #print('The symbol file does not exist!!!!')
            quit()
        if not os.path.exists(self.modelFile):
            #print('The model file does not exist!!!!')
            quit()
        self.symbolNet = mx.symbol.load(self.symbolFile)
        self.arg_name_arrays = {}
        self.aux_name_arrays = {}
        saved_dict = mx.nd.load(self.modelFile)
        for k, v in saved_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                self.arg_name_arrays.update({name: v.as_in_context(self.ctx)})
            if tp == 'aux':
                self.aux_name_arrays.update({name: v.as_in_context(self.ctx)})

        self.w, self.h = 0, 0

        self.time, _ = get_time_stamp()
        # if not os.path.exists("./worker"):
        #     os.mkdir("./worker")
        # self.warning_path0 = os.path.join(os.path.abspath('./'), 'worker')
        # if not os.path.exists("./stationmaster"):
        #     os.mkdir("./stationmaster")
        # self.warning_path1 = os.path.join(os.path.abspath('./'), 'stationmaster')
        # if not os.path.exists("./others"):
        #     os.mkdir("./others")
        # self.warning_path2 = os.path.join(os.path.abspath('./'), 'others')

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
        if self.h != height or self.w != width:
            self.arg_name_arrays['data'] = mx.nd.zeros((1, 3, height, width), self.ctx)
            self.executor = self.symbolNet.bind(ctx=self.ctx, \
                                                args=self.arg_name_arrays, \
                                                grad_req='null', \
                                                aux_states=self.aux_name_arrays)
            self.h, self.w = height, width

    # 这个函数用于某个像素范围内的人脸，需要使用这提前估计出单个人脸在输入图像中的像素大小范围
    # interval: [bound_l, bound_r] 例如：[40, 100] 表示希望检测出输入图像中像素范围在40-100之间的人脸
    def detect_interval(self, image, interval=None, score_threshold=0.8, top_k=100, NMS_threshold=0.3):

        if image.ndim != 3 or image.shape[2] != 3:
            #print('Only RGB images  are supported.')
            return None

        # 如果不指定interval参数，默认使用网络可以检测的范围
        if interval is None:
            interval = [min(self.bbox_small), max(self.bbox_large)]

        if interval[0] < 15 or interval[1] < interval[0]:
            #print('The algorithm only detects faces larger than 20 pixels.')
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
            if input_image.shape[0] < 83 or input_image.shape[1] < 93:
                short_side = min(input_image.shape[:2])
                pad = (83 - short_side) / 2 + 1
                pad_image = np.zeros(
                    (input_image.shape[0] + 2 * pad, input_image.shape[1] + 2 * pad, input_image.shape[2]),
                    dtype=np.uint8)
                pad_image[pad:pad + input_image.shape[0], pad:pad + input_image.shape[1], :] = input_image
                input_image = pad_image
            # cv2.imshow('im',input_image)
            # cv2.waitKey(0)

            # prepare input images
            input_height, input_width = input_image.shape[:2]
            h = int(floor(input_height / 8) * 8 + 7)
            w = int((floor(input_width / 8) * 8 + 7))
            input_image = cv2.resize(input_image, (w, h))
            self._bindModel(h, w)  # bind first

            input_image = input_image[:, :, :, np.newaxis]
            input_image = input_image.transpose([3, 2, 0, 1])

            # feedforward
            start = time.time()
            self.executor.copy_params_from({'data': mx.nd.array(input_image, self.ctx)})
            self.executor.forward(is_train=False)

            outputs = []
            for output in self.executor.outputs:
                outputs.append(output.asnumpy())
            # #print 'Forward time: %f ms' % ((time.time()-start)*1000)

            for idx, output in enumerate(outputs):
                output = np.squeeze(output)
                score_map = output[0, :, :]
                reg_map = output[1:, :, :]

                score_map_mask = np.zeros(score_map.shape, dtype=np.uint8)
                score_map_mask[score_map > score_threshold] = 255
                _, contours, hierarchy = cv2.findContours(score_map_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    # (x,y),radius = cv2.minEnclosingCircle(contour)
                    # x = int(x)
                    # y = int(y)
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
        # #print(bbox_collection)
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
        final_bboxes_ = []
        for i in range(final_bboxes.shape[0]):
            final_bboxes_.append([final_bboxes[i, 0], final_bboxes[i, 1], final_bboxes[i, 2], final_bboxes[i, 3]])

        return final_bboxes_


def pick_bbox(bboxes, target_areas, thresholed=.3):
    '''
    :param bboxes: [[x0,y0,x1,y1],[x0,y0,x1,y1],...]
    :param thresholed: IOU threshold for judging who to keep
    :return: resulted_bboxs
    '''
    if bboxes == []:
        return []

    resulted_bboxs = []
    bboxes = np.array(bboxes)

    for area in target_areas:
        # area = np.array(area)
        # compare secend highest score boxes
        xx1 = np.maximum(area[0], bboxes[:, 0])
        yy1 = np.maximum(area[1], bboxes[:, 1])
        xx2 = np.minimum(area[2], bboxes[:, 2])
        yy2 = np.minimum(area[3], bboxes[:, 3])

        # compute the width and height of the bbox
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # #print("**************", (bboxes[:, 2], bboxes[:, 3]))

        # compute the ratio of overlap
        # overlap = (w * h) / ((area[2] - area[0]) * (area[3] - area[1]))
        overlap = (w * h) / ((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]))
        # #print(overlap)
        for idx, iou in enumerate(overlap):
            if iou > thresholed:
                resulted_bboxs.append(bboxes[idx])
    return resulted_bboxs
    # delete all indexes from the index list that have


class Manager(AlgorithmBase):
    def __init__(self, save_flag,params1):
        super(AlgorithmBase, self).__init__()
        self.target_areas = []
        self.ratio = cfg.ratio
        self.skip = cfg.video_skip
        self.img_count = 0
        self.fram_index = 0
        # self.camera = cv2.VideoCapture(cfg.People_Detect_video)

        self.fourcc = 828601953
        self.input_Q = queue.Queue(cfg.queue_lenth)


        # if (self.camera.isOpened()):  # 判断视频是否打开
        #     #print('Open')
        # else:
        #     #print('摄像头未打开')

        # 测试用,查看视频size
        # if self.camera.get(5) > 50:
        #     #print('warning: we get fps %d.' % self.camera.get(5))
        #     self.fps = 25
        # else:
        #     self.fps = int(self.camera.get(5))
        # self.size = (int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        #              int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # #print('fps: %d. size: %s.' % (self.fps, str(self.size)))

        self.font_path = os.path.join(os.path.dirname(__file__), '../util/guanjiaKai.ttf')
        self.area_config = os.path.join(os.path.dirname(__file__), '../config_scripts/cloth_config.txt')
        self.selectedFont = ImageFont.truetype(self.font_path, 35)
        self.output_img = None
        self.state = True
        self.save_flag = save_flag

        self.remember_json = {}
        self.alarm_json = {}
        self.resize = (960, 540)
        self.output_Q = params1

        # 保存视频
        # if self.save_flag:
        #     self.fourcc = cv2.VideoWriter_fourcc(*'MP42')
        #     self.save_video = cv2.VideoWriter('result/output_video/people.avi', self.fourcc, self.fps, self.size)

        def get_area(area_config):
            temp_area = []
            with open(area_config, 'r') as fin:
                areas = fin.readlines()
            for area in areas:
                area = area.strip().split(' ')

                if len(area) == 0:
                    pass
                else:
                    for n in range(len(area) // 4):
                        temp_area.append(
                            [int(area[4 * n + 0]), int(area[4 * n + 1]), int(area[4 * n + 2]), int(area[4 * n + 3])])
            fin.close()
            return temp_area

        self.area = get_area(self.area_config)

    def loop(self):
        my_predictor = MyPredict(mx.gpu(0))
        my_classify = Classify(mx.gpu(0))

        push_get = threading.Thread(target=PUSHER(self.input_Q, cfg.People_Detect_video, ).Image_pusher)
        push_get.setDaemon(True)
        push_get.start()

        # time.sleep(2)

        while True:


            # time.sleep(0.3)
            # ok, img = self.camera.read()
            img = self.input_Q.get()
            start = time.time()
            # if not ok:
            #     self.camera.release()
            #     if self.save_flag:
            #         self.save_video.release()
            #     break

            self.fram_index += 1
            if self.fram_index % self.skip != 0:
                continue

            if len(self.area) == 0:
                self.target_areas = [[0, 0, 200, 50]]
            elif len(self.area) >= 1:
                self.target_areas = self.area
                # #print("**************", self.target_areas)

            self.ratio_ = 1 / float(self.ratio)
            _img = cv2.resize(img, (0, 0), fx=self.ratio_, fy=self.ratio_)

            if self.state == True:
                #print('着装检测正在运行')
                try:
                    tx = datetime.datetime.now()
                    _bboxes = my_predictor.detect_interval(_img, interval=[15, 140], score_threshold=.3, top_k=100,
                                                           NMS_threshold=.3)

                    roi_region = [0, 0, 200, 50]
                    img[roi_region[1]:roi_region[3], roi_region[0]:roi_region[2], :] = [255, 255, 255]
                    img_PIL = Image.fromarray(img)
                    draw = ImageDraw.Draw(img_PIL)
                    draw.text((24, 0), "着装监控", (0, 255, 0), font=self.selectedFont)
                    img = np.array(img_PIL)

                    temp_bbs = []
                    for bb in _bboxes:
                        _bb = [int(item * self.ratio) for item in bb]
                        temp_bbs.append(_bb)

                    bboxes = pick_bbox(temp_bbs, self.target_areas)

                    ins = []
                    for bb in bboxes:
                        part = img[bb[1]:bb[3], bb[0]:bb[2], :]
                        # people_status = predict_imgs(part)
                        people_status = my_classify.Work(part)
                        self.img_count += 1

                        p3 = (max(bb[0], 15), max(bb[1], 15))
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        _, frame_NUM = get_time_stamp()

                        if people_status == 0:
                            ins.append(0)
                            img[roi_region[1]:roi_region[3], roi_region[0]:roi_region[2], :] = [255, 255, 255]
                            img_PIL = Image.fromarray(img)
                            draw = ImageDraw.Draw(img_PIL)
                            draw.text((48, 0), "安全", (0, 255, 0), font=self.selectedFont)
                            img = np.array(img_PIL)
                            title = "work people"
                            cv2.putText(img, title, p3, font, 0.6, (0, 255, 0), 1)
                            cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), thickness=1)
                            # cv2.imwrite('%s/C_%s.jpg' % (my_predictor.warning_path0, frame_NUM), part)

                        elif people_status == 1:
                            ins.append(1)
                            img[roi_region[1]:roi_region[3], roi_region[0]:roi_region[2], :] = [255, 255, 255]
                            img_PIL = Image.fromarray(img)
                            draw = ImageDraw.Draw(img_PIL)
                            draw.text((48, 0), "安全", (0, 255, 0), font=self.selectedFont)
                            img = np.array(img_PIL)
                            title = "stationmaster"
                            cv2.putText(img, title, p3, font, 0.6, (255, 0, 0), 1)
                            cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), thickness=1)
                            # cv2.imwrite('%s/C_%s.jpg' % (my_predictor.warning_path1, frame_NUM), part)

                        elif people_status == 2:
                            ins.append(2)
                            img[roi_region[1]:roi_region[3], roi_region[0]:roi_region[2], :] = [0, 0, 0]
                            img_PIL = Image.fromarray(img)
                            draw = ImageDraw.Draw(img_PIL)
                            draw.text((40, 0), "着装报警!", (0, 0, 255), font=self.selectedFont)
                            img = np.array(img_PIL)
                            title = "stationmaster"
                            cv2.putText(img, title, p3, font, 0.6, (0, 0, 255), 1)
                            cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), thickness=1)
                            # cv2.imwrite('%s/C_%s.jpg' % (my_predictor.warning_path2, frame_NUM), part)

                    for bb in self.target_areas:
                        cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), thickness=2)

                    self.output_img = img

                    if 2 not in ins:
                        if 'ALARM' in self.alarm_json:
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
                                        'types': 'cloths'
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
                                os.environ['HOME'] + "/AIResult"  + '/ClothsResult/' + str(time.ctime()) + str(uuid.uuid1()) + '/')
                            if not os.path.exists(alarm_package):
                                os.makedirs(alarm_package)
                            alarm_jpg = str(uuid.uuid1()) + '.jpg'
                            alarm_video = str(uuid.uuid1()) + '.mp4'
                            jpg_path = os.path.join(alarm_package + alarm_jpg)
                            video_path = os.path.join(alarm_package + alarm_video)
                            cv2.imwrite(jpg_path, img)
                            video_writer = cv2.VideoWriter(video_path, self.fourcc, 25, self.resize)
                            self.alarm_json['ALARM'] = [video_writer, datetime.datetime.now(), jpg_path, video_path]
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
                                        'types': 'cloths'
                                    }
                                )
                                self.alarm_json.pop('ALARM')

                    #print(time.ctime() + 'it-着装 costs %s' % ((datetime.datetime.now() - tx) * 1000), 'ms.')
                except Exception as E4R:
                    #print(('着装监控报错'+time.ctime() + str(E4R) + '\n') * 10)
                    time.sleep(0.5)
                    continue

    # def set_state(self, state):
    #     self.state = state

    # def get_output(self):
    #     return self.output_img

    def __del__(self):
        pass


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
            time.sleep(0.5)
            try:
                img = reader.readFrame(pixel_format=FourCC.BGR24, width=img_size[0], height=img_size[1])
                self.Q.put(img)
                #print(time.ctime()+'着装监控frame队列长度===>',self.Q.qsize())
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


def get_time_stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_time = time.strftime("%Y%m%d_%H:%M:%S", local_time)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    stamp = ("".join(time_stamp.split()[0].split("-")) + "".join(time_stamp.split()[1].split(":"))).replace('.', '')
    return data_time, stamp


# if __name__ == '__main__':
#     from threading import Thread
#
#     DP = Detect_People_Thread(True)
#     T = Thread(target=DP.loop, args=())
#     T.setDaemon(True)
#
#     T.start()
    # while True:
    #     _img =
    # time.sleep(1000)
    # DP.loop()
    # check_draw_circle(video=cfg.People_Detect_video)
