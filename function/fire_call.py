#!/usr/bin/python
# coding:utf8
import mxnet as mx
from collections import namedtuple
import cv2
import numpy as np
import time
import os
import logging
from util import AlgorithmBase

import datetime
import uuid
from tensor_stream import TensorStreamConverter, FourCC
import queue
import threading

import config as cfg

# 参数0表示第一个摄像头, rtsp://admin:q1w2e3r4@192.168.10.51:554
"""
model-1 为通过训练出来的模型 OK

model-2 为通过模型 model-1 的误报数据作为负样本训练出来的模型 

model-3 加入了误检样本为第三类 OK
"""


class Fire_Detecte(object):
    def __init__(self):
        self.ctx = mx.gpu(0)
        self.fire_model = cfg.fire_model
        self.fire_model_num = cfg.fire_model_num
        self.net_file = os.path.join(os.path.dirname(__file__), '../resources/fire-model-3/%s' % self.fire_model)
        sym, arg_params, aux_params = mx.model.load_checkpoint(self.net_file, self.fire_model_num)
        self.mod = mx.mod.Module(symbol=sym, context=self.ctx, label_names=None)
        self.mod.bind(for_training=False, data_shapes=[('data', (1, 3, 128, 128))],
                      label_shapes=self.mod._label_shapes)
        self.mod.set_params(arg_params, aux_params, allow_missing=True)
        self.Batch = namedtuple('Batch', ['data'])
        # #print("*********Fire_Detecte 构造")

    def Work(self, img_BGR):
        # img_BGR = cv2.imread(img_BGR)
        img_BGR = cv2.resize(img_BGR, (128, 128))
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        img = img - 127.5
        input_image = img[:, :, :, np.newaxis]
        input_image = input_image.transpose([3, 2, 0, 1])
        self.mod.forward(self.Batch([mx.nd.array(input_image, self.ctx)]))
        prob = self.mod.get_outputs()[0].asnumpy()
        prob = np.squeeze(prob)
        max_id = np.argmax(prob)
        return max_id


class Background_Segment(object):
    def __init__(self, kernel_size=3, reduction_ratio=.5, perimeterThreshold=120, topk=10, record_len=10, valid_num=8):
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        self.ratio = reduction_ratio
        self.perimeter_thr = perimeterThreshold
        self.F = Fire_Detecte()
        self.time, _ = get_time_stamp()
        if not os.path.exists("result/FireInfo"):
            os.mkdir("result/FireInfo")
        self.warning_path = os.path.join(os.path.abspath('result/'), 'FireInfo')
        self.out_txt = os.path.join(os.path.abspath('result/'),
                                    'FireInfo/fire_boxes_%s.txt' % self.time)  # 生成文件名最好加日期
        self.topk = topk
        # #print("*********************Background_Segment 构造")

        self.status_json = {'status':0}

    def Work(self, img_input):
        # start = time.time()
        frame = cv2.resize(img_input, None, fx=self.ratio, fy=self.ratio)

        # 对帧进行预处理，>>转灰度图>>高斯滤波（降噪：摄像头震动、光照变化）。
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
        # fgmask = self.fgbg.apply(gray_frame)

        fgmask = self.fgbg.apply(frame)
        diff = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)

        per_list = []

        # 显示矩形框：计算一幅图像中目标的轮廓
        image, contours, hierarchy = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        start = time.time()
        _contours = []
        for c in contours:
            perimeter = cv2.arcLength(c, True)
            # perimeter = cv2.contourArea(c, True)
            if perimeter > self.perimeter_thr:
                per_list.append(perimeter)
                # #print("******perimeter:", perimeter)
                _contours.append(c)

        # 周长跟区域对应
        Contour_list = zip(per_list, _contours)
        # #print("*****Contour_list1:", Contour_list)

        # 利用周长长短进行排序
        Contour_list = sorted(Contour_list, key=lambda x: x[0], reverse=True)

        #print(time.ctime()+'It costs %0.2f ms for 火焰检测-sorting!' % ((time.time() - start) * 1000))

        # for c in contours:
        # Contour_list = sorted(Contour_list, key=lambda x: x[0], reverse=False)
        # #print("*****Contour_list2:", Contour_list)

        for idx in range(min(self.topk, len(Contour_list))):
            _perimeter = Contour_list[idx][0]
            c = Contour_list[idx][1]
            # if perimeter > self.perimeter_thr:
            x, y, w, h = cv2.boundingRect(c)
            cropImg = frame[y:y + h, x:x + w, :]
            fire_bool = self.F.Work(cropImg)
            # #print(fire_bool)

            self.status_json['status'] = 0
            if fire_bool == 0:
                self.status_json['status'] = 1
                # 对误报图片进行记录
                _, frame_NUM = get_time_stamp()
                # cv2.imwrite('%s/fire_crop_%s.jpg' % (self.warning_path, frame_NUM), cropImg)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # cv2.imwrite('%s/fire_alarm_%s.jpg' % (self.warning_path, frame_NUM), frame)
                cv2.rectangle(img_input, (int(x / self.ratio), int(y / self.ratio)),
                              (int((x + w) / self.ratio), int((y + h) / self.ratio)), (0, 0, 255), 2)
                # lines_list = "fire_alarm_%s.jpg" % frame_NUM + ' 1' + ' 1' + ' ' + str(x) + ' ' + str(
                #     y) + ' ' + str(
                #     w) + ' ' + str(h) + '\n'
                # # _temp_img = frame_lwpCV.copy() #在未画的image上画框
                # with open(self.out_txt, 'a+') as out_f:
                #     out_f.writelines(lines_list)
                #     out_f.close()

        # cv2.imshow('contours', frame)
        # # cv2.imshow('dis', diff)
        # cv2.waitKey(1)
        # #print('Result: It costs %f ms.' % ((time.time() - start) * 1000.0))
        return img_input,self.status_json['status']


class Manager(AlgorithmBase):
    def __init__(self, save_flag,params1):
        super(AlgorithmBase, self).__init__()
        self.count = 0
        self.skip = cfg.video_skip
        self.video = cfg.fire_video
        self.B = Background_Segment()
        # self.camera = cv2.VideoCapture(self.video)  # 参数0表示第一个摄像头

        self.fourcc = 828601953
        self.input_Q = queue.Queue(cfg.queue_lenth)

        self.remember_json = {}
        self.alarm_json = {}
        self.resize = (960, 540)
        self.output_Q = params1

        # if (self.camera.isOpened()):  # 判断视频是否打开
        #     #print('Open')
        # else:
        #     #print('摄像头未打开')

        # 测试用,查看视频size
        # self.fps = int(self.camera.get(5))
        # self.size = (int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        #              int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # #print('size:' + repr(self.size))

        # 保存视频

        self.save_flag = save_flag
        self.state = True
        self.output_img = None

        # if self.save_flag:
        #     self.fourcc = cv2.VideoWriter_fourcc(*'MP42')
        #     self.save_video = cv2.VideoWriter('result/output_video/fire.avi', self.fourcc, self.fps, self.size)

    def loop(self):

        push_get = threading.Thread(target=PUSHER(self.input_Q, cfg.fire_video, ).Image_pusher)
        push_get.setDaemon(True)
        push_get.start()

        # time.sleep(2)

        while True:
            # time.sleep(0.05)
            # 读取视频流
            # ok, frame = self.camera.read()
            # if not ok:
            #     self.camera.release()
            #     if self.save_flag:
            #         self.save_video.release()
            #     break
            # time.sleep(0.3)


            frame = self.input_Q.get()
            self.count += 1
            if self.count % self.skip != 0:
                continue
            if self.state == True:
                #print('火检测正在运行')
                try:
                    frame, status = self.B.Work(frame)
                    # #print(status,frame)
                    self.output_img = frame
                    if status != 1:
                        if 'ALARM' in self.alarm_json:
                            # frame =
                            self.alarm_json['ALARM'][0].write(cv2.resize(frame, self.resize))
                            if (datetime.datetime.now() - self.alarm_json['ALARM'][1]).seconds > 10:
                                self.alarm_json['ALARM'][0].release()
                                self.output_Q.put(
                                    {
                                        'alarm_jpg': self.alarm_json['ALARM'][2],
                                        'alarm_video': self.alarm_json['ALARM'][3],
                                        'alarm_time': self.alarm_json['ALARM'][1],
                                        'alarm_index': str(uuid.uuid1()),
                                        'types': 'fire',
                                    }
                                )
                                self.alarm_json.pop('ALARM')


                    else:
                        # #print(11111111111111111, 'fire' * 20)
                        if 'ALARM' not in self.alarm_json:
                            # #print(2222222222)

                            '''Control warnings for every miniute.'''
                            curr_time = time.time()

                            if 'ING' not in self.remember_json:
                                self.remember_json['ING'] = curr_time
                            elif curr_time - self.remember_json['ING'] > 18:
                                self.remember_json['ING'] = curr_time
                            else:
                                continue
                            # #print(3333333333333)
                            alarm_package = os.path.join(
                                os.environ['HOME'] + "/AIResult"  + '/FireResult/' + str(time.ctime()) + str(uuid.uuid1()) + '/')
                            if not os.path.exists(alarm_package):
                                os.makedirs(alarm_package)
                            # #print(444444444444)
                            alarm_jpg = str(uuid.uuid1()) + '.jpg'
                            alarm_video = str(uuid.uuid1()) + '.mp4'
                            jpg_path = os.path.join(alarm_package + alarm_jpg)
                            video_path = os.path.join(alarm_package + alarm_video)
                            cv2.imwrite(jpg_path, frame)
                            video_writer = cv2.VideoWriter(video_path, self.fourcc, 25, self.resize)
                            self.alarm_json['ALARM'] = [video_writer, datetime.datetime.now(), jpg_path, video_path]
                            # frame =
                            self.alarm_json['ALARM'][0].write(cv2.resize(frame, self.resize))

                        else:
                            # frame =
                            self.alarm_json['ALARM'][0].write(cv2.resize(frame, self.resize))
                            if (datetime.datetime.now() - self.alarm_json['ALARM'][1]).seconds > 10:
                                self.alarm_json['ALARM'][0].release()
                                self.output_Q.put(
                                    {
                                        'alarm_jpg': self.alarm_json['ALARM'][2],
                                        'alarm_video': self.alarm_json['ALARM'][3],
                                        'alarm_time': self.alarm_json['ALARM'][1],
                                        'alarm_index': str(uuid.uuid1()),
                                        'types': 'fire',
                                    }
                                )
                                self.alarm_json.pop('ALARM')

                except Exception as E4R:
                    #print(('火焰检测报错'+time.ctime() + str(E4R) + '\n') * 10)
                    time.sleep(0.5)
                    continue

    def set_state(self, state):
        self.state = state

    def get_output(self):
        return self.output_img

    def __del__(self):
        pass


def get_time_stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_time = time.strftime("%Y%m%d_%H:%M:%S", local_time)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    stamp = ("".join(time_stamp.split()[0].split("-")) + "".join(time_stamp.split()[1].split(":"))).replace('.', '')
    return data_time, stamp


def fire_run(video):
    count = 0
    B = Background_Segment()
    # camera = cv2.VideoCapture("rtsp://admin:q1w2e3r4@192.168.10.53:554")  # 参数0表示第一个摄像头
    # camera = cv2.VideoCapture("/home/lijun/Videos/firemanyface_mv.avi")  # 参数0表示第一个摄像头
    camera = cv2.VideoCapture(video)  # 参数0表示第一个摄像头


    # 测试用,查看视频size
    fps = int(camera.get(5))
    size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #print('size:' + repr(size))

    # 保存视频
    # fourcc = cv2.VideoWriter_fourcc(*'MP42')
    # save_video = cv2.VideoWriter('output3.avi', fourcc, fps, size)

    while True:
        # 读取视频流
        ok, frame = camera.read()
        if not ok:
            camera.release()
            break
        count += 1
        if count % 1 != 0:
            continue

        frame = B.Work(frame)
        # save_video.write(frame)  # 对火焰的存储

    # save_video.release()
    # 释放资源并关闭窗口
    camera.release()
    cv2.destroyAllWindows()


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
                #print(time.ctime()+'火焰检测frame队列长度===>',self.Q.qsize())
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
    video = cfg.fire_video
    fire_run(video)
