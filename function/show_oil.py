# coding: utf-8
import cv2
import numpy as np
from collections import namedtuple
import mxnet as mx
import os
import time,uuid
import config as cfg
import argparse
from util import AlgorithmBase

import datetime
import uuid
from tensor_stream import TensorStreamConverter, FourCC
import queue
import threading

from PIL import Image, ImageDraw, ImageFont


# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def arg_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', default='/home/tiger/works/workspace/oil_discharge/test/demo3.mp4', help='')
    parser.add_argument('--batch_size', default=1, help='')
    parser.add_argument('--image_height', default=224, help='')
    parser.add_argument('--image_width', default=224, help='')
    parser.add_argument('--anni_model', default=cfg.oil_model['anni'], help='')
    parser.add_argument('--pipe_model', default=cfg.oil_model['pipe'], help='')
    parser.add_argument('--el_model', default=cfg.oil_model['el'], help='')

    return parser.parse_args()


def get_predict(sym_dict):
    ctx = mx.gpu()

    for key in sym_dict:
        Batch = namedtuple('Batch', ['data'])
        # 定义要测试的图片
        # cv2.imshow('img', sym_dict[key][2])
        # cv2.waitKey(200)
        img = cv2.cvtColor(sym_dict[key].pop(-1), cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (120, 256))
        img = img - 127.5
        input_image = img[:, :, :, np.newaxis]
        input_image = input_image.transpose([3, 2, 0, 1])
        sym_dict[key][1].forward(Batch([mx.nd.array(input_image, ctx)]))
        prob = sym_dict[key][1].get_outputs()[0].asnumpy()
        prob = np.squeeze(prob)

        sym_dict[key][0].append(np.argmax(prob))


class Manager(AlgorithmBase):
    def __init__(self, save_flag,params1):
        super(AlgorithmBase, self).__init__()

        self.output_img = None
        self.save_flag = save_flag
        self.state = True
        self.fourcc=828601953
        self.input_Q = queue.Queue(cfg.queue_lenth)

        self.remember_json = {}
        self.alarm_json = {}
        self.resize = (960, 540)
        self.output_Q = params1

        self.prompt_message = {}
        assert isinstance(self.prompt_message, dict)
        # self.prompt_message['anni_warning'] = 'Waring: The Annihilator is not found!!!'
        # self.prompt_message['pipe_warning'] = 'Waring: The oil_Pipe is not connected!!!'
        # self.prompt_message['el_warning'] = 'Waring: The E-line is not connected!!!'
        self.prompt_message['anni_warning'] = '灭火器不存在,请注意防范!!!'
        self.prompt_message['pipe_warning'] = '油管不存在,请注意防范!!!'
        self.prompt_message['el_warning'] = '接地线不存在,请注意防范!!!'
        self.prompt_message['anni_status'] = '存在'
        self.prompt_message['pipe_status'] = '存在'
        self.prompt_message['el_status'] = '存在'

        self.sym_dict = {}
        ctx = mx.gpu()
        assert isinstance(self.sym_dict, dict)

        self.sym_dict.setdefault('anni', [])
        self.sym_dict.setdefault('pipe', [])
        self.sym_dict.setdefault('el', [])

        self.sym_dict['anni'].append(list(mx.model.load_checkpoint(cfg.oil_model['anni'], 10)))
        self.sym_dict['pipe'].append(list(mx.model.load_checkpoint(cfg.oil_model['pipe'], 10)))
        self.sym_dict['el'].append(list(mx.model.load_checkpoint(cfg.oil_model['el'], 10)))

        for key in self.sym_dict:
            # 为载入的模型构建模板
            mod = mx.mod.Module(symbol=self.sym_dict[key][0][0], context=ctx, label_names=None)

            # mod.bind 是在显卡上分配所需的显存, 所以需要把data_shapes label_shapes 传给它, 分显存没有tf流氓
            mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))],
                     label_shapes=mod._label_shapes)

            # 然后定义网络的参数, 开始训的时候是初始化网络参数 如: mod.init_params()
            mod.set_params(self.sym_dict[key][0][1], self.sym_dict[key][0][2], allow_missing=True)

            self.sym_dict[key].append(mod)

    # def addchinese(frame, chin, font_color=(0, 0, 255)):
    #     frame_PIL = Image.fromarray(frame)
    #     draw = ImageDraw.Draw(frame_PIL)
    #     # font_path = os.path.join(cfg.pwd,'util/guanjiaKai.ttf')
    #     # logging.info('font_path:%s.' % font_path)
    #     selectedFont = cfg.selectedFont
    #     draw.text((0, 0), chin, font_color, font=selectedFont)
    #     return np.array(frame_PIL)


    def addchinese222(self,image, strs, locals, sizes, colour=(0, 0, 255)):
        # try:
        #     cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #     pilimg = Image.fromarray(cv2img)
        #     draw = ImageDraw.Draw(pilimg)  # 图片上打印
        #     font = ImageFont.truetype(cfg.selectedFont, sizes, encoding="utf-8")
        #     draw.text(locals, strs, colour, font=font)
        #     image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
        #     return image
        # except Exception as e:
        #     print("?????",e)
        try:
            cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pilimg = Image.fromarray(cv2img)
            draw = ImageDraw.Draw(pilimg)  # 图片上打印
            # print(11111111111111111111111)
            font = ImageFont.truetype(cfg.font_path, sizes, encoding="utf-8")
            # print(222222222222222222222)
            draw.text(locals, strs, colour, font=font)
            # print(3333333333333333333)
            image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
            # print(44444444444)
            # cv2.imshow("isdasad", cv2.resize(image, (1920, 1080)))
            # cv2.waitKey()
            # cv2.imshow("isdasad", cv2.resize(image, (1920, 1080)))
            # cv2.waitKey("1")
            return image
        except Exception as e:
            print("文字写入图片错误",e)
            return image


    def loop(self, threshold=7):

        anni_notes, pipe_notes, el_notes = [], [], []
        anni_count, pipe_count, el_count = 0, 0, 0

        i, j = 0, 0

        # 这里直接改成读文件就行
        with open(os.path.join(cfg.config_path, 'oil_config.txt'), mode='r') as f:
            anni_area, pipe_area, el_area = [list(map(int, line.split(' '))) for line in f.readlines()]

        anni_a, anni_b = tuple(anni_area[:2]), tuple(anni_area[2:])
        pipe_a, pipe_b = tuple(pipe_area[:2]), tuple(pipe_area[2:])
        el_a, el_b = tuple(el_area[:2]), tuple(el_area[2:])

        push_get = threading.Thread(target=PUSHER(self.input_Q,cfg.oil_video,).Image_pusher)
        push_get.setDaemon(True)
        push_get.start()

        while True:
            frame = self.input_Q.get()

            if self.state == True:
                i += 1
                # print(i)
                try:
                    try:
                        test_images = frame.copy()
                        # cv2.imshow("isdasad", cv2.resize(test_images, (1920, 1080)))
                        # cv2.waitKey("1")
                    except:
                        continue

                    frame = cv2.rectangle(frame, anni_a[::-1], anni_b[::-1], (0, 0, 255), 2)
                    frame = cv2.rectangle(frame, pipe_a[::-1], pipe_b[::-1], (0, 0, 255), 2)
                    frame = cv2.rectangle(frame, el_a[::-1], el_b[::-1], (0, 0, 255), 2)

                    anni_image = test_images[anni_area[0]: anni_area[2], anni_area[1]: anni_area[3], :]
                    pipe_image = test_images[pipe_area[0]: pipe_area[2], pipe_area[1]: pipe_area[3], :]
                    el_image = test_images[el_area[0]: el_area[2], el_area[1]: el_area[3], :]
                    self.sym_dict['anni'].append(anni_image)
                    self.sym_dict['pipe'].append(pipe_image)
                    self.sym_dict['el'].append(el_image)

                    # 推理 获取预测结果
                    get_predict(self.sym_dict)
                    anni_notes.append(self.sym_dict['anni'][0].pop(3))
                    pipe_notes.append(self.sym_dict['pipe'][0].pop(3))
                    el_notes.append(self.sym_dict['el'][0].pop(3))

                    anni_count += anni_notes[-1]
                    pipe_count += pipe_notes[-1]
                    el_count += el_notes[-1]

                    if len(anni_notes) == 10:
                        if anni_count < threshold:
                            self.prompt_message['anni_status'] = '不存在'

                        else:
                            self.prompt_message['anni_status'] = '存在'
                        anni_count = anni_count - anni_notes.pop(0)

                    if len(pipe_notes) == 10:
                        if pipe_count < threshold:
                            self.prompt_message['pipe_status'] = '不存在'

                        else:
                            self.prompt_message['pipe_status'] = '存在'
                        pipe_count = pipe_count - pipe_notes.pop(0)

                    if len(el_notes) == 10:
                        if el_count < threshold:
                            self.prompt_message['el_status'] = '不存在'

                        else:
                            self.prompt_message['el_status'] = '存在'
                        el_count = el_count - el_notes.pop(0)

                    # if frame is not None:
                    # self.output_img = frame
                    # time.sleep(10)

                    # 　:           标注报警内容英文
                    #todo: 标注报警内容为中文.
                    if len(anni_notes) >= 9 and self.prompt_message['anni_status'] == '不存在':
                        # frame = cv2.putText(frame, self.prompt_message['anni_warning'], (1200, 100),
                        #                     cv2.FONT_HERSHEY_SIMPLEX, 1,
                        #                     [0, 0, 255], 2)
                        frame = self.addchinese222(frame, self.prompt_message["anni_warning"], (800,100), 30, (255, 0, 0))

                    if len(pipe_notes) >= 9 and self.prompt_message['pipe_status'] == '不存在':
                        # frame = cv2.putText(frame, self.prompt_message['pipe_warning'], (1200, 130),
                        #                     cv2.FONT_HERSHEY_SIMPLEX, 1,
                        #                     [0, 0, 255], 2)
                        frame = self.addchinese222(frame, self.prompt_message["pipe_warning"], (800, 140), 30, (255, 0, 0))

                    if len(el_notes) >= 9 and self.prompt_message['el_status'] == '不存在':
                        # frame = cv2.putText(frame, self.prompt_message['el_warning'], (1200, 160),
                        #                     cv2.FONT_HERSHEY_SIMPLEX,1,
                        #                     [0, 0, 255], 2)
                        frame = self.addchinese222(frame, self.prompt_message["el_warning"], (800, 180), 30, (255, 0, 0))

                    # default_anni_text = 'Annihilator state: '
                    default_anni_text = '灭火器状态: '
                    default_pipe_text = '油管状态: '
                    default_el_text = '接地线状态: '

                    # frame = cv2.putText(frame, default_anni_text, (100, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 0],
                    #                     3)
                    #     三个参数: cap,"大撒旦撒多撒撒旦法撒旦",(100,140),18
                    # print(frame)
                    frame = self.addchinese222(frame, default_anni_text, (100, 100), 30)
                    # cv2.imshow("isdasad", cv2.resize(frame, (1920, 1080)))
                    # cv2.waitKey("1")

                    # frame = cv2.putText(frame, default_pipe_text, (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 0],
                    #                     3)
                    frame = self.addchinese222(frame, default_pipe_text, (100,140), 30)

                    # cv2.imshow("isdasad", cv2.resize(frame, (1920, 1080)))
                    # cv2.waitKey("1")
                    # frame = cv2.putText(frame, default_el_text, (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 0],
                    #                     3)
                    frame = self.addchinese222(frame, default_el_text, (100,180), 30)
                    # cv2.imshow("isdasad", cv2.resize(frame, (1920, 1080)))
                    # cv2.waitKey("1")


                    if self.prompt_message['anni_status'] == "不存在":
                        # frame = cv2.putText(frame, self.prompt_message['anni_status'], (400, 140),
                        #                     cv2.FONT_HERSHEY_SIMPLEX, 1,
                        #                     [0, 0, 255], 3)
                        frame = self.addchinese222(frame, self.prompt_message['anni_status'], (350, 100), 30, (255, 0, 0))
                    else:
                        # frame = cv2.putText(frame, self.prompt_message['anni_status'], (400, 140),
                        #                     cv2.FONT_HERSHEY_SIMPLEX, 1,
                        #                     [0, 255, 0], 3)
                        frame = self.addchinese222(frame, self.prompt_message['anni_status'], (350, 100), 30, (0, 255, 0))

                    if self.prompt_message['pipe_status'] == "不存在":
                        # frame = cv2.putText(frame, self.prompt_message['pipe_status'], (400, 180),
                        #                     cv2.FONT_HERSHEY_SIMPLEX, 1,
                        #                     [0, 0, 255], 3)
                        frame = self.addchinese222(frame, self.prompt_message['pipe_status'], (350, 140), 30, (255, 0, 0))
                    else:
                        frame = self.addchinese222(frame, self.prompt_message['pipe_status'], (350, 140), 30, (0, 255, 0))

                    if self.prompt_message['el_status'] == "不存在":
                        # frame = cv2.putText(frame, self.prompt_message['el_status'], (400, 220),
                        #                     cv2.FONT_HERSHEY_SIMPLEX,1,
                        #                     [0, 0, 255], 3)
                        frame = self.addchinese222(frame, self.prompt_message['el_status'], (350, 180), 30, (255, 0, 0))
                    else:
                        # frame = cv2.putText(frame, self.prompt_message['el_status'], (400, 220),
                        #                     cv2.FONT_HERSHEY_SIMPLEX, 1,
                        #                     [0, 255, 0], 3)
                        frame = self.addchinese222(frame, self.prompt_message['el_status'], (350, 180), 30, (0, 255, 0))
                    # cv2.imshow("isdasad", cv2.resize(frame, (960, 540)))
                    # cv2.waitKey("1")

                    # if frame is not None:
                    self.output_img = frame

                    time.sleep(1)
                    # print(frame)

                    if self.prompt_message['el_status'] == '不存在' or \
                            self.prompt_message['pipe_status'] == '不存在' or \
                            self.prompt_message['anni_status'] == '不存在':

                        if 'ALARM' not in self.alarm_json:

                            '''Control warnings for every miniute.'''
                            curr_time = time.time()

                            if 'ING' not in self.remember_json:
                                self.remember_json['ING'] = curr_time
                            elif curr_time - self.remember_json['ING'] > 30:
                                self.remember_json['ING'] = curr_time
                            else:
                                continue
                            alarm_package = os.path.join(
                                os.environ['HOME'] + "/AIResult" + '/OilResult/' + str(time.ctime()) + str(uuid.uuid1()) + '/')
                            if not os.path.exists(alarm_package):
                                os.makedirs(alarm_package)

                            alarm_jpg = str(uuid.uuid1()) + '.jpg'
                            alarm_video = str(uuid.uuid1()) + '.mp4'
                            jpg_path = os.path.join(alarm_package + alarm_jpg)
                            video_path = os.path.join(alarm_package + alarm_video)
                            cv2.imwrite(jpg_path, frame)
                            insk = ""
                            if self.prompt_message['anni_status'] == '不存在':
                                insk += '1,'
                            if self.prompt_message['pipe_status'] == '不存在':
                                insk += '2,'
                            if self.prompt_message['el_status'] == '不存在':
                                insk += '3,'
                            insk = insk.rstrip(",")
                            # print("==============\n", insk)
                            video_writer = cv2.VideoWriter(video_path, self.fourcc, 1, self.resize)
                            self.alarm_json['ALARM'] = [video_writer, datetime.datetime.now(), jpg_path, video_path, insk]
                            # frame =
                            self.alarm_json['ALARM'][0].write(cv2.resize(frame, self.resize))

                        else:
                            self.alarm_json['ALARM'][0].write(cv2.resize(frame, self.resize))
                            # print("报警视频间隔秒数是:",(datetime.datetime.now() - self.alarm_json['ALARM'][1]).seconds,"序号是:",i)
                            if (datetime.datetime.now() - self.alarm_json['ALARM'][1]).seconds > 25:
                                self.alarm_json['ALARM'][0].release()
                                self.output_Q.put(
                                    {
                                        'alarm_jpg': self.alarm_json['ALARM'][2],
                                        'alarm_video': self.alarm_json['ALARM'][3],
                                        'alarm_time': self.alarm_json['ALARM'][1],
                                        'alarm_index': str(uuid.uuid1()),
                                        'types': 'unload_area',
                                        'type_flag': self.alarm_json['ALARM'][4],
                                    }
                                )
                                self.alarm_json.pop('ALARM')
                    # if frame is not None:
                    #     #print(time.ctime() + "卸油区存储一帧图片时间**************** It's costs time seconds is %d s." % (datetime.datetime.now() - tx).seconds)

                    # if self.save_flag:
                    #     self.output_movie.write(frame)

                    # if frame is not None:
                    #     # self.output_img = frame
                    #     self.output_img = frame.copy(
                    # #print(time.ctime() + 'it-卸油区 costs %s' % ((time.time() - t0) * 1000), 'ms.')
                except Exception as E4R:
                    print(('卸油区报错'+time.ctime()+str(E4R)+'\n'),i,frame)
                    cv2.imshow(i, cv2.resize(frame, (960, 540)))
                    cv2.waitKey("1")
                    time.sleep(0.5)
                    continue

    def set_state(self, state):
        """
        Set status for working thread.
        """
        self.state = state

    def get_output(self):
        return self.output_img

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
            # time.sleep(1)
            try:
                img = reader.readFrame(pixel_format=FourCC.BGR24, width=img_size[0], height=img_size[1])
                self.Q.put(img)
                #print(time.ctime()+'卸油区检测frame队列长度===>',self.Q.qsize())
                if self.Q.qsize() >= 99:
                    self.Q.queue.clear()

                # assert img is not None
            except Exception as e:
                print("流媒体转码失败",e)
                wrong += 1
                if wrong > 5:
                    #print('--------------------------------------------CameraID:%s stream fialed.' % self.video_stream)
                    self.running = False
                    break
                else:
                    continue

if __name__ == '__main__':

    pass
