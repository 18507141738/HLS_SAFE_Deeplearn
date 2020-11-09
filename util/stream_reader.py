# ---coding: utf-8---
import os
import time
from tensor_stream import TensorStreamConverter, FourCC
import threading
import cv2
import config as cfg

class CameraImagesPusher(object):
    '''用于推送视频流服务'''
    def __init__(self,rtsp_add, camera_id):
        self.running = False
        self.camera_id = camera_id
        self.rtsp = rtsp_add
        self.function_Q_dict = {}
        self.SV = False
        self.tm = 0
        # self.lock_Q_dict = 0

    def push_stream_loop_sv(self):

        # 创建流媒体并初始化
        reader = TensorStreamConverter(self.rtsp, repeat_number=2)

        try:
            reader.initialize()
            reader.start()
            print('Start Stream Successfully!')
            img_size = reader.getFrameSize()
        except RuntimeError:
            print('Start channel Wrong happened in %s' % self.rtsp)
            reader.stop()
            reader.release()

        '''录制测试视频'''
        save_dir = os.environ['HOME'] + '/测试视频'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fourcc=828601953

        time_gap = 1.0
        _watchQ = True
        _watchCurrentTime = time.time()
        wrong = 0
        while self.running:
            time.sleep(1)
            if self.SV == False:
                self.tm = time.time()
                tm = str(round(self.tm * 1000))
                save_path = os.path.join(save_dir, self.rtsp[7:].replace('/','_') + '-' + tm + '.mp4')
                saveVideo = cv2.VideoWriter(save_path, fourcc, 25, (int(img_size[0]), int(img_size[1])))
                self.SV = True
            try:
                img = reader.readFrame(pixel_format=FourCC.BGR24, width=img_size[0], height=img_size[1])
                time_stamp = time.time()
                assert img is not None
                saveVideo.write(img)
            except:
                wrong += 1
                if wrong > 5:
                    print('--------------------------------------------CameraID:%s stream fialed.' % self.rtsp)
                    self.running=False
                    break
                else:
                    continue
            wrong = 0
            if time.time()-_watchCurrentTime<time_gap:
                _watchQ = False
            else:
                _watchCurrentTime = time.time()
                _watchQ = True
            # while self.lock_Q_dict == 1:
            #     print('wait------------------------------------------------------------------------------')
            #     time.sleep(0.001)
            # start = time.time()
            # if self.lock_Q_dict == 0:
            for Q in self.function_Q_dict.values():
                Q.put((img, self.camera_id, time_stamp))
            # print('It costs %f ms.' % ((time.time()-start)*1000))
            if _watchQ:
                # print('-------------------CameraID:%s,pusher input Q_len:%d,--------------------'% (self.camera_id,Q.qsize()))
                pass
            if time.time() - self.tm >= 60:
                saveVideo.release()
                self.SV = False
        reader.stop()
        try:
            while True:
                img = reader.readFrame(pixel_format=FourCC.BGR24, width=img_size[0], height=img_size[1])

        except Exception as e:
            pass
        reader.release()
        print('Stream from %s stopped.' % self.rtsp)

    def push_stream_loop(self):

        wrong = 0
        reader = TensorStreamConverter(self.rtsp, repeat_number=2)
        try:
            reader.initialize()
            reader.start()
            print('Start Stream Successfully!')
            img_size = reader.getFrameSize()
            # print(img_size,'2222222222222')
        except RuntimeError:
            print('Start channel Wrong happened in %s' % self.rtsp)
            reader.stop()
            reader.release()

        time_gap = 1.0
        _watchQ = True
        _watchCurrentTime = time.time()

        print(1111111111111111111)
        while self.running:
            time.sleep(3)
            try:
                img = reader.readFrame(pixel_format=FourCC.BGR24, width=img_size[0], height=img_size[1])
                time_stamp = time.time()
                assert img is not None
            except:
                wrong += 1
                if wrong > 5:
                    print('--------------------------------------------CameraID:%s stream fialed.' % self.rtsp)
                    self.running=False
                    break
                else:
                    continue
            wrong = 0
            # if self.lock_Q_dict == 0:
            for Q in self.function_Q_dict.values():
                Q.put((img, self.camera_id, time_stamp))

            if time.time()-_watchCurrentTime<time_gap:
                _watchQ = False
            else:
                _watchCurrentTime = time.time()
                _watchQ = True

            if _watchQ:
                print('-------------------CameraID:%s,pusher input Q_len:%d,--------------------'% (self.camera_id,Q.qsize()))

        reader.stop()
        try:
            while True:
                img = reader.readFrame(pixel_format=FourCC.BGR24, width=img_size[0], height=img_size[1])

        except Exception as e:
            pass
        reader.release()
        print('Stream from %s stopped.' % self.rtsp)

    def start(self):
        if not self.running:
            if cfg.VEDIOSAVE == False:
                self.running = True
                self.work_thread = threading.Thread(target=self.push_stream_loop, args=())
                self.work_thread.setDaemon(True)
                self.work_thread.start()

            elif cfg.VEDIOSAVE == True:
                self.running = True
                self.work_thread = threading.Thread(target=self.push_stream_loop_sv, args=())
                self.work_thread.setDaemon(True)
                self.work_thread.start()

    def kill(self):
        self.running = False

    def update_func(self,function_type,state,Q=None):
        '''
        :param function_type: 'boundary'
        :param state: True/False
        :param Q: Queue() needed for adding functions
        :return:
        '''
        # self.lock.acquire()
        # self.lock_Q_dict = 1
        # time.sleep(1)
        if state and Q is not None:
            self.function_Q_dict[function_type] = Q
            # return 1
        elif function_type in self.function_Q_dict:
            del(self.function_Q_dict[function_type])
            if self.function_Q_dict =={}:
                print('camera %s has been killed.' % self.camera_id)
                self.kill()
            # return 1
        else:
            print('Task not work.')
        # self.lock_Q_dict = 0