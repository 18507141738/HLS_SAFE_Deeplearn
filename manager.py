# coding: utf-8
import config as cfg
import threading as T
# from function import PhoneManager, OILManager, ClothManager, FireManager
from function import PhoneManager, OILManager, ClothManager, FireManager
import time
import numpy as np
import cv2

import queue
import threading as td
import random
import string
from pysm4 import encrypt_ecb
import base64
import json
import hashlib
import requests
import datetime



'''
控制器提供全算法显示功能。具体功能如下：
显示功能共分5种模式，即4个小窗口模式和每种算法布满全屏幕模式，编号如下
1 抽烟打电话 （左上角）
2 火检测     （右上角）
3 着装识别   （左下角）
4 卸油区     （右下角）
窗口分布：
 1 2
 3 4

摁下’1‘～’4‘键，放大其中一个显示的窗口至全屏幕。
摁下‘q’/'0'键，退回至4宫格模式。
摁下’Esc‘键，退出显示，算法结束。
'''
class Public(object):

    def __init__(self):
        self.header = {'Content-Type': 'application/x-www-form-urlencoded'}
        self.appids = "cnpczhhthd"
        self.signs_key = 'CNPC99CDZ6782c!^'
        self.oil_key = 'KeF8U9cDzHEfs7Q4'  # 密钥长度小于等于16字节
        self.remoteServerOIL = "http://jlapp.95504.net/ssttra/device/sstSvAlarmInfo4External/saveSvAlarmInfo"
        # self.remoteServerOIL = "http://10.30.225.133:9999/device/sstSvAlarmInfo4External/saveSvAlarmInfo"
        # self.remoteServerOIL = "http://192.168.10.156:8887/distribute/oilunload_time_save"

    def _md5Sum(self, params):
        # times = time.time()
        # print("加密时间是：",str(times))
        # HashStr = self.APPID+str(times)+self.APPKEY
        HashStr = params
        # print("加密前内容是: ",HashStr)
        mm = hashlib.md5()
        mm.update(HashStr.encode("UTF-8"))

        return mm.hexdigest()

    def XML_Process(self, alarm_id, paramsA, paramsB, rtmp, alarm_type, alarm_msg, alarm_location,
                     alarm_lavel, alarm_time, alarm_system,type_flag):
        print("------------------------------------SM4_POST_START!!!------------------------------------")
        '''
        :params:接收报警参数
        :return:None,进入其他处理流程 ------行为科技对接新增违章记录接口
        '''

        remote_server = self.remoteServerOIL

        # print("接入参数是：", remote_server, alarm_id, paramsA, paramsB, rtmp, alarm_type, alarm_msg, alarm_location,
        #       alarm_lavel, alarm_time, alarm_system,type_flag)

        xss = self.random_str()
        print("----------加密之后的32位随机字符是--->>> %s------------------------------------" % xss)

        IMAGE = self._encode(paramsA)
        VIDEO = self._encode(paramsB)

        params_config = {
            "deviceNo": "SP1870658", #'0001',
            "standardCode": 'KKKK',
            "locationNo": alarm_location,
            "eventType": alarm_type,
            "eventNo": "KKKK"+time.strftime("%Y%m%d%H%M%S", time.localtime())+"000000",  #   todo: (站编号+时间戳+6位随机数)
            "priority": alarm_lavel,
            "cameraNo": rtmp,
            "info": str(alarm_msg),
            "eventTime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "evidenceImg": IMAGE,
            "evidenceVideo": VIDEO,
            "eventSubType": type_flag,
        }

        print("==============================")
        # print(params_config['eventSubType'])
        # print(params_config['locationNo'])
        print(params_config['eventType'])
        # print(params_config['eventTime'])
        # print(params_config['cameraNo'])
        # print(paramsA)
        # print(paramsB)
        # print(params_config["eventNo"])
        # print("==============================")

#<?xml version='1.0' encoding='utf-8'?>
        tradeDatas = "<TradeData><MonitorHeader><GLMsgId></GLMsgId><TransID>24684462</TransID><TransTime>2019-08-16 10:35:26</TransTime><BizKey>111</BizKey><ServiceType></ServiceType><Caller>02</Caller></MonitorHeader><ServiceCode>01002</ServiceCode><Message>%s</Message></TradeData>" % params_config

        # aes = self._SM4_Process(tradeDatas)
        aes = tradeDatas
        strs = 'tradeData=%s&appid=%s&randomchar=%s&secretkey=%s' % (aes, self.appids, 'a'*32, self.signs_key)

        # print('AAA:',strs)

        md5_signs = self._md5Sum(strs)

        # print('BBB', md5_signs)

        # print("signs md5内容是--->>> %s" % md5_signs)
        body = {
            "tradeData": aes,  # 业务数据，接口请求报文，必须加密（加密采用采用AES/CBC/PKCS5Padding+base64位加密模式）
            "appid": self.appids,  # 系统标识, B系统接入的授权appid，上线前由A系统指定。
            "randomchar": 'a'*32,  # 随机字符串32位
            "sign": md5_signs,  # 使用MD5对传输参数进行签名，签名字串如下：
        }

        try:
            reta = ""
            # print('开始调用～')
            to = datetime.datetime.now()

            print('请求地址是：', self.remoteServerOIL)

            time.sleep(0.05)
            # x = json.dumps(body)
            try:
                reta = requests.post(self.remoteServerOIL, headers=self.header, data=body, timeout=9)
            except Exception as e:
                print("报警接收端返回异常! ===> %s" % e)
            '''
            返回值
            {
            "resultHeader": {
                "resultCode": "0",
                "resultMsg": "5L+d5a2Y5oiQ5Yqf77yB"
            },
            "message": null,
            "sign": "3b9080ca9d5f202b9703c8b220072f24"
            }
            '''
            # print('调用成功～')
            if not reta:
                print('返回失败～')
            # print(reta.text,'==============')
            # if isinstance(reta,dict) and reta["resultHeader"]["resultCode"] == "0":
            #
            #     print("返回成功~", reta["resultHeader"]["resultCode"],)

            # print('调用时长：',time.time()-t0 ,'ms.')
            print('调用时长：', (datetime.datetime.now() - to).seconds, 's.')

        except Exception as e:
            print("请求远程服务器错误:", e)

        print("------------------------------------SM4_POST_Stop---------------------------------------")
        return

    def _encode(self,params):

        if len(params) <= 6:
            return ""
        with open(params, 'rb') as f:
            base64_data = base64.b64encode(f.read())
            s = base64_data.decode()
            # print('data:image/jpeg;base64,%s' % s)
            # with open("state.txt","a") as f:
            #     f.write(s)
            # print(type(s),"********")
        # time.sleep(0.05)
        return s

    def random_str(self):
        '''
        :return:生成32位随机字符串
        '''
        # 存储大小写字母和数字，特殊字符列表

        STR = [chr(i) for i in range(65, 91)]  # 65-91对应字符A-Z
        str = [chr(i) for i in range(97, 123)]  # a-z
        number = [chr(i) for i in range(48, 58)]  # 0-9

        # 特殊字符串列表获取有点不同
        initspecial = string.punctuation  # 这个函数获取到全部特殊字符，结果为字符串形式
        special = []  # 定义一个空列表

        # 制作特殊符号列表
        for i in initspecial:
            special.append(i)

        total = STR + str + number #+ special

        insi = ''.join(random.sample(total, int('32')))
        # print("\033[32m生成的\033[0m" + '\033[32m32\033[0m' + "\033[32m位随机字符密码为:\033[0m\n" + insi, len(insi))
        return insi

    def _SM4_Process(self,params):
        # print('---------------SM4 ECB 加密中---------- %s' % str(params))
        # 明文
        plain_text0 = params
        # 加密
        cipher_text0 = encrypt_ecb(plain_text0, self.oil_key)

        return cipher_text0


class Oil_Phone_Fire_Cloths_Output(object):
    '''
        oil output queue to other sdk
    '''
    def __init__(self,oil_Q):
        self.oil_Q = oil_Q

    def loop(self):
        index = 0
        while True:

            time.sleep(0.5)

            signs = self.oil_Q.empty()

            if signs == True:
                print(time.ctime()+'ALARM output_Q is None ...%d' % index)
                time.sleep(3)
                continue

            else:
                time.sleep(1.5)
                index += 1

                # print(time.ctime() + 'ALARM output_Q is Translating ...%d' % index)
                print(time.ctime() + '报警队列长度是:%d' % self.oil_Q.qsize())
                if self.oil_Q.qsize() >= 90:
                    self.oil_Q.get()

                # print('oil out putting ...%d' % index)
                poila = self.oil_Q.get()
                # print(('%dALARM>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n' % index) * 10, poila)
                '''alarm_id, paramsA, paramsB, rtmp, alarm_type, alarm_msg, alarm_location,
                         alarm_lavel, alarm_time, alarm_system)'''
                if index == 1:
                    pass
                else:
                    if poila['types'] == 'unload_area':
                        Public().XML_Process(
                            index,
                            poila['alarm_jpg'],
                            poila['alarm_video'],
                            cfg.oil_video,
                            'unload_area',
                            'Oil unloading is not standard ,please note.',
                            '04',
                            '20',
                            poila['alarm_time'],
                            'XINGWEI-SYSTEM-2019',
                            poila['type_flag'],
                        )
                    if poila['types'] == 'phone_smoke':
                        Public().XML_Process(
                            str(index),
                            poila['alarm_jpg'],
                            poila['alarm_video'],
                            cfg.phone_video,
                            "phone_smoke",
                            'Please deal with Phone call or smoking immediately.',
                            '01',
                            '20',
                            poila['alarm_time'],
                            'XINGWEI-SYSTEM-2019',
                            poila['type_flag'],
                        )
                    elif poila['types'] == 'fire':
                        Public().XML_Process(
                            str(index),
                            poila['alarm_jpg'],
                            poila['alarm_video'],
                            cfg.fire_video,
                            "fire",
                            'There are unknown fireworks,please deal with them immediately.',
                            '03',
                            '10',
                            poila['alarm_time'],
                            'XINGWEI-SYSTEM-2019',
                            "0",
                        )
                    elif poila['types'] == 'cloths':
                        Public().XML_Process(
                            str(index),
                            poila['alarm_jpg'],
                            poila['alarm_video'],
                            cfg.People_Detect_video,
                            "cloths",
                            'Please not that unidentified person stay.',
                            '02',
                            '20',
                            poila['alarm_time'],
                            'XINGWEI-SYSTEM-2019',
                            "0",
                        )
                    else:
                        pass


class MyThread(T.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        time.sleep(2)
        self.result = self.func(*self.args)

    def get_result(self):
        T.Thread.join(self)  # 等待线程执行完毕
        try:
            return next(self.result)
        except Exception:
            return None


class Manager(object):
    def __init__(self):

        '''
            Add main_workers_dict images docker
        '''
        self.net_alarm_Q = queue.Queue(cfg.queue_lenth)  # out_put_alarm

        '''
            Add SystemAlarm_API to other sdk
        '''
        # self.FuncDocker = ['oil', 'phone', 'fire', 'cloths']
        # for insk in self.FuncDocker:
        #     # 为了保证报警实时性,故每个算法单独API报警
        #     self.net_alarm_Q_dict[insk] = queue.Queue(cfg.queue_lenth)

            # time.sleep(1.5)

        Oil_Phone_Fire_Cloths = td.Thread(target=Oil_Phone_Fire_Cloths_Output(self.net_alarm_Q).loop)
        Oil_Phone_Fire_Cloths.setDaemon(True)
        Oil_Phone_Fire_Cloths.start()

        self.frame_shape = cfg.frame_shape
        self.subframe_shape = cfg.subframe_shape

        self.cellphone_img = None
        self.fire_img = None
        self.cloth_img = None
        self.oil_img = None

        self.state = 0
        self.new_state = 0

        self.oilManager = OILManager(False, self.net_alarm_Q)
        self.phoneManager = PhoneManager(False, self.net_alarm_Q)
        self.fireManager = FireManager(False, self.net_alarm_Q)
        self.clothManager = ClothManager(False, self.net_alarm_Q)

        self.oil_thread = T.Thread(target=self.oilManager.loop)
        self.phone_thread = T.Thread(target=self.phoneManager.loop)
        self.fire_thread = T.Thread(target=self.fireManager.loop)
        self.cloth_thread = T.Thread(target=self.clothManager.loop)

        self.oil_thread.setDaemon(True)
        self.phone_thread.setDaemon(True)
        self.fire_thread.setDaemon(True)
        self.cloth_thread.setDaemon(True)


    def four_image(self):
        self.oil_img = cv2.resize(self.oil_img, (int(self.frame_shape[0] / 2), int(self.frame_shape[1] / 2)),
                                  interpolation=cv2.INTER_CUBIC)

        self.cellphone_img = cv2.resize(self.cellphone_img,
                                        (int(self.frame_shape[0] / 2), int(self.frame_shape[1] / 2)),
                                        interpolation=cv2.INTER_CUBIC)
        self.fire_img = cv2.resize(self.fire_img, (int(self.frame_shape[0] / 2), int(self.frame_shape[1] / 2)),
                                   interpolation=cv2.INTER_CUBIC)
        self.cloth_img = cv2.resize(self.cloth_img, (int(self.frame_shape[0] / 2), int(self.frame_shape[1] / 2)),
                                    interpolation=cv2.INTER_CUBIC)

        # 初始化一个零矩阵
        frame = np.zeros(shape=(self.frame_shape[1], self.frame_shape[0], 3), dtype=np.uint8)

        frame[0: self.subframe_shape[1], 0: self.subframe_shape[0], :] = self.oil_img

        frame[0: self.subframe_shape[1], self.subframe_shape[0]: self.subframe_shape[0] * 2, :] = self.cellphone_img

        frame[self.subframe_shape[1]: self.subframe_shape[1] * 2, 0: self.subframe_shape[0], :] = self.cloth_img

        frame[self.subframe_shape[1]: self.subframe_shape[1] * 2, self.subframe_shape[0]: self.subframe_shape[0] * 2,
        :] = self.fire_img
        '''将几个图像拼接起来
        frame_up = np.hstack((self.oil_img, self.cellphone_img))
        frame_down = np.hstack((self.fire_img, self.cloth_img))
        frame = np.vstack((frame_up, frame_down))
        '''
        return frame

    def change_state(self, state=0):
        '''state int 0/1/2/3/4'''

        if state == 1:
            frame = self.oil_img

        elif state == 2:
            frame = self.cellphone_img

        elif state == 3:
            frame = self.cloth_img

        elif state == 4:
            frame = self.fire_img

        else:
            frame = self.four_image()

        return frame

    def generate_img_show(self):
        '''
        get the images for showing

        :return: (ret,img)
        ret = True if a images for showing has been generated successfully!
        otherwise False
        '''

        while True:
            # 这一步是刚需
            self.oil_img = self.oilManager.get_output()
            self.cellphone_img = self.phoneManager.get_output()
            self.fire_img = self.fireManager.get_output()
            self.cloth_img = self.clothManager.get_output()

            if self.oil_img is None:
                self.oil_img = np.zeros(shape=(self.frame_shape[1], self.frame_shape[0], 3), dtype=np.uint8)
            if self.cellphone_img is None:
                self.cellphone_img = np.zeros(shape=(self.frame_shape[1], self.frame_shape[0], 3), dtype=np.uint8)
            if self.cloth_img is None:
                self.cloth_img = np.zeros(shape=(self.frame_shape[1], self.frame_shape[0], 3), dtype=np.uint8)
            if self.fire_img is None:
                self.fire_img = np.zeros(shape=(self.frame_shape[1], self.frame_shape[0], 3), dtype=np.uint8)

            # 在这里需要判断 这个状态 在 获取图像之前还需要判断一下状态 如果不显示其 那么set_state = False
            # TODO 非常严重的问题是 点击完之后 不立刻变成 将原本的切回来之后 才被设为 False, 再次设置后 才能变为 True
            if self.new_state != self.state:
                self.state = self.new_state
                if self.state == 0:
                    self.oilManager.set_state(True)
                    self.phoneManager.set_state(True)
                    self.clothManager.set_state(True)
                    self.fireManager.set_state(True)

                elif self.state == 1:
                    self.oilManager.set_state(True)
                    self.phoneManager.set_state(False)
                    self.clothManager.set_state(False)
                    self.fireManager.set_state(False)

                elif self.state == 2:
                    self.oilManager.set_state(False)
                    self.phoneManager.set_state(True)
                    self.clothManager.set_state(False)
                    self.fireManager.set_state(False)

                elif self.state == 3:
                    self.oilManager.set_state(False)
                    self.phoneManager.set_state(False)
                    self.clothManager.set_state(True)
                    self.fireManager.set_state(False)

                elif self.state == 4:
                    self.oilManager.set_state(False)
                    self.phoneManager.set_state(False)
                    self.clothManager.set_state(False)
                    self.fireManager.set_state(True)

            frame = self.change_state(self.state)

            if frame is not None:
                # 捕获键盘事件
                cv2.namedWindow('中国石油规划院', 0)
                # cv2.resizeWindow("中国石油规划院", 1920,1080)
                cv2.resizeWindow("中国石油规划院", 960, 540)
                cv2.imshow('中国石油规划院', frame)
                flag = cv2.waitKey(1)
                # 响应事件
                if flag == ord('q'):
                    self.new_state = 0
                elif flag == ord('1'):
                    self.new_state = 1
                elif flag == ord('2'):
                    self.new_state = 2
                elif flag == ord('3'):
                    self.new_state = 3
                elif flag == ord('4'):
                    self.new_state = 4

                if flag == 27:
                    exit(1)

            # 这里的状态 比如之前是 1, 现在如果没有改变的话 我要在没有任何运算的情况下保持 1

    def start(self):
        '''runing_threads'''

        # TODO 要实现单独展示某一算法, 停止其他算法 在循环的过程中 改变算法是否推理的状态
        self.phone_thread.start()
        self.oil_thread.start()
        self.cloth_thread.start()
        self.fire_thread.start()

        # 展示图像
        self.generate_img_show()

    def unit_start(self):
        '''unit test'''
        # cap = cv2.VideoCapture(cfg.test_video)
        # _, images = cap.read()

        # self.oil_thread.start()
        # self.phone_thread.start()
        self.fire_thread.start()
        # self.cloth_thread.start()

        while True:
            self.oil_img = self.oilManager.get_output()
            self.cellphone_img = self.phoneManager.get_output()
            self.cloth_img = self.clothManager.get_output()
            self.fire_img = self.fireManager.get_output()

            if self.fire_img is None:
                time.sleep(0.01)
            else:
                cv2.imshow('output', self.fire_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


if __name__ == '__main__':
    'Test a specific threading'
    M = Manager()
    M.start()
    # M.unit_start()
    # Public().XML_Process(
    #     '',
    #     1,
    #     '/home/xw/SmokePhoneResult/Mon Nov 18 18:42:04 20190f9b22c0-09f0-11ea-b0fe-99a982b9fc01/0f9b2806-09f0-11ea-b0fe-99a982b9fc01.jpg',
    #     '/home/xw/SmokePhoneResult/Mon Nov 18 18:42:04 20190f9b22c0-09f0-11ea-b0fe-99a982b9fc01/0f9b28b0-09f0-11ea-b0fe-99a982b9fc01.mp4',
    #     '1',
    #     1,
    #     '1',
    #     '04',
    #     '1',
    #     'sad',
    #     'sda',
    # )
