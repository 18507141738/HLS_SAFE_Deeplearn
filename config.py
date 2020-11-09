# coding: utf-8
'''
将4个frame合成2x2形式，每个 subframe 的大小由 subframe_shape 定义
'''

import os
from PIL import ImageFont

frame_shape = (1920, 1080)
subframe_shape = (960, 540)
skip_time = 30  # 显示间隔时间 ms

phone_video = 'rtsp://192.168.10.30:8557/unload_area.mkv'#'rtsp://192.168.1.12:8557/smokephone.mkv' #'rtmp://127.0.0.1:1935/live?vhost=stream16/livestream'#rtsp://admin:q1w2e3r4@192.168.10.58:554'#'rtmp://mobliestream.c3tv.com:554/live/goodtv.sdp'
# phone_video = '/media/tiger/MySSD/石油规划院/2019-05-17/11.0.18.15_01_20190517135153329.mp4'
# phone_video = 'rtmp://192.168.10.120/live/livestreamse'
# phone_video = 'rtsp://admin:q1w2e3r4@192.168.10.51:554'
# phone_video = 'rtmp://127.0.0.1:1935/live?vhost=stream16/livestream'
# oil_video = 'rtmp://192.168.10.120/live/livestreamsb'
# oil_video = 'rtmp://127.0.0.1:1'
oil_video = 'rtsp://192.168.10.30:8557/unload_area.mkv'#'rtmp://127.0.0.1:1935/live?vhost=stream15/livestream'#'rtsp://admin:q1w2e3r4@192.168.10.58:554'#'rtmp://202.69.69.180:443/webcast/bshdlive-pc'#'rtmp://58.200.131.2:1935/livetv/hunantv'#0#'http://ivi.bupt.edu.cn/hls/cctv6hd.m3u8'#0#'http://ivi.bupt.edu.cn/hls/cctv3hd.m3u8'#''rtmp://58.200.131.2:1935/livetv/hunantv'

# ---------------------------------------- lu bin -------------------------------------------------
phone_model = 'resources/cellphone_model/smokephone'
# phone_model = 'resources/cellphone_model/resnet-18-5hand'


phone_symbol_1 = \
    'resources/cellphone_model/ASSD_4scale_L16_RC_32_32_64_deploy.json'
phone_model_1 = \
    'resources/cellphone_model/ASSD_MD_32_32_64_People-detection_iter_850000_model.params'

phone_model_2 = \
    'resources/cellphone_model/head-4scaleV3_iter_1500000_model.params'

phone_symbol_2 = \
    'resources/cellphone_model/ASSD_4scale_16layers_V3_32_32_64_deploy.json'

# ------------------------------------------ w --------------------------------------------
oil_model = {
    'anni': 'resources/oil-model/resnet-18-Annihilator-86000/Annihilator',
    'pipe': 'resources/oil-model/resnet-18-Pipeline-86000/Pipeline',
    'el': 'resources/oil-model/resnet-18-E-line'}


config_path = 'config_scripts'
phone_save = True
oil_save = True

"***********************************Fire_config**********************************"

# fire_video = "/home/lijun/Videos/firemanyface_mv.avi"
# fire_video = "rtmp://192.168.10.120/live/livestreamsb"
# fire_video = "/media/tiger/MySSD/石油规划院/2019-05-17/11.0.18.15_01_20190517135153329.mp4"
fire_video = '/home/shy/linshi/NVR.mp4'#'rtmp://127.0.0.1:1935/live?vhost=stream15/livestream'#'rtsp://admin:q1w2e3r4@192.168.10.58:554'#'rtmp://58.200.131.2:1935/livetv/hunantv'
# fire_video = "rtmp://127.0.0.1:1935/live?vhost=stream15/livestream"
fire_model = "resnet-18-fire"
fire_model_num = 89
video_skip = 1
fire_runing = True


"******************************People_Detect_config*******************************"

# People_Detect_video = "rtsp://admin:q1w2e3r4@192.168.10.51:554"
# People_Detect_video = "/media/tiger/MySSD/石油规划院/2019-05-17/11.0.18.15_01_20190517135153329.mp4"
People_Detect_video = '/home/shy/linshi/NVR.mp4'#'rtmp://127.0.0.1:1935/live?vhost=stream16/livestream'#'rtsp://admin:q1w2e3r4@192.168.10.58:554'#'rtmp://58.200.131.2:1935/livetv/hunantv'#'/home/xw/young1080p-01.mp4'#'rtmp://202.69.69.180:443/webcast/bshdlive-pc'#'rtmp://58.200.131.2:1935/livetv/hunantv'
# People_Detect_video = "rtmp://192.168.10.120/live/livestreamsb"
# People_Detect_video = "rtmp://127.0.0.1:1935/live?vhost=stream16/livestream"
People_Classify_model = "resnet-18-Cloth"
People_Classify_num = 8
symbolFile = \
    "ASSD_4scale_L16_RC_32_32_64_deploy.json"
modelFile = \
    "ASSD_MD_32_32_64_People-detection_iter_850000_model.params"

ratio = 6.0  # 4.0
people_save = True


queue_lenth = 100

VEDIOSAVE = True


font_path = os.path.join(os.path.dirname(__file__), 'guanjiaKai.ttf')
# -------------------------------------------------------------------------------------------------------------------------------------------------------
selectedFont = ImageFont.truetype(font_path, 35)
#
