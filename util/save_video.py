# coding: utf-8

import cv2
import threading
import numpy as np
import os


def save_stream15(ip_path, output_file):
    cap = cv2.VideoCapture(ip_path)
    size = (int(cap.get(3)), int(cap.get(4)))
    fourcc = cv2.VideoWriter_fourcc(*'MP42')  # 最小压缩格式
    fps = cap.get(5)
    output_movie = cv2.VideoWriter(output_file, fourcc, fps, size)
    while True:
        success, frame = cap.read()
        if not success:
            cap.release()
            print('over')
            exit(1)

        output_movie.write(frame)


def save_stream16(ip_path, output_file):
    cap = cv2.VideoCapture(ip_path)
    size = (int(cap.get(3)), int(cap.get(4)))
    fourcc = cv2.VideoWriter_fourcc(*'MP42')  # 最小压缩格式
    fps = cap.get(5)
    output_movie = cv2.VideoWriter(output_file, fourcc, fps, size)
    while True:
        success, frame = cap.read()
        if not success:
            cap.release()
            print('over')
            exit(1)

        output_movie.write(frame)


if __name__ == '__main__':
    stream_15 = 'rtmp://127.0.0.1:1935/live?vhost=stream15/livestream'
    stream_16 = 'rtmp://127.0.0.1:1935/live?vhost=stream16/livestream'

    stream15_out = '/media/tiger/Elements/Sinter-5-30/stream15.avi'
    stream16_out = '/media/tiger/Elements/Sinter-5-30/stream16.avi'
    t1 = threading.Thread(target=save_stream15, args=[stream_15, stream15_out])
    t2 = threading.Thread(target=save_stream16, args=[stream_16, stream16_out])

    t1.start()
    t2.start()

    t1.join()
    t2.join()
