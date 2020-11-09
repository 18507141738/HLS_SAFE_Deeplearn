# coding: utf-8

'''
function:
    1. 鼠标事件

作用：
    1. 为视频选定区域


'''

import cv2
import os
import config as cfg
import numpy as np


def get_rect(image, title='get_rect'):
    '''
    获取两次鼠标点击事件得到的矩形框
    :param image:
    :param title:
    :return: (x1, y1), (x2, y2)
    '''
    mouse_params = {'tl': None, 'br': None, 'current_pos': None,
                    'released_once': False}

    cv2.namedWindow(title)
    cv2.moveWindow(title, 100, 100)

    def on_mouse(event, x, y, flags, param):

        param['current_pos'] = (x, y)

        if param['tl'] is not None and not (flags & cv2.EVENT_FLAG_LBUTTON):
            param['released_once'] = True

        if flags & cv2.EVENT_FLAG_LBUTTON:
            if param['tl'] is None:
                param['tl'] = param['current_pos']
            elif param['released_once']:
                param['br'] = param['current_pos']

    cv2.setMouseCallback(title, on_mouse, mouse_params)
    cv2.imshow(title, image)

    while mouse_params['br'] is None:
        im_draw = np.copy(image)

        if mouse_params['tl'] is not None:
            cv2.rectangle(im_draw, mouse_params['tl'],
                          mouse_params['current_pos'], (255, 0, 0))

        cv2.imshow(title, im_draw)
        cv2.waitKey(10)

    cv2.destroyWindow(title)

    start = (min(mouse_params['tl'][1], mouse_params['br'][1]),
             min(mouse_params['tl'][0], mouse_params['br'][0]))

    end = (max(mouse_params['tl'][1], mouse_params['br'][1]),
           max(mouse_params['tl'][0], mouse_params['br'][0]))

    return start, end


def get_unload_oil():
    '''
    将选定的区域保存到 config_scripts 目录下
    :return: None
    '''
    cap = cv2.VideoCapture(cfg.oil_video)
    while cap.isOpened():
        first_success, first_frame = cap.read()
        if first_success:
            anni_a, anni_b = get_rect(first_frame, title='get_rect')
            pipe_a, pipe_b = get_rect(first_frame, title='get_rect')
            el_a, el_b = get_rect(first_frame, title='get_rect')

            anni_area = anni_a + anni_b
            pipe_area = pipe_a + pipe_b
            el_area = el_a + el_b

            cv2.destroyAllWindows()
            cap.release()
            break
        else:
            continue
    print([anni_area, pipe_area, el_area])
    # 获取图像区域后将其保存
    with open('../config_scripts/oil_config.txt', mode='w') as f:
        for area in [anni_area, pipe_area, el_area]:
            # area --> y1 x1 y2 x2
            f.write('%d %d %d %d\n' % (area[0], area[1], area[2], area[3]))


if __name__ == '__main__':
    get_unload_oil()
