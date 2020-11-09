# coding: utf-8

import cv2
import config as cfg

# import ..config as cfg

drawing = 0  # 鼠标按下为真
ix, iy = -1, -1
px, py = -1, -1
coordinate = [-1, -1, -1, -1]
area = []


def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, px, py
    global mouse_flag
    coordinate = [-1] * 4
    if event == cv2.EVENT_MOUSEMOVE:
        return
    elif event == cv2.EVENT_LBUTTONDOWN:
        drawing += 1
        px, py = ix, iy
        # ix, iy = 2 * x, 2 * y
        ix, iy = x, y
        if drawing == 3:
            drawing = 1
        if drawing == 2:
            coordinate = px, py, ix, iy
            area.append(coordinate)

    elif event == cv2.EVENT_RBUTTONDOWN:
        area.pop()
    # print("*****************coordinate: ", coordinate)
    # print("*****************area: ", area)


def check_draw_circle(video):
    count = 0
    output_str = ''
    # camera = cv2.VideoCapture("rtsp://admin:q1w2e3r4@192.168.10.51:554")  # 参数0表示第一个摄像头
    camera = cv2.VideoCapture(video)

    if (camera.isOpened()):  # 判断视频是否打开
        print('Open')
    else:
        print('摄像头未打开')

    while True:
        # 读取视频流
        ok, frame = camera.read()
        if not ok:
            camera.release()
            break
        count += 1
        if count % 1 != 0:
            continue

        if drawing == 2:
            for idx, output in enumerate(area):
                cv2.rectangle(frame, (area[idx][0], area[idx][1]), (area[idx][2], area[idx][3]), (0, 255, 0), 2)

        cv2.imshow("camera", frame.copy())
        cv2.setMouseCallback('camera', draw_circle)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            # break
            # print(area)
            for idx, output in enumerate(area):
                output_str += str(area[idx][0]) + ' ' + str(area[idx][1]) + ' ' + str(area[idx][2]) + ' ' + str(area[idx][3]) + ' '
            with open('../config_scripts/cloth_config.txt', 'w') as fout:
                fout.write(output_str)
            break
        elif k == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video = cfg.People_Detect_video
    # video = 'rtmp://127.0.0.1:1935/live?vhost=stream16/livestream'
    check_draw_circle(video)
