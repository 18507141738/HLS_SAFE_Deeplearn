import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import config as cfg

def addchinese222(image, strs, locals, sizes, colour=(0, 0, 255)):
    cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    print(11111111111111111111111)
    font = ImageFont.truetype(cfg.font_path, sizes, encoding="utf-8")
    print(222222222222222222222)
    draw.text(locals, strs, colour, font=font)
    print(3333333333333333333)
    image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    print(44444444444)
    cv2.imshow("isdasad",cv2.resize(image,(1920,1080)))
    cv2.waitKey()
    return image

if __name__ == "__main__":
    cap = cv2.imread("/home/shy/dev/ICPMS_HSL/resources/1114fd3e-a639-11ea-974e-0f916bf54c0a.jpg")
    print(cap)
    s = addchinese222(cap,"大撒旦撒多撒撒旦法撒旦",(100,140),18)
    # cv2.imshow("的撒多撒的撒多",s)
    cv2.imshow("isdasad", cv2.resize(s, (1920, 1080)))
    cv2.waitKey("1")