# -*- coding:utf-8 -*-

import cv2

img = cv2.imread('image_path')
cv2.rectangle(org_img0,  (x1 , y1),(x2,y2),(0,255,0), 1)
cv2.imwrite('save_image_name', img)
