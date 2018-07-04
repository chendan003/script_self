#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 04:17:24 2017

@author: zhaoy
"""
import os
import sys
import os.path as osp
import time
import json

import urllib2
import imghdr

import cv2
import numpy as np

#import skimage

#in_file_test = True  # set to false for command line calling
in_file_test = False  # set to false for command line calling

output_base_dir = 'facex_resp_vis_imgs_wan'


def print_help_info():
    print('''USAGE:
        python facex_response_visualize facex_response_json_file src_image_save_dir [output_image_save_dir show_image_flag]
        output_image_save_dir = ./facex_resp_vis_imgs_#yymmdd-HHMMSS/, if default;
        show_image_flag = Flase, in default;
    '''
          )


def get_time_string():
    _str = time.strftime('#%y%m%d-%H%M%S')
    return _str


def load_facex_response(file_name):
    rlt = None
    fp = open(file_name, 'r')
    rlt = json.load(fp)
    fp.close()

    return rlt


def find_dict_key_starts_with(dict_in, key_prefix):
    for key in dict_in.keys():
        if key.startswith(key_prefix):
            return key

    return None


def get_response_list(resp_dict, key_prefix='facex_'):
    key = find_dict_key_starts_with(resp_dict, key_prefix)
    if key:
        return resp_dict[key]
    else:
        return None


def get_url_image_name(url, content=None, try_download=False):
    base_name = osp.basename(url)
    img_extensions = ['.jpg', '.bmp', '.png', '.jpeg', '.gif', '.tiff']

    for it in img_extensions:
        if base_name.endswith(it):
            return base_name

    if not content and try_download:
        content = urllib2.urlopen(url).read()

    if content:
        imgtype = imghdr.what('', h=content)

        if not imgtype:
            imgtype = 'txt'

        return base_name + '.' + imgtype
    else:
        return base_name


def download_image(url, save_dir='./'):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    content = urllib2.urlopen(url).read()
    img_name = get_url_image_name(url, content)
    save_name = osp.join(save_dir, img_name)
    #print('===>Test: Save {} into {}\n'.format(url, save_name))

    with open(save_name, 'wb') as f:
        f.write(content)

    return save_name


def cv2_imread_unicode(img_path):
    if isinstance(img_path, unicode):
        stream = open(img_path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        return bgrImage
    else:
        return cv2.imread(img_path)


def cv2_put_text_to_image(img, text, x, y, font_pix_h, color=(255, 0, 0)):
    if font_pix_h < 10:
        font_pix_h = 10

    # print img.shape

    h = img.shape[0]

    if x < 0:
        x = 0

    if y > h - 1:
        y = h - font_pix_h

    if y < 0:
        y = font_pix_h

    font_size = font_pix_h / 30.0
    # print font_size
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_size, color, 1)


def resize_image_roi(roi, roi_scale=1.0, output_square=0):
    cx = int(roi[0] + roi[2] / 2.0)
    cy = int(roi[1] + roi[3] / 2.0)
    wd = int(roi[2] * roi_scale)
    ht = int(roi[3] * roi_scale)

    if output_square:
        wd = max(wd, ht)
        ht = wd

    roi_new = [cx - wd / 2, cy - ht / 2, wd, ht]
    return roi_new


def resize_image_roi_4pts(pts, roi_scale=1.0, output_square=0):

    cx = int((pts[0][0] + pts[1][0]) / 2.0)
    cy = int((pts[0][1] + pts[3][1]) / 2.0)
    wd = int((pts[1][0] - pts[0][0]) * roi_scale)
    ht = int((pts[3][1] - pts[0][1]) * roi_scale)

    if output_square:
        wd = max(wd, ht)
        ht = wd

    pts_new = [[cx - wd / 2, cy - ht / 2],
               [cx + wd / 2, cy - ht / 2],
               [cx + wd / 2, cy + ht / 2],
               [cx - wd / 2, cy + ht / 2],
               ]
    return pts_new


def visualize_response(input_response_file, src_save_dir, output_save_dir=None, show_image_flag=False):
    if not output_save_dir:
        time_str = get_time_string()
        output_save_dir = output_base_dir + '_' + time_str
        #output_save_dir = osp.join(osp.dirname(src_save_dir), output_save_dir)

    if not osp.exists(output_save_dir):
        os.makedirs(output_save_dir)

    facex_list = load_facex_response(input_response_file)

    fp_succeed_list = open(
        osp.join(output_save_dir, 'facex_vis_succeeded_url_list.txt'), 'w')
    fp_failed_list = open(
        osp.join(output_save_dir, 'facex_vis_failed_url_list.txt'), 'w')

    #item = facex_list[1]
    for item in facex_list:
        print item

        img_url = item['uri']
        img_name = get_url_image_name(img_url)
        local_save_name = osp.join(src_save_dir, img_name)

        print("\n===>Process image URL: " + img_url)

        if osp.exists(local_save_name):
            print('--->Found a local image for URL: ' + img_url)
            print('    Local image is: ' + local_save_name)
            print('    Will not download this image URL.')

        else:
            local_save_name = download_image(img_url, src_save_dir)
            print('--->Download image URL: ' + img_url)
            print('    into local path: ' + local_save_name)

        if ('code' not in item or
                'result' not in item or
               (item['code'] != 0 and item['code'] != 200)
        ):
            print("--->Error in response for: " + img_url)
            fp_failed_list.write(img_url + '\n')
            continue

        item_result = item['result']

        faces_kw = None

        if 'detections' in item_result:  # for response of facex_det API
            faces_kw = 'detections'
        elif 'faces' in item_result:  # for response of [facex_age, facex_gender] APIs
            faces_kw = 'faces'
        else:
            print("--->No faces found in: " + img_url)
            fp_failed_list.write(img_url + '\n')
            continue

        if faces_kw:
            # if item['code']==0 and 'detections' in item:
            # print(type(local_save_name))
            #print('local_save_name: ' + local_save_name)

            print('--->Try to load saved local image: ' + local_save_name)

            if isinstance(local_save_name, unicode):
                #                utf_name = local_save_name.encode('utf-8')
                #                print('type(utf_name): ' + str(type(utf_name)))
                #                print('utf_name: ' + utf_name)
                #                img = cv2.imread(utf_name)
                img = cv2_imread_unicode(local_save_name)
            else:
                img = cv2.imread(local_save_name)

            if img is None:
                print('--->Error: failed to open image: \n' + local_save_name)
                fp_failed_list.write(img_url + '\n')
                continue

            for face in item_result[faces_kw]:
                pts = face['pts']
                cv2.rectangle(img, tuple(pts[0]),
                              tuple(pts[2]), (0, 255, 0), 3)

                new_pts = resize_image_roi_4pts(pts, 1.5, 1)
                cv2.rectangle(img, tuple(new_pts[0]), tuple(
                    new_pts[2]), (0, 255, 255), 3)

                if 'facial_5pts' in face:
                    for f_pt in face['facial_5pts']:
                        cv2.circle(img, tuple(f_pt), 1, (255, 0, 0), 2)

                if 'landmarks' in face:
                    for lm_pt in face['landmarks']:
                        cv2.circle(img, tuple(lm_pt), 1, (0, 0, 255), 2)

                font_pix_h = max(pts[2][0] - pts[0][0],
                                 pts[0][1] - pts[0][1]) / 5
                if font_pix_h < 10:
                    font_pix_h = 10

                txt_x = pts[0][0]
                txt_y = pts[3][1] + font_pix_h

                # put detection "score" into output image, if have this
                # attribute
                if 'score' in face:
                    print('--->put detection "score" into output image')
                    score = float(face['score'])
                    #print('score = {}, type(score)={}'.format(score, type(score)))
                    text = 'score=' + str(round(score, 3))

                    # print font_pix_h
                    cv2_put_text_to_image(
                        img, text, txt_x, txt_y, font_pix_h, (255, 0, 255))

                    txt_y -= font_pix_h

                # put 'age' into output image, if have this attribute
                if 'age' in face:
                    print('--->put "age" into output image')
                    age = float(face['age'])
                    text = 'age=' + str(round(age, 1))

                    # print font_pix_h
                    cv2_put_text_to_image(
                        img, text, txt_x, txt_y, font_pix_h, (255, 255, 255))

                    txt_y -= font_pix_h

                # put 'gender' into output image, if have this attribute
                if 'gender' in face:
                    print('--->put "gender" into output image')
                    text = face['gender']

                    # print font_pix_h
                    cv2_put_text_to_image(
                        img, text, txt_x, txt_y, font_pix_h, (255, 255, 255))

                    txt_y -= font_pix_h

            base_name = osp.basename(local_save_name)
            #print('base_name: ', base_name)

            output_save_name = osp.join(output_save_dir, base_name)
            cv2.imwrite(output_save_name, img)
            print("--->Save output image into file: " + output_save_name)

            fp_succeed_list.write(img_url + '\n')

            if show_image_flag:
                cv2.imshow('img', img)
                cv2.waitKey(0)

        if show_image_flag:
            cv2.destroyAllWindows()

    fp_failed_list.close()
    fp_succeed_list.close()

#print('__name__=' + __name__)


if __name__ == '__main__':
    if len(sys.argv)==1:
        in_file_test = True

    if not in_file_test:  # test by command line
        if len(sys.argv) < 3:
            print_help_info()
            exit
#  command line's first parameter is facex_det dir 
#  second para is original image dir
        input_response_file = osp.join(sys.argv[1], 'facex_det_resp_list.json')
        local_src_img_dir = sys.argv[2] 

        output_save_dir = None if len(sys.argv) < 4 else sys.argv[3]
        show_image_flag = False if len(sys.argv) < 5 else sys.argv[4]

        visualize_response(input_response_file, local_src_img_dir,
                           output_save_dir, show_image_flag)
