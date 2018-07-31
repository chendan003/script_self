# -*- coding: utf-8 -*-
"""
Created on Wed May 17 03:52:42 2017

@author: zhaoy
"""
import os
import sys
import os.path as osp
import time
import json
import math
import operator
from compare_feature import *
import requests
from ava_auth import AuthFactory
from numpy.linalg import norm
from matio import load_mat
import struct
import numpy as np


#default_config_file = 'facex_api_config.json'
default_config_file = 'facex_api_config_v3_ava.json'
output_base_dir = './facex_api_response_test'

_API_NAMES = ['det', 'age', 'gender', 'feature', 'cluster']


def print_help_info():
    print('''USAGE:
        python facex_dora_api_test api_names input_file [output_dir config_file]

        api_names:
            1) one of ['det', 'age', 'gender', 'feature'];
            2) a combination of them: 'det+age', 'age+gender';
        input_file:
            1) a .txt file, gives a list of image full urls, one url per line;
            2) a .json file, gives 'facex-det API'  response;
            3) a list of image names, will add 'facex_api_config.bucket_url' in each of the names;
        output_dir:
            path to save response json files;
            './facex_api_response_#yymmdd-HHMMSS, if default;
        config_file:
            config json file, which gives facex API urls;
            'facex_api_config.json', if default;
    '''
          )


def get_time_string():
#    _str = time.strftime('#%y%m%d-%H%M%S')
    _str = time.strftime('%y%m%d-%H%M%S')
    return _str


def load_config_file(config_json):
    fp = open(config_json, 'r')
    configs = json.load(fp)
#    print(configs)
    fp.close()
    return configs



def send_request_to_url(url, data, header=None, token=None):
    hdr = {'content-type': 'application/json'}
#    hdr={"Content-Type": "application/x-www-form-urlencoded"},
    if header:
        hdr.update(header)

    # print("Request header: " + str(hdr))
    # print("Request data: " + str(data))

    resp_content = None

    if isinstance(data, str) and data.endswith('.json'):
        fp = open(data)
        data = json.load(fp)
        fp.close()

    if isinstance(data, dict):
        resp = requests.post(url, None, data, headers=hdr, auth=token)
        print('--->resp.headers: ' + str(resp.headers))
        print('--->resp.content: ' + str(resp.content))

        resp_content = resp.content

    return resp_content


def convert_feature_data_2_npy(feat_data):
    if len(feat_data) != 2048:
        print('feature length is not 2048')
        return ""

    feat_len = len(feat_data) / 4
    fmt = '>%df' % feat_len
    print ('struct size: ', struct.calcsize(fmt))
    feat_t = struct.unpack(fmt, feat_data)
#    print feat_t

    feat = np.array(feat_t, np.float32)
    print ('feat.shape:', feat.shape)

    return feat


def request_facex_api(api_names, input_file, output_dir, config_file=None):
    #    print_help_info()

    if not config_file:
        config_file = default_config_file

    if not osp.exists(config_file):
        print("===> Error: Cannot find config_file: " + config_file)
        return

    print("===> Load configs from config_file: " + config_file)
    configs = load_config_file(config_file)
    print('===> configs: {}'.format(configs))
    if 'bucket_url' not in configs:
        print("===> Warning: 'bucket_url' not in config_file " + config_file)
        configs['bucket_url'] = ''
        return

    header = None

    token = None
    if 'ava_auth_conf' in configs:
        conf = configs['ava_auth_conf']
        factory = AuthFactory(conf["access_key"], conf["secret_key"])
        if conf["auth"] == "qiniu/mac":
            fauth = factory.get_qiniu_auth
        else:
            fauth = factory.get_qbox_auth

        token = fauth()
    elif 'Authorization' in configs:
        header = {"Authorization": configs['Authorization']}
    else:
        header = {"Authorization": "QiniuStub uid=1&ut=2"}

#    print 'token: ', token

    splits = api_names.split('+')
    api_names = []

    for it in splits:
        it = it.strip()
        if it == 'detection':
            it = 'det'
        if it == 'feat':
            it = 'feature'

        if it in _API_NAMES and it not in api_names:
            api_names.append(it)

    if len(api_names) < 0:
        print("===> Error: Wrong api_names: " + api_names)
        return

    if 'det' not in api_names:
        api_names.insert(0, 'det')

    for name in api_names:
        api_name = 'facex_%s_url' % name

        if api_name not in configs:
            print("===> Error: {} not in config_file: {}, continue to test next API".format(
                api_name, config_file))
            return

    if not osp.exists(input_file):
        print("===> Error: Cannot find input_file: " + input_file)
        return

    url_list = []
    fp = open(input_file)

    for line in fp:
        url = line.strip()
        if len(url) < 1 or url.startswith('#'):
            continue

        if 'bucket_url' in configs and not url.startswith('http'):
            url = configs['bucket_url'] + '/' + url
        url_list.append(url)

    fp.close()

    if len(url_list) < 1:
        print("===> Error: No valid image url")

#    if not output_dir:
#        time_str = get_time_string()
#        output_dir = output_base_dir + '_' + time_str
    #time_str = get_time_string()
    #output_dir = output_base_dir + '_' + time_str

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    req_fp_list = []

    det_req_file = osp.join(output_dir, 'facex_det_req_data_list.json')
    fp = open(det_req_file, 'w')
    fp.write('[\n')
    req_fp_list.append(fp)

    attr_req_file = osp.join(output_dir, 'facex_attr_req_data_list.json')
    fp = open(attr_req_file, 'w')
    fp.write('[\n')
    req_fp_list.append(fp)

    resq_save_fps = {}
    for api in api_names:
        resp_save_file = 'facex_' + api + '_resp_list.json'
        resp_save_file = osp.join(output_dir, resp_save_file)

        fp = open(resp_save_file, 'w')
        fp.write('[\n')
        resq_save_fps[api] = fp

    is_first = True

    for url_idx, url in enumerate(url_list):

        # facex-det request data
        data_dict = {'data': {'uri': url}}

        print("\n\n===>Process image: {}".format(url))

        api_url_name = 'facex_det_url'
        print("\n---> Test {}: {}".format(api_url_name, configs[api_url_name]))

        js_str = json.dumps(data_dict, indent=4)
        if not is_first:
            req_fp_list[0].write(',\n')

        req_fp_list[0].write(js_str)
        req_fp_list[0].flush()

        # call facex-det api
        tm1 = time.clock()
        resp = send_request_to_url(
            configs[api_url_name], data_dict, header, token)
        tm2 = time.clock()
        print("---> Request takes {} seconds".format(tm2 - tm1))

        # save facex-det response
        facex_det_resp = json.loads(resp)
        facex_det_resp['uri'] = url
#        print facex_det_resp
        js_str = json.dumps(facex_det_resp, indent=4)
        if not is_first:
            resq_save_fps['det'].write(',\n')

        resq_save_fps['det'].write(js_str)
        resq_save_fps['det'].flush()

        print 'facex_Det_Resp',facex_det_resp
	print 'facex_det_resp code', facex_det_resp['code']
        print 'length of detection', len(facex_det_resp['result']['detections'])	

        if (not facex_det_resp
           # or facex_det_resp['code']
            or 'result' not in facex_det_resp
            or 'detections' not in facex_det_resp['result']
            or len(facex_det_resp['result']['detections']) < 1
            ):
            print '--->facex-det failed or no face detected '
            continue
        
	# facex-det request data
        faces = facex_det_resp['result']['detections']
        print '\n--->%d faces detected' % len(faces)

        for face_idx, face in enumerate(faces):
            print '\n--->process face #%d' % face_idx
            # facex-[age, gender, feature] request data
            attr_data_dict = {
                'data': {
                    'uri': url,
                    'attribute': {'pts': face['pts']}
                }
            }

            js_str = json.dumps(attr_data_dict, indent=4)
            if not is_first:
                req_fp_list[1].write(',\n')

            req_fp_list[1].write(js_str)
            req_fp_list[1].flush()

            for api in api_names:
                if api == 'det':
                    continue

                api_url_name = 'facex_%s_url' % api

                print("\n---> Test {}: {}".format(api_url_name,
                                                  configs[api_url_name]))

                tm1 = time.clock()
                resp = send_request_to_url(
                    configs[api_url_name], attr_data_dict, header, token)
                tm2 = time.clock()
                print("---> Request takes {} seconds".format(tm2 - tm1))

                if api is 'feature':
		    print ('feature is in api')
                    facex_attr_resp = attr_data_dict
                    base_name = osp.basename(url)
                    base_name = osp.splitext(base_name)[0]
                    feat_fn_prefix = '%s_%d' % (base_name, face_idx)
                    fn_feat = feat_fn_prefix + '.dat'
		    print ('fn_feat%s' % fn_feat)
                    fn_feat = osp.join(osp.abspath(output_dir), fn_feat)
                    fp_feat = open(fn_feat, 'wb')
                    fp_feat.write(resp)

                    feat_np = convert_feature_data_2_npy(resp)
                    fn_npy = feat_fn_prefix + '.npy'
                    fn_npy = osp.join(osp.abspath(output_dir), fn_npy)
                    np.save(fn_npy, feat_np)

                    facex_attr_resp['feat_data'] = fn_feat
                    facex_attr_resp['feat_npy'] = fn_npy

                else:
                    facex_attr_resp = json.loads(resp)

                facex_attr_resp['uri'] = url
                facex_attr_resp['pts'] = face['pts']

                js_str = json.dumps(facex_attr_resp, indent=4)
                if not is_first:
                    resq_save_fps[api].write(',\n')

                resq_save_fps[api].write(js_str)
                resq_save_fps[api].flush()

        is_first = False

    for fp in req_fp_list:
        fp.write('\n]\n')
        fp.close()

    for (k, fp) in resq_save_fps.items():
        fp.write('\n]\n')
        fp.close()


#    if 'code' in facex_det_resp and facex_det_resp['code']!=0:
#        print('===> Error: code!=0 in facex-det response')
#        return
#
#    if 'facex_det' in facex_det_resp and len(facex_det_resp['facex_det'])<1:
#        print("===> Error: len(resp['facex_det'])<1")
#        return
#
#    data_dict = {
#            'image': data_dict['image'],
#            'facex_det': facex_det_resp['facex_det']
#            }
#
    return output_dir


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # api_names = 'age + det + gender + feat'
        # api_names = 'age + gender'
        api_names = 'det + feat'
        input_file = 'url_list.txt'
        time_str = get_time_string()
        output_dir = output_base_dir + '_' + time_str
        #output_dir = None 
        config_file = 'facex_api_config_v3_ava.json'

        request_facex_api(api_names, input_file, output_dir, config_file)

    elif len(sys.argv) < 3:
        print_help_info()
        exit
    else:
        api_names = sys.argv[1]
        input_file = sys.argv[2]

        output_dir = None if len(sys.argv) < 4 else sys.argv[3]
        config_file = False if len(sys.argv) < 5 else sys.argv[4]

        output_dir = request_facex_api(
            api_names, input_file, output_dir, config_file)

    src_img_dir = 'src_image'
    dst_cropimg_dir = 'cropped_image'
    save_similar_file = 'similarity_result.txt'
    print 'output_Dir is:', output_dir
    out_put_det_feat_result(output_dir, src_img_dir, input_file, dst_cropimg_dir, save_similar_file)

