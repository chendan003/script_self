# Preprocessing: 
图片上传到bucket，底库图片用"idcard" 作前缀，需要对比的图片（例如视频中截出的人等）
用“scene” 作前缀. 然后将图片对应的url写入 url_list.txt（每行对应一个url）
# exec script：
python facex_api_v3_request.py 

