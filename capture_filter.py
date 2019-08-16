import os
import re
import cv2


capture_path = 'D:/BaiduNetdiskDownload/20181211/data/export/crawler_captcha/correct/'
un_support_capture_name_list = [f for f in os.listdir(capture_path) if not re.match(r'^[0-9a-zA-Z]{4}$', f.split('-')[0])]
un_support_capture_path_list = [capture_path + f for f in un_support_capture_name_list]
for f in un_support_capture_path_list:
    os.remove(f)
capture_name_list = [f for f in os.listdir(capture_path) if re.match(r'^[0-9a-zA-Z]{4}$', f.split('-')[0])]
capture_path_list = [capture_path + f for f in capture_name_list]
for f in capture_path_list:
    img = cv2.imread(f)
    img_new = cv2.resize(img, (100, 60))
    cv2.imwrite(f, img_new)