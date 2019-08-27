import os
import re
import cv2
import shutil
import time
import random

# capture_path = 'D:/data/20181216/correct/'
# support_capture_name_list = [f for f in os.listdir(capture_path) if re.match(r'^[0-9a-zA-Z]{4}$', f.split('_')[0])]
# support_capture_path_list = [os.path.join(capture_path, f) for f in support_capture_name_list]
# for f in support_capture_path_list:
#     print(f)
#     shutil.copy(f, 'D:/data/20181216/correct_11/')

# capture_path = 'D:/data/20181216/correct_11/'
# to_dir = 'D:/data/20181216/correct_12/'
# capture_name_list = [f for f in os.listdir(capture_path)]
# index = 1
# for f in capture_name_list:
#     path = os.path.join(capture_path, f)
#     to_path = os.path.join(to_dir, f)
#     print('{}, {} ----> {}'.format(index, path, to_path))
#     img = cv2.imread(path)
#     img_new = cv2.resize(img, (100, 60))
#     cv2.imwrite(to_path, img_new)
#     index += 1

capture_path = 'D:/data/20181216/100x60/'
image_name_list = os.listdir(capture_path)
random.seed(time.time())
random.shuffle(image_name_list)
train_dir = 'D:/data/20181216/train/'
test_dir = 'D:/data/20181216/test/'
for i in range(100):
    print('{} ----> {}'.format(os.path.join(capture_path, image_name_list[i]),
                               os.path.join(test_dir, image_name_list[i])))
    shutil.move(os.path.join(capture_path, image_name_list[i]), os.path.join(test_dir, image_name_list[i]))
