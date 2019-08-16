import os
import numpy as np
from PIL import Image


def load(path=r'D:/mac_temp/captcha/'):
    captcha_names, file_paths = [], []
    index = 0
    for captcha_name in sorted(os.listdir(path)):
        captcha_dir = path + captcha_name
        if not os.path.isdir(captcha_dir):
            continue
        paths = [captcha_dir + '/' + f for f in sorted(os.listdir(captcha_dir))]
        n_pictures = len(paths)
        captcha_names.extend([captcha_name] * n_pictures)
        file_paths.extend(paths)
        print(captcha_dir + ' : ' + str(n_pictures))
        index = index + 1
        if index >= 10:
            break

    target_names = np.unique(captcha_names)
    target = np.searchsorted(target_names, captcha_names)

    captchas = []
    for file_path in file_paths:
        im = np.array(Image.open(file_path).convert('L'), 'f')
        im_list = []
        for row in im:
            for col in row:
                im_list.append(col)
        captchas.append(im_list)

    return np.array(captchas), target, target_names


if __name__ == '__main__':
    captchas, target, target_names = load()
