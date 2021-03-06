import ffmpeg
import requests
import sys
import os
import threadpool
import platform
from requests import urllib3

urllib3.disable_warnings()

from datetime import datetime


def concat(video_name='out.mp4'):
    current_path = sys.path[0] + '/'
    if platform.system() == 'Windows':
        current_path = sys.path[0].replace('\\', '\\\\') + '\\\\'
    input_file = current_path + 'ts.txt'
    output_file = current_path + video_name
    try:
        out, err = (
            ffmpeg.input(input_file, f='concat', safe='0').output(output_file, c='copy').run(quiet=False,
                                                                                             overwrite_output=True)
        )
    except Exception as e:
        print(e)


def download(url, thread_num=5):
    prefix = url[:url.rfind('/') + 1]
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/74.0.3729.131 Safari/537.36',
        'Connection': 'Keep-Alive'
    }
    m3u8_res = requests.get(url=url, headers=headers, verify=False)
    m3u8_txt = m3u8_res.text
    lines = m3u8_txt.split('\n')
    lines = list(filter(lambda x: x.endswith('.ts'), lines))
    index = 100000000
    args = []
    for line in lines:
        args.append((None, {'url': prefix + line, 'index': index}))
        index = index + 1
    pool = threadpool.ThreadPool(thread_num)
    reqs = threadpool.makeRequests(download_worker, args)
    [pool.putRequest(req) for req in reqs]
    pool.wait()
    generate_txt()


def download_worker(url, index):
    current_dir = sys.path[0].replace('\\', '/') + '/'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/74.0.3729.131 Safari/537.36',
        'Connection': 'Keep-Alive'
    }
    video_res = requests.get(url=url, headers=headers, verify=False)
    with open(current_dir + str(index) + '.ts', 'wb') as ts:
        ts.write(video_res.content)


def generate_txt():
    current_path = sys.path[0].replace('\\', '/') + '/'
    files = os.listdir(current_path)
    files = list(sorted(filter(lambda x: x.endswith('.ts'), files)))
    with open(current_path + 'ts.txt', 'w') as txt:
        for file in files:
            txt.write('file \'' + sys.path[0].replace('\\', '/') + '/' + file + '\'\n')


def clear():
    current_path = sys.path[0].replace('\\', '/') + '/'
    files = os.listdir(current_path)
    for file in files:
        # if file.endswith('.ts') or file.endswith('.txt'):
        if file.endswith('.ts'):
            os.remove(current_path + file)


if __name__ == '__main__':
    download("https://media.wanmen.org/a81d11d0-f1bb-4469-b630-f1dd674081bb_pc_high.m3u8")
    generate_txt()
    concat('1.4 生活实例与本章答疑.mp4')
    clear()
