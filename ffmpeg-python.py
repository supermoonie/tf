import ffmpeg
import requests
import sys
import os
import threadpool
from requests import urllib3

urllib3.disable_warnings()


def concat(video_name='out.mp4'):
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
    current_dir = sys.path[0].replace('\\', '\\\\') + '\\\\'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/74.0.3729.131 Safari/537.36',
        'Connection': 'Keep-Alive'
    }
    video_res = requests.get(url=url, headers=headers, verify=False)
    with open(current_dir + str(index) + '.ts', 'wb') as ts:
        ts.write(video_res.content)


def generate_txt():
    current_path = sys.path[0].replace('\\', '\\\\') + '\\\\'
    files = os.listdir(current_path)
    files = list(sorted(filter(lambda x: x.endswith('.ts'), files)))
    with open(current_path + 'ts.txt', 'w') as txt:
        for file in files:
            txt.write('file \'' + sys.path[0].replace('\\', '/') + '/' + file + '\'\n')


def clear():
    current_path = sys.path[0].replace('\\', '\\\\') + '\\\\'
    files = os.listdir(current_path)
    for file in files:
        if file.endswith('.ts') or file.endswith('.txt'):
            os.remove(current_path + file)


if __name__ == '__main__':
    # download("https://media.wanmen.org/4f6a1de2-28a5-4f28-ab6d-f5b3d2fb7e0b_pc_high.m3u8")
    # generate_txt()
    # concat('物理预测的胜利与失效.mp4')
    # clear()
    url = "https://api.wanmen.org/4.0/content/courses/593e086f206e46163b6dd5c8"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/74.0.3729.131 Safari/537.36',
        'Connection': 'Keep-Alive'
    }
    lectures = requests.get(url=url, headers=headers, verify=False).json()['lectures']
    for lecture in lectures:
        print(lecture['name'])
        for course in lecture['children']:
            print('\t' + course['name'] + ' ' + course['hls']['pcHigh'])
