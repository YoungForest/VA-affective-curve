import getVideoFeatures
import getAudioFeatures
import moviepy.editor as mvp
import subprocess
import argparse
import sys
import glob
from pathlib import Path
import numpy as np
import cv2 as cv
import pandas as pd
import os
import time


def extractFeature(video_path):
    # 获得该视频的视觉特征
    optical_flow_mean, optical_flow_std, video_warmc_total, video_heavyl_total, video_activep_total, video_hards_total, video_darkProportion_total, video_lightPropertion_total, video_saturation_total, video_color_energy_total, video_color_std_total = getVideoFeatures.getVideoFeatures(video_path)

    # 获得该视频的音频特征
    audio_path = convert_video2audio(video_path)
    audio_harmonic, audio_engergy, audio_centroid, audio_contrast, audio_zero_crossing_rate, audio_slience_rate = getAudioFeatures.getAudioFeatures(
        audio_path)

    return optical_flow_mean, optical_flow_std, video_warmc_total, video_heavyl_total, video_activep_total, video_hards_total, video_darkProportion_total, video_lightPropertion_total, video_saturation_total, video_color_energy_total, video_color_std_total, audio_harmonic, audio_engergy, audio_centroid, audio_contrast, audio_zero_crossing_rate, audio_slience_rate


def convert_video2audio(video_path):
    filename, file_extension = os.path.splitext(video_path)
    audio_path = filename + ".wav"

    video_file = Path(video_path)
    if video_file.is_file():
        audio_file = Path(audio_path)
        if not audio_file.is_file():
            # prequire: command "ffmpeg" is installed and added to execute path
            command = "ffmpeg -i %s %s" % (video_path, audio_path)
            subprocess.call(command, shell=True)
        else:
            print("audio already exists: %s; convert pass" %(audio_path))
    else:
        print("video not exists: %s" %(video_path))
    return audio_path


FLAGS = None

if __name__ == '__main__':
    # options
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dir', type=str, default='.',
    #                     help='the directory including videos being processed')
    # parser.add_argument('--expansion', type=str, default='flv',
    #                     help="the extension name of video, e.g. flv, mp4")
    # FLAGS, unparsed = parser.parse_known_args()

    #dir_path = FLAGS.dir
    #expansion = FLAGS.expansion

    # 当视频数比较多时，使用多进程并行提取特征。1225个需要大概15h.
    process_number = 32
    dir_path = '/root/yangsen-data/LIRIS-ACCEDE-data/data'
    expansion = 'mp4'
    i = int(sys.argv[1])
    assert(i >= 0 and i < process_number)
    #print("unparse options: ", unparsed)

    # timer
    start = time.time()

    videos = glob.glob(dir_path + "/*." + expansion)
    videos_length = len(videos)
    part_length = len(videos) // process_number
    videos = videos[i * part_length:min((i+1)*part_length, videos_length)]
    print(len(videos))

    # pandas DataFrame
    video_features = pd.DataFrame(columns=['id', 'optical_flow_mean', 'optical_flow_std', 'video_warmc', 'video_heavyl', 'video_activep', 'video_hards', 'video_darkProportion', 'video_lightPropertion', 'video_saturation', 'video_color_energy', 'video_color_std'])
    audio_features = pd.DataFrame(columns=['id', 'audio_harmonic', 'audio_engergy', 'audio_centroid', 'audio_contrast_low', 'audio_contrast_middle', 'audio_contrast_high', 'audio_zero_crossing_rate', 'audio_slience_rate'])

    for v in videos:
        name = os.path.basename(v)
        filename, file_extension = os.path.splitext(name)
        optical_flow_mean, optical_flow_std, video_warmc_total, video_heavyl_total, video_activep_total, video_hards_total, video_darkProportion_total, video_lightPropertion_total, video_saturation_total, video_color_energy_total, video_color_std_total, audio_harmonic, audio_engergy, audio_centroid, audio_contrast, audio_zero_crossing_rate, audio_slience_rate = extractFeature(v)
        contrast_low, contrast_middle, contrast_high = audio_contrast
        video_features = video_features.append({'id': filename, 'optical_flow_mean': optical_flow_mean, 'optical_flow_std': optical_flow_std, 'video_warmc': video_warmc_total, 'video_heavyl': video_heavyl_total, 'video_activep': video_activep_total, 'video_hards': video_hards_total, 'video_darkProportion': video_darkProportion_total, 'video_lightPropertion': video_lightPropertion_total, 'video_saturation': video_saturation_total, 'video_color_energy': video_color_energy_total, 'video_color_std': video_color_std_total}, ignore_index=True)
        audio_features = audio_features.append({'id': filename, 'audio_harmonic': audio_harmonic, 'audio_engergy': audio_engergy, 'audio_centroid': audio_centroid, 'audio_contrast_low': contrast_low, 'audio_contrast_middle': contrast_middle, 'audio_contrast_high': contrast_high, 'audio_zero_crossing_rate': audio_zero_crossing_rate, 'audio_slience_rate': audio_slience_rate}, ignore_index=True)
        
    video_features = video_features.set_index('id')
    audio_features = audio_features.set_index('id')

    video_features.to_csv(path_or_buf=f'video_features-part{i}.csv')
    audio_features.to_csv(path_or_buf=f'audio_features-part{i}.csv')

    print("features extracted over")
    end = time.time()
    print('Task runs %0.2f seconds.' % (end-start))
    print(f'task length: {len(videos)}')
    print(f'cost per clip: {(end - start) / len(videos)}')
