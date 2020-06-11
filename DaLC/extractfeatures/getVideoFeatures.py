import numpy as np
import cv2 as cv
import sys
import video
import getFrameFeatures

# "set PYTHONPATH=C:/Users/young/Desktop/test/opencv/samples/python" for "import video"
# optical flow(motion vector): https://docs.opencv.org/master/d7/d8b/tutorial_py_lucas_kanade.html


def getVideoFeatures(video_path):
    '''
        return 光流均值，光流标准差，颜色冷暖，颜色轻重，颜色活跃度，颜色柔和度，暗色比例，亮色比例，饱和度，颜色能量，颜色方差
    '''
    optical_flow_list = []

    cam = video.create_capture(video_path)
    ret, prev = cam.read()
    # 2次图形下采样
    prev = cv.pyrDown(prev)
    prev = cv.pyrDown(prev)
    prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)

    video_warmc_total = 0
    video_heavyl_total = 0
    video_activep_total = 0
    video_hards_total = 0
    video_darkProportion_total = 0
    video_lightPropertion_total = 0
    video_saturation_total = 0
    video_color_energy_total = 0
    video_color_std_total = 0

    count = 0
    n = 0

    SAMPLE_BAND = 10

    while True:
        ret, img = cam.read()
        if not ret:
            break

        count += 1
        # 每x帧采一次样
        print(count)
        if count % SAMPLE_BAND != 0 and count % SAMPLE_BAND != 1:
            continue

        # 图像下采样
        img = cv.pyrDown(img)
        img = cv.pyrDown(img)

        # 获得该帧的运动矢量
        #   https://docs.opencv.org/master/d7/d8b/tutorial_py_lucas_kanade.html
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if count % SAMPLE_BAND == 1:
            flow = cv.calcOpticalFlowFarneback(
                prevgray, gray, None, 0.5, 3, 150, 3, 5, 1.2, 0)
            optical_flow_list.append(flow)

            # 获得该帧的颜色特征
            video_warmc, video_heavyl, video_activep, video_hards, video_darkProportion, video_lightPropertion, video_saturation, video_color_energy, video_color_std = getFrameFeatures.getColorFeatures(
                img)

            video_warmc_total += video_warmc
            video_heavyl_total += video_heavyl
            video_activep_total += video_activep
            video_hards_total += video_hards
            video_darkProportion_total += video_darkProportion
            video_lightPropertion_total += video_lightPropertion
            video_saturation_total += video_saturation
            video_color_energy_total += video_color_energy
            video_color_std_total += video_color_std

            n += 1
        prevgray = gray

    cam.release()

    optical_flow_array = np.array(optical_flow_list)
    print("optical_flow_array.shape: ", optical_flow_array.shape)
    optical_flow_mean = np.mean(optical_flow_array, axis=0)
    print("optical_flow_mean.shape: ", optical_flow_mean.shape)
    optical_flow_std = np.std(optical_flow_array, axis=0)
    print("optical_flow_std.shape: ", optical_flow_std.shape)

    video_optical_flow_mean = optical_flow_mean.mean(axis=0).mean(axis=0)
    video_optical_flow_std = optical_flow_std.mean(axis=0).mean(axis=0)

    video_warmc_total /= n
    video_heavyl_total /= n
    video_activep_total /= n
    video_hards_total /= n
    video_darkProportion_total /= n
    video_lightPropertion_total /= n
    video_saturation_total /= n
    video_color_energy_total /= n
    video_color_std_total /= n

    x, y = video_optical_flow_mean
    a, b = video_optical_flow_std

    return x * x + y * y, (abs(a) + abs(b)) / 2, video_warmc_total, video_heavyl_total, video_activep_total, video_hards_total, video_darkProportion_total, video_lightPropertion_total, video_saturation_total, video_color_energy_total, video_color_std_total


if __name__ == '__main__':
    video_path = sys.argv[1]
    # audio_path = 'noiseShort.wav'
    print('Video feauture of %s: %s' %
          (video_path, getVideoFeatures(video_path)))
