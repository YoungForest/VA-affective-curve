import cv2
import numpy as np
import sys


def getColorFeatures(frame):
    '''
        return warmc, heavyl, activep, hards, darkProportion, lightProportion, saturation, color_energy, color_std
# 颜色冷暖, 颜色轻重, 颜色活跃度, 颜色柔和度, 暗色比例, 亮色比例, 颜色饱和度, 颜色能量, 颜色方差
    '''
    # covert color space from BGR to LAB: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
    colorLAB = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # convert color space from LAB to Lch: https://en.wikipedia.org/wiki/Lab_color_space#Cylindrical_representation:_CIELCh_or_CIEHLC
    L = colorLAB[:, :, 0]
    a = colorLAB[:, :, 1]
    b = colorLAB[:, :, 2]

    C = np.sqrt(np.square(a) + np.square(b))
    h = np.arctan(b / a)

    # 颜色柔和度
    HS = 11.1 - 0.03 * (100 - L) - 11.4 * np.power(C, 0.02)
    hards = np.mean(HS)

    # 颜色冷暖
    WC = -0.5 + 0.02 * np.power(C, 1.07) * np.cos(h - (50 / 180) * np.pi)
    warmc = np.mean(WC)

    # 颜色活跃度
    grayColor = np.zeros((1, 1, 3), dtype=np.uint8)
    grayColor[0, 0, 0] = 128
    grayColor[0, 0, 1] = 128
    grayColor[0, 0, 2] = 128
    grayLAB = cv2.cvtColor(grayColor, cv2.COLOR_BGR2LAB)
    grayL = grayLAB[:, :, 0]
    grayA = grayLAB[:, :, 1]
    grayB = grayLAB[:, :, 2]
    grayC = np.sqrt(np.square(grayA) + np.square(grayB))
    grayH = np.arctan(grayB / grayA)
    AP = -1.1 + 0.03 * \
        np.power(np.power((C - grayC - 0.0107), 2) +
                 np.power((L - grayL) / 1.5, 2), 1 / 2)
    activep = np.mean(AP)

    # 颜色轻重
    HL = -2.1 + 0.05 * (100 - L)
    heavyl = np.mean(HL)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    vmax = np.max(v)
    vmin = np.min(v)
    # normalized
    v_normailized = v / (vmax - vmin + 1e-8)

    darkCount = 0
    lightCount = 0
    for element in v_normailized.flat:
        if element < 0.3:
            darkCount += 1
        elif element > 0.7:
            lightCount += 1
        else:
            pass

    # 暗色比例
    darkProportion = darkCount / v.size
    # 亮色比例
    lightProportion = lightCount / v.size

    s = hsv[:, :, 1]
    saturation = np.mean(s)

    # 颜色能量
    [height, width] = h.shape
    h = hsv[:, :, 0]
    color_energy = np.sum(s * v) / (np.std(h) * height * width + 1e-8)

    # 颜色方差
    l_mean = np.mean(L)
    c_mean = np.mean(C)
    h_mean = np.mean(h)

    [row, cols] = L.shape
    tmp_ll = 0
    tmp_cc = 0
    tmp_hh = 0
    tmp_lc = 0
    tmp_lh = 0
    tmp_ch = 0
    for i in range(row):
        for j in range(cols):
            tmp_ll += (L[i, j] - l_mean) * (L[i, j] - l_mean)
            tmp_cc += (C[i, j] - c_mean) * (C[i, j] - c_mean)
            tmp_hh += (h[i, j] - h_mean) * (h[i, j] - h_mean)
            tmp_lc += (L[i, j] - l_mean) * (C[i, j] - c_mean)
            tmp_lh += (L[i, j] - l_mean) * (h[i, j] - h_mean)
            tmp_ch += (C[i, j] - c_mean) * (h[i, j] - h_mean)

    size = L.size
    sita_ll_power_2 = tmp_ll / (size - 1)
    sita_cc_power_2 = tmp_cc / (size - 1)
    sita_hh_power_2 = tmp_hh / (size - 1)
    sita_lc_power_2 = tmp_lc / (size - 1)
    sita_lh_power_2 = tmp_lh / (size - 1)
    sita_ch_power_2 = tmp_ch / (size - 1)

    p = np.array([[sita_ll_power_2, sita_lc_power_2, sita_lh_power_2], [sita_lc_power_2,
                                                                        sita_cc_power_2, sita_ch_power_2], [sita_lh_power_2, sita_ch_power_2, sita_hh_power_2]])
    color_std = np.linalg.det(p)

    return warmc, heavyl, activep, hards, darkProportion, lightProportion, saturation, color_energy, color_std


if __name__ == '__main__':
    im = cv2.imread(sys.argv[1])
