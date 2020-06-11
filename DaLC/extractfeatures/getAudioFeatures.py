import librosa
import librosa.display
import matplotlib.pyplot as plt
import math
import numpy as np
import sys

def getAudioFeatures(audio_path):
    '''
        return 声音和谐度, 声音能量, 频率质心, 音频对比度, 过零率, 静音比率
    '''
    # http://librosa.github.io/librosa/feature.html

    # sr是采样率
    y, sr = librosa.load(audio_path, sr=11025)
    D = librosa.stft(y)

    # 声音和谐度
    # amplitue = librosa.amplitude_to_db(D, ref=np.max)
    amplitue = D

    M = np.zeros_like(amplitue)

    for i in range(amplitue.shape[0]):
        for j in range (2, amplitue.shape[1] - 2):
            if amplitue[i][j] > amplitue[i][j-1] and amplitue[i][j] > amplitue[i][j-2] and amplitue[i][j] > amplitue[i][j+1] and amplitue[i][j] > amplitue[i][j+2]:
                    M[i][j] = 1

    width = 0
    H = 0
    for j in range(amplitue.shape[1]):
        for i in range(amplitue.shape[0]):
            if M[i][j] == 1:
                width += 1
            else:
                if width != 0:
                    H += (width - 1)**2
                    width = 0
        if width != 0:
            H += (width - 1)**2
            width = 0
        
    H /= len(y)

    # 静音比例
    winSize = round(sr / 10)
    N = math.floor(len(y) / winSize)
    y_power_2 = np.power(y, 2)
    silenceCount = 0
    for i in range(1, N):
        if np.mean(y_power_2[(i-1)*winSize+1:i*winSize]) < 0.0005:
            silenceCount += 1

    silencePropertion = silenceCount / N
    rmse = librosa.feature.rms(y)
    centroid = librosa.feature.spectral_centroid(y, sr=sr)
    constrast = librosa.feature.spectral_contrast(y, sr=sr, fmin=100.0, n_bands=2)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

    return H, rmse.mean(), centroid.mean(), constrast.mean(axis=1), zero_crossing_rate.mean(), silencePropertion


if __name__=='__main__':
    audio_path = sys.argv[1]
    # audio_path = 'noiseShort.wav'
    print('Audio feauture of %s: %s' %(audio_path, getAudioFeatures(audio_path)))
    