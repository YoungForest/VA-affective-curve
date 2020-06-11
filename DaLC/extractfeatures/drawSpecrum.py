import librosa.display
import librosa
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

if __name__ == '__main__':
    # sr是采样率
    y, sr = librosa.load('C:/Users/young/Desktop/test/piano.wav', sr=11025)
    D = librosa.stft(y)

    plt.figure()
    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
    plt.title(u'钢琴')
    plt.colorbar(format='%+2.0f dB')

    y, sr = librosa.load('C:/Users/young/Desktop/test/audio1.mp3', sr=11025)
    D = librosa.stft(y)

    plt.subplot(3, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
    plt.title(u'人声')
    plt.colorbar(format='%+2.0f dB')

    y, sr = librosa.load('C:/Users/young/Desktop/test/noiseShort.wav', sr=11025)
    D = librosa.stft(y)

    plt.subplot(3, 1, 3)
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
    plt.title(u'噪音')
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig('result.png')