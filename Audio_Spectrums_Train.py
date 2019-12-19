import wave
import numpy as np
import pylab as plt
import cv2


wavs_path = 'D:\\iTsinghua\\Major\\Visual&Audio\\hwk\\comprehensive_hwk3\\dataset\\train'
N_pos = 100
N_neg = 100
size_spec_pic = 138

spectrum_pictures = np.ndarray((N_pos+N_neg, size_spec_pic, size_spec_pic))


def spect_resize(spect):
    A = np.copy(spect)
    B = cv2.resize(A.astype('float'), (size_spec_pic, size_spec_pic), interpolation=cv2.INTER_CUBIC)
    return B


# Open all the wave file.
for classes in range(2):
    if classes == 0:
        class_name = 'positive'
    else:
        class_name = 'negative'

    for k in range(N_pos):
        path = wavs_path + '\\' + class_name + '\\' + str(k) + '\\audio.wav'
        wav_file = wave.open(path, 'rb')
        params = wav_file.getparams()
        nchannels, samplewidth, framerate, nframes = params[:4]
        str_data = wav_file.readframes(nframes)
        wav_file.close()

        wave_data = np.fromstring(str_data, dtype=np.short)
        wave_data = wave_data * 1.0/max(abs(wave_data))
        wave_data.shape = (nframes,)

        spect_data, freqs, ts, fig = plt.specgram(wave_data, NFFT=4096, Fs=framerate, noverlap=2048)

        spect_data = np.log10(np.abs(spect_data[0:1000, :]))
        maxim = np.max(np.max(spect_data, axis=0), axis=0)
        minim = np.min(np.min(spect_data, axis=0), axis=0)
        # spect_data = spect_data * 1.0 / maxim
        # spect_data = 2*(spect_data - minim)/(maxim-minim)-1

        index = k if classes == 0 else k+N_pos
        spect_data = spect_resize(spect_data)
        spectrum_pictures[index, :, :] = spect_data
        print('index %d completed.' % index)

np.save('train_data', spectrum_pictures)




