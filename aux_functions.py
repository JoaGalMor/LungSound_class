
import numpy as np
def getFilenameInfo(file):
    return file.split('_')


def getFileName(file):
    return file.split('/')


def per_class_accuracy(y_preds, y_true, class_labels):
    return [np.mean([
        (y_true[pred_idx] == np.round(y_pred)) for pred_idx, y_pred in enumerate(y_preds)
        if y_true[pred_idx] == int(class_label)
    ]) for class_label in class_labels]


def getfilename_2(file):
    names = file.split('_')
    name = ''
    for i in range(5):
        if i == 0:
            name = name + names[i]
        else:
            name = name + '_' + names[i]
    return name


def getFeatures(path):
    soundArr, sample_rate = lb.load(path)
    mfcc = lb.feature.mfcc(y=soundArr, sr=sample_rate)
    croma = lb.feature.chroma_stft(y=soundArr, sr=sample_rate)
    mSpec = lb.feature.melspectrogram(y=soundArr, sr=sample_rate)
    # coeff, freq=pywt.cwt(soundArr,range(1,50), 'gaus1')
    # coeff_ = coeff[:,:49]
    return mfcc, croma, mSpec


def getPureSample(raw_data, start, end, sr=22050):
    '''
    Takes a numpy array and spilts its using start and end args

    raw_data=numpy array of audio sample
    start=time
    end=time
    sr=sampling_rate
    mode=mono/stereo

    '''
    max_ind = len(raw_data)
    start_ind = min(int(start * sr), max_ind)
    end_ind = min(int(end * sr), max_ind)
    return raw_data[start_ind: end_ind]


def extractId(filename):
    return filename.split('_')[0]