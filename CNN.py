from tensorflow import keras
from keras import backend as K
import tensorflow as tf
import pandas as pd
from utils import *
import os
from datetime import datetime
import sklearn

from datetime import datetime as dt
import time
import numpy as np
import matplotlib.pyplot as plt

class CNN:
    def __init__(self,mfcc_shape,croma_shape,melspec_shape,contrast_shape,tonnetz_shape):
        self.mfcc_shape=mfcc_shape
        self.croma_shape = croma_shape
        self.mspec_shape=melspec_shape
        self.contrast_shape = contrast_shape
        self.tonnetz_shape = tonnetz_shape

    def create_net(self,chronic_yn,features):
        print('Creating Net...')
        mfcc_input = keras.layers.Input(shape=(self.mfcc_shape[0], self.mfcc_shape[1], 1), name="mfccInput")
        x_mfcc_1 = keras.layers.Conv2D(64, 5, strides=(1, 3), padding='same')(mfcc_input)
        x_mfcc_2 = keras.layers.BatchNormalization()(x_mfcc_1)
        x_mfcc_3 = keras.layers.Activation(keras.activations.relu)(x_mfcc_2)
        x_mfcc_4 = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x_mfcc_3)

        x_mfcc_5 = keras.layers.Conv2D(64, 3, strides=(1, 2), padding='same')(x_mfcc_4)
        x_mfcc_6 = keras.layers.BatchNormalization()(x_mfcc_5)
        x_mfcc_7 = keras.layers.Activation(keras.activations.relu)(x_mfcc_6)
        x_mfcc_8 = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x_mfcc_7)

        x_mfcc_9 = keras.layers.Conv2D(96, 3, padding='same')(x_mfcc_8)
        x_mfcc_10 = keras.layers.BatchNormalization()(x_mfcc_9)
        x_mfcc_11 = keras.layers.Activation(keras.activations.relu)(x_mfcc_10)
        x_mfcc_12 = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x_mfcc_11)

        x_mfcc_13 = keras.layers.Conv2D(96, 3, padding='same')(x_mfcc_12)
        x_mfcc_14 = keras.layers.BatchNormalization()(x_mfcc_13)
        x_mfcc_15 = keras.layers.Activation(keras.activations.relu)(x_mfcc_14)
        mfcc_output = keras.layers.GlobalMaxPooling2D()(x_mfcc_15)

        mfcc_model = keras.Model(mfcc_input, mfcc_output, name="mfccModel")

        croma_input = keras.layers.Input(shape=(self.croma_shape[0], self.croma_shape[1], 1),
                                         name="cromaInput")

        x_croma_1 = keras.layers.Conv2D(64, 5, strides=(1, 3), padding='same')(croma_input)
        x_croma_2 = keras.layers.BatchNormalization()(x_croma_1)
        x_croma_3 = keras.layers.Activation(keras.activations.relu)(x_croma_2)
        x_croma_4 = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x_croma_3)

        x_croma_5 = keras.layers.Conv2D(64, 3, strides=(1, 2), padding='same')(x_croma_4)
        x_croma_6 = keras.layers.BatchNormalization()(x_croma_5)
        x_croma_7 = keras.layers.Activation(keras.activations.relu)(x_croma_6)
        x_croma_8 = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x_croma_7)

        x_croma_9 = keras.layers.Conv2D(96, 3, padding='same')(x_croma_8)
        x_croma_10 = keras.layers.BatchNormalization()(x_croma_9)
        x_croma_11 = keras.layers.Activation(keras.activations.relu)(x_croma_10)
        x_croma_12 = keras.layers.MaxPooling2D()(x_croma_11)

        x_croma_13 = keras.layers.Conv2D(96, 3, padding='same')(x_croma_12)
        x_croma_14 = keras.layers.BatchNormalization()(x_croma_13)
        x_croma_15 = keras.layers.Activation(keras.activations.relu)(x_croma_14)
        croma_output = keras.layers.GlobalMaxPooling2D()(x_croma_15)

        croma_model = keras.Model(croma_input, croma_output, name="cromaModel")


        mspec_input = keras.layers.Input(shape=(self.mspec_shape[0], self.mspec_shape[1], 1),
                                         name="mspecInput")
        x = keras.layers.Conv2D(64, 5, strides=(1, 3), padding='same')(mspec_input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(keras.activations.relu)(x)
        x = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x)

        x = keras.layers.Conv2D(64, 3, strides=(1, 2), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(keras.activations.relu)(x)
        x = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x)

        x = keras.layers.Conv2D(96, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(keras.activations.relu)(x)
        x = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x)

        x = keras.layers.Conv2D(96, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(keras.activations.relu)(x)
        mspec_output = keras.layers.GlobalMaxPooling2D()(x)

        mspec_model = keras.Model(mspec_input, mspec_output, name="mspecModel")

        contrast_input = keras.layers.Input(shape=(self.contrast_shape[0], self.contrast_shape[1], 1), name="contrastInput")
        x_contrast_1 = keras.layers.Conv2D(64, 2, strides=(1, 3), padding='same')(contrast_input)
        x_contrast_2 = keras.layers.BatchNormalization()(x_contrast_1)
        x_contrast_3 = keras.layers.Activation(keras.activations.relu)(x_contrast_2)
        x_contrast_4 = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x_contrast_3)

        x_contrast_5 = keras.layers.Conv2D(64, 2, strides=(1, 3), padding='same')(x_contrast_4)
        x_contrast_6 = keras.layers.BatchNormalization()(x_contrast_5)
        x_contrast_7 = keras.layers.Activation(keras.activations.relu)(x_contrast_6)

        contrast_output = keras.layers.GlobalMaxPooling2D()(x_contrast_7)

        contrast_model = keras.Model(contrast_input, contrast_output, name="contrastModel")


        tonnetz_input = keras.layers.Input(shape=(self.tonnetz_shape[0], self.tonnetz_shape[1], 1), name="tonnetzInput")
        x_tonnetz_1 = keras.layers.Conv2D(64, 2, strides=(1, 3), padding='same')(tonnetz_input)
        x_tonnetz_2 = keras.layers.BatchNormalization()(x_tonnetz_1)
        x_tonnetz_3 = keras.layers.Activation(keras.activations.relu)(x_tonnetz_2)
        x_tonnetz_4 = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x_tonnetz_3)

        x_tonnetz_5 = keras.layers.Conv2D(64, 2, strides=(1, 3), padding='same')(x_tonnetz_4)
        x_tonnetz_6 = keras.layers.BatchNormalization()(x_tonnetz_5)
        x_tonnetz_7 = keras.layers.Activation(keras.activations.relu)(x_tonnetz_6)
        tonnetz_output = keras.layers.GlobalMaxPooling2D()(x_tonnetz_7)

        tonnetz_model = keras.Model(tonnetz_input, tonnetz_output, name="tonnetzModel")


        input_mfcc = keras.layers.Input(shape=(self.mfcc_shape[0], self.mfcc_shape[1], 1), name="mfcc")
        mfcc = mfcc_model(input_mfcc)

        input_croma = keras.layers.Input(shape=(self.croma_shape[0], self.croma_shape[1], 1), name="croma")
        croma = croma_model(input_croma)

        input_mspec = keras.layers.Input(shape=(self.mspec_shape[0], self.mspec_shape[1], 1), name="mspec")
        mspec = mspec_model(input_mspec)

        input_contrast = keras.layers.Input(shape=(self.contrast_shape[0], self.contrast_shape[1], 1), name="contrast")
        contrast = contrast_model(input_contrast)

        input_tonnetz = keras.layers.Input(shape=(self.tonnetz_shape[0], self.tonnetz_shape[1], 1), name="tonnetz")
        tonnetz = tonnetz_model(input_tonnetz)

        features_concat=[]
        for feature in eval(features):
            features_concat.append(eval(feature))

        concat=keras.layers.concatenate(features_concat)
        hidden = keras.layers.Dense(256, activation='relu')(concat)
        hidden = keras.layers.Dropout(0.6)(hidden)
        hidden = keras.layers.Dense(128, activation='relu')(hidden)
        hidden = keras.layers.Dropout(0.3)(hidden)
        hidden = keras.layers.Dense(64, activation='relu')(hidden)
        hidden = keras.layers.Dropout(0.15)(hidden)
        hidden = keras.layers.Dense(32, activation='relu')(hidden)
        hidden = keras.layers.Dropout(0.075)(hidden)
        hidden = keras.layers.Dense(16, activation='relu')(hidden)
        hidden = keras.layers.Dropout(0.0325)(hidden)

        output = keras.layers.Dense(6, activation='softmax')(hidden)
        output_chronic = keras.layers.Dense(3, activation='softmax')(hidden)

        # model_file='C:/Users/Joaquin/BIG_DATA_analysis_master/TFM/modelo.png'

        # keras.utils.plot_model(net,  to_file=model_file, show_shapes=True)
        input_features=[]
        for feature in eval(features):
            input_features.append(eval('input_'+feature))


        if chronic_yn == False:
            net = keras.Model(input_features, output, name="Net")
        else:
            net = keras.Model(input_features, output_chronic, name="Net")

        self.net=net
        self.chronic_yn=chronic_yn

    def run_net(self, parameters,epochs,features):

        self.net.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        K.set_value(self.net.optimizer.learning_rate, 0.001)

        my_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3,monitor='val_accuracy',restore_best_weights=True)
        ]
        print('Running net algorithm...')
        start=time.time()
        train_set=[]
        test_set=[]
        val_set=[]
        for feature in eval(features):
            evaluation_str_train='parameters.'+feature+'_train'
            evaluation_str_val = 'parameters.' + feature + '_val'
            evaluation_str_test= 'parameters.' + feature + '_test'
            train_set.append(eval(evaluation_str_train))
            test_set.append(eval(evaluation_str_test))
            val_set.append(eval(evaluation_str_val))


        history = self.net.fit(
            train_set,
            parameters.ytrain,
            validation_data=(val_set, parameters.yval),
            epochs=epochs, verbose=1)
        end=time.time()

        test_dict={}
        for feature in eval(features):
            str_eval="parameters."+feature+"_test"
            test_dict[feature]=eval(str_eval)
        acc = self.net.evaluate(test_dict, parameters.ytest.values)

        y_pred = self.net.predict(test_dict)

        matrix = sklearn.metrics.confusion_matrix(parameters.ytest.values, y_pred.argmax(axis=1))

        cm_sum = np.sum(matrix, axis=1, keepdims=True)
        cm_perc = matrix / cm_sum.astype(float) * 100

        date_and_time=datetime.now().isoformat(sep='_',timespec='minutes').replace(':','-')
        features_saving=''
        for feature in eval(features):
            features_saving=features_saving+'_'+str(feature)

        if self.chronic_yn==True:
            path_saving=path_experiments+'/'+date_and_time+features_saving+'_chronic_'+str(epochs)
        else:
            path_saving = path_experiments + '/' + date_and_time + features_saving + '_diseases_'+str(epochs)
        os.mkdir(path_saving)

        plt.figure(figsize=(7, 5))
        for key in ['val_accuracy', 'accuracy']:
            plt.plot(history.history[key], label=key)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.gca().set_ylim(0.2, 1.1)
        plt.savefig(path_saving + '/history_accuracy.png')

        plt.figure(figsize=(7,5))
        for key in ['val_loss', 'loss']:
            plt.plot(history.history[key], label=key)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.gca().set_ylim(0, 1.8)
        plt.savefig(path_saving + '/history_loss.png')

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.matshow(cm_perc, cmap=plt.cm.Blues, alpha=0.9)
        ax.set_xticks(range(len(list(dict_diseases_numbers.keys()))))
        ax.xaxis.set_ticklabels(list(dict_diseases_numbers.keys()))
        ax.set_yticks(range(len(list(dict_diseases_numbers.keys()))))
        ax.yaxis.set_ticklabels(list(dict_diseases_numbers.keys()))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(x=j, y=i, s=np.round(cm_perc[i, j], 2), va='center', ha='center', size='xx-large')

        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        plt.tight_layout()
        plt.savefig(path_saving + '/matr_normal.png')


        df_history=pd.DataFrame()
        for parameter in history.history:
            df_history[parameter]=history.history[parameter]
        if self.chronic_yn==True:
            df_history['type']="Chronic"
        else:
            df_history['type'] = "Disease"
        df_history['test_loss'] = acc[0]
        df_history['test_accuracy']=acc[1]

        df_history['delay (min)']=np.round((end-start)/60,2)
        df_history['n_epochs']=epochs
        df_history['features']=features

        df_history.to_csv(path_saving+'/df_history.csv',sep=';')

        return history