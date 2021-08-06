from tensorflow import keras
from keras import backend as K
import tensorflow as tf
import pandas as pd
from utils import *
import os
import datetime
import sklearn
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

class CNN:
    def __init__(self,mfcc_shape,croma_shape,melSpec_shape):
        self.mfcc_shape=mfcc_shape
        self.croma_shape = croma_shape
        self.mSpec_shape=melSpec_shape
    def create_net(self,chronic_yn):
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

        mSpec_input = keras.layers.Input(shape=(self.mSpec_shape[0], self.mSpec_shape[1], 1),
                                         name="mSpecInput")
        x = keras.layers.Conv2D(64, 5, strides=(2, 3), padding='same')(mSpec_input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(keras.activations.relu)(x)
        x = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x)

        x = keras.layers.Conv2D(64, 3, strides=(2, 2), padding='same')(x)
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
        mSpec_output = keras.layers.GlobalMaxPooling2D()(x)

        mSpec_model = keras.Model(mSpec_input, mSpec_output, name="mSpecModel")

        input_mfcc = keras.layers.Input(shape=(self.mfcc_shape[0], self.mfcc_shape[1], 1), name="mfcc")
        mfcc = mfcc_model(input_mfcc)

        input_croma = keras.layers.Input(shape=(self.croma_shape[0], self.croma_shape[1], 1), name="croma")
        croma = croma_model(input_croma)

        input_mSpec = keras.layers.Input(shape=(self.mSpec_shape[0], self.mSpec_shape[1], 1), name="mspec")
        mSpec = mSpec_model(input_mSpec)

        concat = keras.layers.concatenate([mfcc, croma, mSpec])
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

        if chronic_yn == False:
            net = keras.Model([input_mfcc, input_croma, input_mSpec], output, name="Net")
        else:
            net = keras.Model([input_mfcc, input_croma, input_mSpec], output_chronic, name="Net")
        self.net=net
        self.chronic_yn=chronic_yn

    def run_net(self, parameters,epochs):

        self.net.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        K.set_value(self.net.optimizer.learning_rate, 0.001)

        my_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10,monitor='val_loss'),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                      patience=4, min_lr=0.00001,mode='min')
        ]

        history = self.net.fit(
            [parameters.mfcc_train, parameters.croma_train, parameters.mSpec_train],
            parameters.ytrain,
            validation_data=([parameters.mfcc_val, parameters.croma_val, parameters.mSpec_val], parameters.yval),
            epochs=epochs, verbose=1,callbacks=my_callbacks)

        acc = self.net.evaluate({"mfcc": parameters.mfcc_test, "croma": parameters.croma_test, "mspec": parameters.mSpec_test}, parameters.ytest.values)

        y_pred = self.net.predict({"mfcc": parameters.mfcc_test, "croma": parameters.croma_test, "mspec": parameters.mSpec_test})

        matrix = sklearn.metrics.confusion_matrix(parameters.ytest.values, y_pred.argmax(axis=1))

        cm_sum = np.sum(matrix, axis=1, keepdims=True)
        cm_perc = matrix / cm_sum.astype(float) * 100

        date_and_time=datetime.now().isoformat(sep='_',timespec='minutes').replace(':','-')
        path_saving=path_experiments+'/'+date_and_time
        os.mkdir(path_saving)

        plt.figure(figsize=(9, 6))
        for key in ['val_accuracy', 'accuracy']:
            plt.plot(history.history[key], label=key)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.gca().set_ylim(0.2, 1.1)
        plt.savefig(path_saving + '/history_accuracy.png')

        plt.figure(figsize=(9, 6))
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

        df_history['type']=self.chronic_yn
        df_history['test_loss'] = acc[0]
        df_history['test_accuracy']=acc[1]

        df_history.to_csv(path_saving+'/df_history.csv')

        return history