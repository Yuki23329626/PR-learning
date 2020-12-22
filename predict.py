#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 05:06:13 2020

@author: marktsao
"""

import warnings

warnings.filterwarnings('ignore')

import os
os.environ['PYTHONHASHSEED']='0'
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TF_CUDNN_USE_AUTOTUNE']='0'
os.environ['KMP_WARNINGS']='off'
import numpy as np
import tensorflow as tf
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
import matplotlib.pyplot as plt

from bert import tokenization

#import nltk
#nltk.download("popular")
#from nltk.tokenize import word_tokenize

# LEARNING_CURVE="eal_disaster_learning_curve"
# LOSS_CURVE="treal_disaster_loss_curve"
# LEARNING_CURVE_T="real_disaster_learning_curve_only_test"
# LOSS_CURVE_T="real_disaster_loss_curve_only_test"

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def build_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)

# train = pd.read_csv("/home/marktsao/DLpractice/LSTM Sentiment Analysis/using_bert_folder/train.csv")
# test = pd.read_csv("/home/marktsao/DLpractice/LSTM Sentiment Analysis/using_bert_folder/test.csv")
# submission = pd.read_csv("/home/marktsao/DLpractice/LSTM Sentiment Analysis/using_bert_folder/sample_submission.csv")

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

# train_input = bert_encode(train.text.values, tokenizer, max_len=160)
# test_input = bert_encode(test.text.values, tokenizer, max_len=160)
# train_labels = train.target.values

model = build_model(bert_layer, max_len=160)
model.summary()

# checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

# train_history = model.fit(
#     train_input, train_labels,
#     validation_split=0.2,
#     epochs=1,
#     callbacks=[checkpoint],
#     batch_size=16
# )

model.load_weights('model.h5')
# test_pred = model.predict(test_input)

# submission['target'] = test_pred.round().astype(int)
# submission.to_csv('submission.csv', index=False)

# train_acc_list = []     # Save train history of train accuracy
# test_acc_list = []      # Save test history of test accuracy
# def show_train_history_acc(train_acc,test_acc):
#     plt.plot(train_history.history[train_acc],'b')
#     train_acc_list.extend(train_history.history[train_acc])
#     #print("train_acc_list =", train_acc_list)
#     plt.plot(train_history.history[test_acc],'r')
#     test_acc_list.extend(train_history.history[test_acc])
#     #print("test_acc_list =", test_acc_list)
    
#     plt.title('Learning curve')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(['train','test'], loc='lower right')
#     plt.savefig(LEARNING_CURVE)
#     plt.show()
# show_train_history_acc('accuracy', 'val_accuracy')

# train_loss_list = []     # Save train history of train accuracy
# test_loss_list = []      # Save test history of test accuracy
# def show_train_history_acc(train_loss,test_loss):
#     plt.plot(train_history.history[train_loss],'b')
#     train_acc_list.extend(train_history.history[train_loss])
#     #print("train_acc_list =", train_acc_list)
#     plt.plot(train_history.history[test_loss],'r')
#     test_acc_list.extend(train_history.history[test_loss])
#     #print("test_acc_list =", test_acc_list)
    
#     plt.title('Loss curve')
#     plt.xlabel('Epoch')
#     plt.legend(['train','test'], loc='upper right')
#     plt.savefig(LOSS_CURVE)
#     plt.show()
# show_train_history_acc('loss', 'val_loss')

# train_acc_list = []     # Save train history of train accuracy
# test_acc_list = []      # Save test history of test accuracy
# def show_train_history_acc(train_acc,test_acc):
#     plt.plot(train_history.history[train_acc],'b')
#     train_acc_list.extend(train_history.history[train_acc])
#     #print("train_acc_list =", train_acc_list)
#     #plt.plot(train_history.history[test_acc],'r')
#     #test_acc_list.extend(train_history.history[test_acc])
#     #print("test_acc_list =", test_acc_list)
    
#     plt.title('Learning curve')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(['train'], loc='lower right')
#     plt.savefig(LEARNING_CURVE_T)
#     plt.show()
# show_train_history_acc('accuracy', 'val_accuracy')

# train_loss_list = []     # Save train history of train accuracy
# test_loss_list = []      # Save test history of test accuracy
# def show_train_history_acc(train_loss,test_loss):
#     plt.plot(train_history.history[train_loss],'b')
#     train_acc_list.extend(train_history.history[train_loss])
#     #print("train_acc_list =", train_acc_list)
#     #plt.plot(train_history.history[test_loss],'r')
#     #test_acc_list.extend(train_history.history[test_loss])
#     #print("test_acc_list =", test_acc_list)
    
#     plt.title('Loss curve')
#     plt.xlabel('Epoch')
#     plt.legend(['train'], loc='upper right')
#     plt.savefig(LOSS_CURVE_T)
#     plt.show()
# show_train_history_acc('loss', 'val_loss')


s = input("Enter sentence to predict:")
s = np.array([s], dtype=object);
bs = bert_encode(s, tokenizer, max_len=160)
bs_pred = model.predict(bs);
print("Result(1 for disaster,0 for non-disaster):",bs_pred)
yn = input("If you want to predict again, please enter y otherwise enter n:");

while (yn == 'y'):
    s = input("Enter sentence to predict:")
    s = np.array([s], dtype=object);
    bs = bert_encode(s, tokenizer, max_len=160)
    bs_pred = model.predict(bs);
    print("Result(1 for disaster,0 for non-disaster):",bs_pred)
    yn = input("If you want to predict again, please enter y otherwise enter n:");
    