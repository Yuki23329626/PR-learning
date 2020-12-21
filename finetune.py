#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 05:06:13 2020

@author: marktsao
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import backend
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
from datetime import datetime
from bert import tokenization
#import nltk
#nltk.download("popular")
#from nltk.tokenize import word_tokenize

start = datetime.now()

tf.debugging.set_log_device_placement(True)
# tf.device('/device:GPU:0')

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

# TensorFlow Hub 是已訓練機器學習模型的存放區，這些模型可供微調，也可在任何地方部署。只要幾行程式碼，就能重複使用 BERT 和 Faster R-CNN 等經過訓練的模型。
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)

# Pandas 是 python 的一個數據分析 lib
# 提供高效能、簡易使用的資料格式(Data Frame)讓使用者可以快速操作及分析資料
train = pd.read_csv("/home/marktsao/DLpractice/LSTM Sentiment Analysis/using_bert_folder/train.csv")
test = pd.read_csv("/home/marktsao/DLpractice/LSTM Sentiment Analysis/using_bert_folder/test.csv")
# 要用來提交的 data
submission = pd.read_csv("/home/marktsao/DLpractice/LSTM Sentiment Analysis/using_bert_folder/sample_submission.csv")


vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

train_input = bert_encode(train.text.values, tokenizer, max_len=160)
test_input = bert_encode(test.text.values, tokenizer, max_len=160)
train_labels = train.target.values

model = build_model(bert_layer, max_len=160)
model.summary()

checkpoint = ModelCheckpoint('modelv2.h5', monitor='val_loss', save_best_only=True)

train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint],
    batch_size=16
)

model.load_weights('model.h5')
test_pred = model.predict(test_input)

submission['target'] = test_pred.round().astype(int)
submission.to_csv('submission_v2.csv', index=False)

end = datetime.now()

print("Time spent: \n", end - start)
