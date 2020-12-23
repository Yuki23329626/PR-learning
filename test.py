#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import warnings

# warnings.filterwarnings('ignore')

# import os
# os.environ['PYTHONHASHSEED']='0'
# os.environ['CUDA_VISIBLE_DEVICES']='0'
# os.environ['TF_CUDNN_USE_AUTOTUNE']='0'
# os.environ['KMP_WARNINGS']='off'
# os.environ['TF_XLA_FLAGS']='--tf_xla_enable_xla_devices'
# import numpy as np
# import tensorflow as tf
# print("Version: ", tf.__version__)
# print("Eager mode: ", tf.executing_eagerly())
# print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

import numpy as np
import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.layers import Dense, Input
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import ModelCheckpoint
# import tensorflow_hub as hub
import matplotlib.pyplot as plt

# from bert import tokenization
# from datetime import datetime

# #import nltk
# #nltk.download("popular")
# #from nltk.tokenize import word_tokenize

# start = datetime.now()

# tf.debugging.set_log_device_placement(True)
# # tf.device('/device:GPU:0')

# def bert_encode(texts, tokenizer, max_len=512):
#     all_tokens = []
#     all_masks = []
#     all_segments = []
    
#     for text in texts:
#         text = tokenizer.tokenize(text)
            
#         text = text[:max_len-2]
#         input_sequence = ["[CLS]"] + text + ["[SEP]"]
#         pad_len = max_len - len(input_sequence)
        
#         tokens = tokenizer.convert_tokens_to_ids(input_sequence)
#         tokens += [0] * pad_len
#         pad_masks = [1] * len(input_sequence) + [0] * pad_len
#         segment_ids = [0] * max_len
        
#         all_tokens.append(tokens)
#         all_masks.append(pad_masks)
#         all_segments.append(segment_ids)
    
#     return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

# def build_model(bert_layer, max_len=512):
#     input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
#     input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
#     segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

#     _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
#     clf_output = sequence_output[:, 0, :]
#     out = Dense(1, activation='sigmoid')(clf_output)
    
#     model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
#     model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
#     return model

# # TensorFlow Hub 是已訓練機器學習模型的存放區，這些模型可供微調，也可在任何地方部署。只要幾行程式碼，就能重複使用 BERT 和 Faster R-CNN 等經過訓練的模型。
# module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
# bert_layer = hub.KerasLayer(module_url, trainable=True)

# # Pandas 是 python 的一個數據分析 lib
# # 提供高效能、簡易使用的資料格式(Data Frame)讓使用者可以快速操作及分析資料
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
# 要用來提交的 data
submission = pd.read_csv("sample_submission.csv")

# names = list(set(train.keyword.values))
names = []
values = []

keys = set(train.keyword)
values = [0]*len(keys)
train_dict = dict(zip(keys, values))

keys = set(test.keyword)
values = [0]*len(keys)
test_dict = dict(zip(keys, values))

# print(train_dict)
# print(test_dict)

for keyword in train.keyword:
    train_dict[keyword] += 1

for keyword in test.keyword:
    test_dict[keyword] += 1

# print(train_dict)
# print(list(train_dict.keys()))
# print(list(train_dict.values()))
# print(np.arange(0, len(set(train.keyword))))
# print(list(train_dict.values()))
plt.figure(figsize=(16, 8))
plt.subplot(111)
plt.plot(np.arange(0, len(set(train.keyword))),list(train_dict.values()),np.arange(0, len(set(test.keyword))),list(test_dict.values()))
plt.suptitle('Numeber of each Category')
plt.show()