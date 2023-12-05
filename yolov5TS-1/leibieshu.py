##!/usr/bin/python3
# -*- coding: utf-8 -*-
# # @Time    : 2022/03/18 18:10
# # @Author  : 任思成
# # @File    : count_classes.py
#
# import os
# import matplotlib.pyplot as plt
#
# folder_path = "D:/DYJ/datasets1/labels"  # 存放Labels路径
# classes_file = "D:/DYJ/ssd-pytorch-master/model_data/voc_classes.txt"  # 存放类别文件路径
#
# classes_data = []
# file_list = os.listdir(folder_path)
#
# for classes in open(classes_file, 'r',encoding='UTF-8').readlines():
#     dr = [classes.replace("\n", ""), 0]
#     classes_data.append(dr)
#
# for item in file_list:
#     if item.split('.')[1] == "txt" and item != "classes.txt":
#         for line in open(folder_path + "/" + item, 'r').readlines():
#             for i in range(len(classes_data)):
#                 if line.split(' ')[0] == str(i):
#                     classes_data[i][1] += 1
#
# fig = plt.figure()
# fig.canvas.set_window_title('训练数据类别统计')
# plt.rcParams["font.sans-serif"] = ['SimHei']
# plt.title("训练数据类别统计")
# plt.xlabel("类别")
# plt.ylabel("数量")
# for i in range(len(classes_data)):
#     plt.bar(classes_data[i][0], classes_data[i][1])
# plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import warnings
# warnings.filterwarnings('ignore')
# import  ssl
# ssl._create_default_https_context = ssl._create_unverified_context
#
# import os
#
# txt_path = r'D:\DYJ\datasets1\labels\\'  # txt文件所在路径
# class_num = 10  # 样本类别数
# class_list = [i for i in range(class_num)]
# class_num_list = [0 for i in range(class_num)]
# labels_list = os.listdir(txt_path)
# for i in labels_list:
#     file_path = os.path.join(txt_path, i)
#     file = open(file_path, 'r')  # 打开文件
#     file_data = file.readlines()  # 读取所有行
#     for every_row in file_data:
#         string=' '.join(every_row)
#         every_row=string.strip('\n')
#         class_val = every_row.split(' ')[0]
#
#         class_ind = class_list.index(int(float(class_val)))
#         class_num_list[class_ind] += 1
#     file.close()
# # 输出每一类的数量以及总数
# print(class_num_list)
# print('total:', sum(class_num_list))
# coding=utf-8
import os
import pandas as pd

sample_dir = 'D:/DYJ/datasets1/labels/'  # 标签文件所在的路径
filenames = os.listdir(sample_dir)

class_list = []
anno_num = 0

# 遍历文件获得类别列表
for filename in filenames:
    if '.txt' in filename:
        label_file = sample_dir + '/' + filename
        with open(label_file, 'r', encoding='gbk') as f:
            for line in f.readlines():
                curLine = line.strip().split(" ")
                label = curLine[0]  # 以DOTA格式的标签为例，获取label字段（其他格式的按读取方式获取字段即可）
                if label not in class_list:
                    class_list.append(label)

class_num = len(class_list)
EachClass_Num = {}
for i in range(class_num):
    EachClass_Num[class_list[i]] = 0

for filename in filenames:
    if '.txt' in filename:
        label_file = sample_dir + '/' + filename
        with open(label_file, 'r', encoding='gbk') as f:
            for line in f.readlines():
                curLine = line.strip().split(" ")
                label_list = curLine[0]
                label = ''.join([str(x) for x in label_list])
                if label:
                    EachClass_Num[label] = EachClass_Num[label] + 1  # 统计各类别的目标个数
                else:
                    continue

print(EachClass_Num)

## 保存输出
# data_out = []
# for key in EachClass_Num:
#     k = [key, EachClass_Num[key]]
#     data_out.append(k)
#
#
# # list转dataframe
# df = pd.DataFrame(data_out, columns=['class', 'num'])
#
# # 保存到本地excel
# df.to_excel("../dota_class.xlsx", index=False, encoding='gbk')