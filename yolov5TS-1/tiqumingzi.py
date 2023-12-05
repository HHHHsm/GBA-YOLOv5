
# "D:/dataccc/cc/images/train"
import os

path = "D:/DYJ/99/images/val"
file_name_list = os.listdir(path)

with open('D:/DYJ/VOC2007/ImageSets/Main/val.txt', 'w') as f:
    for file_name in file_name_list:
        f.write(file_name[0:-4] + '\n')  # -5表示从后往前数，到小数点位置

# import os
#
# path = r'C:\Users\asus\Desktop\specair\specair_spectrum'  # 文件路径
#
# filenames = os.listdir(path)
# filenames.sort(key=lambda x: int(x[:-4]))    # 解决自动排序问题：按照多少维排序
#
# f = open('234.txt', 'w')       # 打开234.txt文件，进行写入
# for name in filenames:
#     name = name.split(".")[0]  # 去后缀名
#     print(name)                # 查看是否去后缀名成功
#     f.write(name + '\n')       # 写入txt文件中
# f.close()
