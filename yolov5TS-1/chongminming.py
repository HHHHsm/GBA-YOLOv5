import os

path = r'D:\DYJ\99Z - 副本\labels\train-txt'  # 首先定义文件夹的路径
file_names = os.listdir(path)  # 创建一个所有文件名的列表

i = 1
for name in file_names:
    i = int(i)
    if i < 10:
        i = '00' + str(i)
    elif 10 <= i and i < 100:
        i = '0' + str(i)
    else:
        i = str(i)  # 001，002,，003......的循环，我比较喜欢采取这种方式命名
    photo_name = str(name).split('.')[0]
    # 我整理照片，photo_name是指文件不含.jpg的名字，split('.')字符按.划分成两个，[0]前[1]后
    photo_format = str(name).split('.')[1]
    new_name = 'DD' + i + '.' + photo_format  # 新名字加上序号
    os.rename(os.path.join(path, name), os.path.join(path, new_name))  # 执行重命名
    i = int(i) + 1