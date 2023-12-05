import os
import xml.etree.ElementTree as ET
from decimal import Decimal

dirpath = 'D:\\DYJ\\99Z\\labels\\test'  # 原来存放xml文件的目录
newdir = 'D:\\DYJ\\99Z\\labels\\test-txt'  # 修改label后形成的txt目录
#classes=  ['plate','0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z','澳','川','鄂','甘','赣','港','贵','桂','黑','沪','吉','冀','津','晋','京','警','辽','鲁','蒙','闽','宁','青','琼','陕','苏','皖','湘','新','学','渝','豫','粤','云','浙','藏']
classes=  ['右手打电话','右手打字','左手打电话','左手打字','调收音机','喝东西','和其他乘客说话','拿后面的东西','Scratchface','Yawn','Glasscleaning','Smoking','rightVoice','leftVoice']
# classes=  ['fruit','cdb','wawa','jiuping','yaohe']
# classes=  ['car', 'bus', 'person', 'bike', 'truck', 'motor','train','rider','traffic sign','traffic light']
# classes=  ['First', 'Second', 'Third', 'Fourth']
# classes=  ['Cracking', 'Deglazing', 'Falling slag', 'missing corner']
if not os.path.exists(newdir):
    os.makedirs(newdir)

for fp in os.listdir(dirpath):

    root = ET.parse(os.path.join(dirpath, fp)).getroot()

    xmin, ymin, xmax, ymax = 0, 0, 0, 0
    sz = root.find('size')
    width = float(sz[0].text)
    height = float(sz[1].text)
    filename = root.find('filename').text
    print(fp)
    with open(os.path.join(newdir, fp.split('.')[0] + '.txt'), 'a+') as f:
        for child in root.findall('object'):  # 找到图片中的所有框

            sub = child.find('bndbox')  # 找到框的标注值并进行读取
            sub_label = child.find('name')
            print(sub_label.text)
            xmin = float(sub[0].text)
            ymin = float(sub[1].text)
            xmax = float(sub[2].text)
            ymax = float(sub[3].text)
            try:  # 转换成yolov的标签格式，需要归一化到（0-1）的范围内
                x_center = Decimal(str(round(float((xmin + xmax) / (2 * width)), 6))).quantize(Decimal('0.000000'))
                y_center = Decimal(str(round(float((ymin + ymax) / (2 * height)), 6))).quantize(Decimal('0.000000'))
                w = Decimal(str(round(float((xmax - xmin) / width), 6))).quantize(Decimal('0.000000'))
                h = Decimal(str(round(float((ymax - ymin) / height), 6))).quantize(Decimal('0.000000'))
                print(str(x_center) + ' ' + str(y_center) + ' ' + str(w) + ' ' + str(h))

                # 读取需要的标签\
                # if sub_label.text == 'car':
                #     f.write(' '.join([str(0), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == 'bus':
                #     f.write(' '.join([str(1), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == 'person':
                #     f.write(' '.join([str(2), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == 'bike':
                #     f.write(' '.join([str(3), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == 'truck':
                #     f.write(' '.join([str(4), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == 'motor':
                #     f.write(' '.join([str(5), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == 'train':
                #     f.write(' '.join([str(6), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == 'rider':
                #     f.write(' '.join([str(7), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == 'traffic sign':
                #     f.write(' '.join([str(8), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == 'traffic light':
                #     f.write(' '.join([str(9), str(x_center), str(y_center), str(w), str(h) + '\n']))


                # if sub_label.text == '0':
                #     f.write(' '.join([str(0), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '1':
                #     f.write(' '.join([str(1), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '2':
                #     f.write(' '.join([str(2), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '3':
                #     f.write(' '.join([str(3), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '4':
                #     f.write(' '.join([str(4), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '5':
                #     f.write(' '.join([str(5), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '6':
                #     f.write(' '.join([str(6), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '7':
                #     f.write(' '.join([str(7), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '8':
                #     f.write(' '.join([str(8), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '9':
                #     f.write(' '.join([str(9), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '10':
                #     f.write(' '.join([str(10), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '11':
                #     f.write(' '.join([str(11), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '12':
                #     f.write(' '.join([str(12), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '13':
                #     f.write(' '.join([str(13), str(x_center), str(y_center), str(w), str(h) + '\n']))

                if sub_label.text == '1':
                    f.write(' '.join([str(0), str(x_center), str(y_center), str(w), str(h) + '\n']))
                if sub_label.text == '2':
                    f.write(' '.join([str(1), str(x_center), str(y_center), str(w), str(h) + '\n']))
                if sub_label.text == '3':
                    f.write(' '.join([str(2), str(x_center), str(y_center), str(w), str(h) + '\n']))
                if sub_label.text == '4':
                    f.write(' '.join([str(3), str(x_center), str(y_center), str(w), str(h) + '\n']))
                if sub_label.text == '5':
                    f.write(' '.join([str(4), str(x_center), str(y_center), str(w), str(h) + '\n']))
                if sub_label.text == '6':
                    f.write(' '.join([str(5), str(x_center), str(y_center), str(w), str(h) + '\n']))
                if sub_label.text == '7':
                    f.write(' '.join([str(6), str(x_center), str(y_center), str(w), str(h) + '\n']))
                if sub_label.text == '8':
                    f.write(' '.join([str(7), str(x_center), str(y_center), str(w), str(h) + '\n']))
                if sub_label.text == '9':
                    f.write(' '.join([str(8), str(x_center), str(y_center), str(w), str(h) + '\n']))
                if sub_label.text == '10':
                    f.write(' '.join([str(9), str(x_center), str(y_center), str(w), str(h) + '\n']))
                if sub_label.text == '11':
                    f.write(' '.join([str(10), str(x_center), str(y_center), str(w), str(h) + '\n']))
                if sub_label.text == '12':
                    f.write(' '.join([str(11), str(x_center), str(y_center), str(w), str(h) + '\n']))
                if sub_label.text == '13':
                    f.write(' '.join([str(12), str(x_center), str(y_center), str(w), str(h) + '\n']))
                if sub_label.text == '14':
                    f.write(' '.join([str(13), str(x_center), str(y_center), str(w), str(h) + '\n']))

                # if sub_label.text == '右手打电话':
                #     f.write(' '.join([str(0), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '右手打字':
                #     f.write(' '.join([str(1), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '左手打电话':
                #     f.write(' '.join([str(2), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '左手打字':
                #     f.write(' '.join([str(3), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '调收音机':
                #     f.write(' '.join([str(4), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '喝东西':
                #     f.write(' '.join([str(5), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '和其他乘客说话':
                #     f.write(' '.join([str(6), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '拿后面的东西':
                #     f.write(' '.join([str(7), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == 'Scratchface':
                #     f.write(' '.join([str(8), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == 'Yawn':
                #     f.write(' '.join([str(9), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == 'Glasscleaning':
                #     f.write(' '.join([str(10), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == 'Smoking':
                #     f.write(' '.join([str(11), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == 'rightVoice':
                #     f.write(' '.join([str(12), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == 'leftVoice':
                #     f.write(' '.join([str(13), str(x_center), str(y_center), str(w), str(h) + '\n']))
                #
                # if sub_label.text == '5':
                #     f.write(' '.join([str(4), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '6':
                #     f.write(' '.join([str(5), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '1':
                #     f.write(' '.join([str(0), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '2':
                #     f.write(' '.join([str(1), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '3':
                #     f.write(' '.join([str(2), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '4':
                #     f.write(' '.join([str(3), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '5':
                #     f.write(' '.join([str(4), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '6':
                #     f.write(' '.join([str(5), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '7':
                #     f.write(' '.join([str(6), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '8':
                #     f.write(' '.join([str(7), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '9':
                #     f.write(' '.join([str(8), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '10':
                #     f.write(' '.join([str(9), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '11':
                #     f.write(' '.join([str(10), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '12':
                #     f.write(' '.join([str(11), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '13':
                #     f.write(' '.join([str(12), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '14':
                #     f.write(' '.join([str(13), str(x_center), str(y_center), str(w), str(h) + '\n']))
                #

                # if sub_label.text == '1':
                #     f.write(' '.join([str(0), str(x_center), str(y_center), str(w), str(h) + '\n']))
                #  if sub_label.text == '2':
                #     f.write(' '.join([str(1), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '3':
                #     f.write(' '.join([str(2), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '4':
                #     f.write(' '.join([str(3), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '5':
                #     f.write(' '.join([str(4), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '6':
                #     f.write(' '.join([str(5), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '7':
                #     f.write(' '.join([str(6), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '8':
                #     f.write(' '.join([str(7), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '9':
                #     f.write(' '.join([str(8), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '10':
                #     f.write(' '.join([str(9), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '11':
                #     f.write(' '.join([str(10), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '12':
                #     f.write(' '.join([str(11), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '13':
                #     f.write(' '.join([str(12), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '14':
                #     f.write(' '.join([str(13), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '15':
                #     f.write(' '.join([str(14), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '16':
                #     f.write(' '.join([str(15), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '17':
                #     f.write(' '.join([str(16), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '18':
                #     f.write(' '.join([str(17), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '19':
                #     f.write(' '.join([str(18), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '20':
                #     f.write(' '.join([str(19), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '21':
                #     f.write(' '.join([str(20), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '22':
                #     f.write(' '.join([str(21), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '23':
                #     f.write(' '.join([str(22), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '24':
                #     f.write(' '.join([str(23), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '25':
                #     f.write(' '.join([str(24), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '26':
                #     f.write(' '.join([str(25), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '27':
                #     f.write(' '.join([str(26), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '28':
                #     f.write(' '.join([str(27), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '29':
                #     f.write(' '.join([str(28), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '30':
                #     f.write(' '.join([str(29), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '31':
                #     f.write(' '.join([str(30), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '32':
                #     f.write(' '.join([str(31), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '33':
                #     f.write(' '.join([str(32), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '34':
                #     f.write(' '.join([str(33), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '35':
                #     f.write(' '.join([str(34), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '36':
                #     f.write(' '.join([str(35), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '37':
                #     f.write(' '.join([str(36), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '38':
                #     f.write(' '.join([str(37), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '39':
                #     f.write(' '.join([str(38), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '40':
                #     f.write(' '.join([str(39), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '41':
                #     f.write(' '.join([str(40), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '42':
                #     f.write(' '.join([str(41), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '43':
                #     f.write(' '.join([str(42), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '44':
                #     f.write(' '.join([str(43), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '45':
                #     f.write(' '.join([str(44), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '46':
                #     f.write(' '.join([str(45), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '47':
                #     f.write(' '.join([str(46), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '48':
                #     f.write(' '.join([str(47), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '49':
                #     f.write(' '.join([str(48), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '50':
                #     f.write(' '.join([str(49), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '51':
                #     f.write(' '.join([str(50), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '52':
                #     f.write(' '.join([str(51), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '53':
                #     f.write(' '.join([str(5), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '54':
                #     f.write(' '.join([str(53), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '55':
                #     f.write(' '.join([str(54), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '56':
                #     f.write(' '.join([str(55), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '57':
                #     f.write(' '.join([str(56), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '58':
                #     f.write(' '.join([str(57), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '59':
                #     f.write(' '.join([str(58), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '60':
                #     f.write(' '.join([str(59), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '61':
                #     f.write(' '.join([str(60), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '62':
                #     f.write(' '.join([str(61), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '63':
                #     f.write(' '.join([str(62), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '64':
                #     f.write(' '.join([str(63), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '65':
                #     f.write(' '.join([str(64), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '66':
                #     f.write(' '.join([str(65), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '67':
                #     f.write(' '.join([str(66), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '68':
                #     f.write(' '.join([str(67), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '69':
                #     f.write(' '.join([str(68), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '70':
                #     f.write(' '.join([str(69), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == '71':
                #     f.write(' '.join([str(70), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == 'Call':
                #     f.write(' '.join([str(0), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == 'Eyes closed':
                #     f.write(' '.join([str(1), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == 'Sleepy':
                #     f.write(' '.join([str(2), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == 'Smoking':
                #     f.write(' '.join([str(3), str(x_center), str(y_center), str(w), str(h) + '\n']))
                # if sub_label.text == 'Yawn':
            except ZeroDivisionError:
                print(filename, '的 width有问题')
'''             
有其他标签选用
                if sub_label.text == 'xxx':
                    f.write(' '.join([str(1), str(x_center), str(y_center), str(w), str(h) + '\n']))
                 if sub_label.text == 'xxx':
                    f.write(' '.join([str(2), str(x_center), str(y_center), str(w), str(h) + '\n']))
'''

# with open(os.path.join(newdir, fp.split('.')[0] + '.txt'), 'a+') as f:
#     f.write(' '.join([str(2), str(x_center), str(y_center), str(w), str(h) + '\n']))
