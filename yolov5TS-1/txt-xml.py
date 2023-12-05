import time
import os
from PIL import Image
import cv2
import numpy as np

'''人为构造xml文件的格式'''
out0 = '''<annotation>
    <folder>%(folder)s</folder>
    <filename>%(name)s</filename>
    <path>%(path)s</path>
    <source>
        <database>None</database>
    </source>
    <size>
        <width>%(width)d</width>
        <height>%(height)d</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
'''
out1 = '''    <object>
        <name>%(class)s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%(xmin)d</xmin>
            <ymin>%(ymin)d</ymin>
            <xmax>%(xmax)d</xmax>
            <ymax>%(ymax)d</ymax>
        </bndbox>
    </object>
'''

out2 = '''</annotation>
'''

'''txt转xml函数'''


def translate(fdir, lists):
    source = {}
    label = {}
    for jpg in lists:
        print(jpg)
        if jpg[-4:] == '.jpg':
            image = cv2.imread(jpg)  # 路径不能有中文
            h, w, _ = image.shape  # 图片大小
            #            cv2.imshow('1',image)
            #            cv2.waitKey(1000)
            #            cv2.destroyAllWindows()

            fxml = jpg.replace('.jpg', '.xml')
            fxml = open(fxml, 'w');
            imgfile = jpg.split('/')[-1]
            source['name'] = imgfile
            source['path'] = jpg
            source['folder'] = os.path.basename(fdir)

            source['width'] = w
            source['height'] = h

            fxml.write(out0 % source)
            txt = jpg.replace('.jpg', '.txt')

            lines = np.loadtxt(txt)  # 读入txt存为数组
            # print(type(lines))

            for box in lines:
               # print(box)
                if box.shape != (5,):
                    box = lines

                '''把txt上的第一列（类别）转成xml上的类别
                   我这里是labelimg标1、2、3，对应txt上面的0、1、2'''
                label['class'] = str(int(box[0]) + 1)  # 类别索引从1开始

                '''把txt上的数字（归一化）转成xml上框的坐标'''
                xmin = float(box[1] - 0.5 * box[3]) * w
                ymin = float(box[2] - 0.5 * box[4]) * h
                xmax = float(xmin + box[3] * w)
                ymax = float(ymin + box[4] * h)

                label['xmin'] = xmin
                label['ymin'] = ymin
                label['xmax'] = xmax
                label['ymax'] = ymax

                # if label['xmin']>=w or label['ymin']>=h or label['xmax']>=w or label['ymax']>=h:
                #     continue
                # if label['xmin']<0 or label['ymin']<0 or label['xmax']<0 or label['ymax']<0:
                #     continue

            fxml.write(out1 % label)
            fxml.write(out2)


if __name__ == '__main__':
    file_dir = 'D:\DYJ\99\labels/test-xml'  #
    lists = []
    for i in os.listdir(file_dir):
        if i[-3:] == 'jpg':
            lists.append(file_dir + '/' + i)
            print(lists)
    translate(file_dir, lists)
    print('---------------Done!!!--------------')



# import time
# import os
# from PIL import Image
# import cv2
# import numpy as np
#
# '''人为构造xml文件的格式'''
# out0 = '''<annotation>
#     <folder>%(folder)s</folder>
#     <filename>%(name)s</filename>
#     <path>%(path)s</path>
#     <source>
#         <database>None</database>
#     </source>
#     <size>
#         <width>%(width)d</width>
#         <height>%(height)d</height>
#         <depth>3</depth>
#     </size>
#     <segmented>0</segmented>
# '''
# out1 = '''    <object>
#         <name>%(class)s</name>
#         <pose>Unspecified</pose>
#         <truncated>0</truncated>
#         <difficult>0</difficult>
#         <bndbox>
#             <xmin>%(xmin)d</xmin>
#             <ymin>%(ymin)d</ymin>
#             <xmax>%(xmax)d</xmax>
#             <ymax>%(ymax)d</ymax>
#         </bndbox>
#     </object>
# '''
#
# out2 = '''</annotation>
# '''
#
# '''txt转xml函数'''
#
#
# def translate(fdir, lists):
#     source = {}
#     label = {}
#     for jpg in lists:
#         print(jpg)
#         if jpg[-4:] == '.jpg':
#             image = cv2.imread(jpg)  # 路径不能有中文
#             h, w, _ = image.shape  # 图片大小
#             #            cv2.imshow('1',image)
#             #            cv2.waitKey(1000)
#             #            cv2.destroyAllWindows()
#
#             fxml = jpg.replace('.jpg', '.xml')
#             fxml = open(fxml, 'w');
#             imgfile = jpg.split('/')[-1]
#             source['name'] = imgfile
#             source['path'] = jpg
#             source['folder'] = os.path.basename(fdir)
#
#             source['width'] = w
#             source['height'] = h
#
#             fxml.write(out0 % source)
#             txt = jpg.replace('.jpg', '.txt')
#
#             lines = np.loadtxt(txt)  # 读入txt存为数组
#             # print(type(lines))
#
#             for box in lines:
#                 # print(box.shape)
#                 if box.shape != (5,):
#                     box = lines
#
#                 '''把txt上的第一列（类别）转成xml上的类别
#                    我这里是labelimg标1、2、3，对应txt上面的0、1、2'''
#                 label['class'] = str(int(box[0]) + 1)  # 类别索引从1开始
#
#                 '''把txt上的数字（归一化）转成xml上框的坐标'''
#                 xmin = float(box[1] - 0.5 * box[3]) * w
#                 ymin = float(box[2] - 0.5 * box[4]) * h
#                 xmax = float(xmin + box[3] * w)
#                 ymax = float(ymin + box[4] * h)
#
#                 label['xmin'] = xmin
#                 label['ymin'] = ymin
#                 label['xmax'] = xmax
#                 label['ymax'] = ymax
#
#                 # if label['xmin']>=w or label['ymin']>=h or label['xmax']>=w or label['ymax']>=h:
#                 #     continue
#                 # if label['xmin']<0 or label['ymin']<0 or label['xmax']<0 or label['ymax']<0:
#                 #     continue
#
#                 fxml.write(out1 % label)
#             fxml.write(out2)
#
#
# if __name__ == '__main__':
#     file_dir = 'D:/AAweixian/images/Dangerous Driving/labels/Test-xml'  #图片和txt放在一起
#    # print('111111111111')
#     lists = []
#     for i in os.listdir(file_dir):
#         if i[-3:] == 'jpg':
#             lists.append(file_dir + '/' + i)
#           #  print(lists)
#             print('111111111111')
#
#     translate(file_dir, lists)
#
#     print('---------------Done!!!--------------')
# import os
# import xml.etree.ElementTree as ET
# from xml.etree.ElementTree import Element, SubElement
# from PIL import Image
#
#
# class Xml_make(object):
#     def __init__(self):
#         super().__init__()
#
#     def __indent(self, elem, level=0):
#         i = "\n" + level * "\t"
#         if len(elem):
#             if not elem.text or not elem.text.strip():
#                 elem.text = i + "\t"
#             if not elem.tail or not elem.tail.strip():
#                 elem.tail = i
#             for elem in elem:
#                 self.__indent(elem, level + 1)
#             if not elem.tail or not elem.tail.strip():
#                 elem.tail = i
#         else:
#             if level and (not elem.tail or not elem.tail.strip()):
#                 elem.tail = i
#
#     def _imageinfo(self, list_top):
#         annotation_root = ET.Element('annotation')
#         # annotation_root.set('verified', 'no')
#         tree = ET.ElementTree(annotation_root)
#         # '''
#         # 0:xml_savepath 1:folder,2:filename,3:path
#         # 4:checked,5:width,6:height,7:depth
#         # '''
#         folder_element = ET.Element('folder')
#         folder_element.text = list_top[1]
#         annotation_root.append(folder_element)
#
#         filename_element = ET.Element('filename')
#         filename_element.text = list_top[2]
#         annotation_root.append(filename_element)
#
#         source_element = ET.Element('source')
#         database_element = SubElement(source_element, 'database')
#         database_element.text = 'KAIST'
#         annotation_root.append(source_element)
#
#         size_element = ET.Element('size')
#         width_element = SubElement(size_element, 'width')
#         width_element.text = str(list_top[3])
#         height_element = SubElement(size_element, 'height')
#         height_element.text = str(list_top[4])
#         depth_element = SubElement(size_element, 'depth')
#         depth_element.text = str(list_top[5])
#         annotation_root.append(size_element)
#
#         segmented_person_element = ET.Element('segmented')
#         segmented_person_element.text = '0'
#         annotation_root.append(segmented_person_element)
#
#         return tree, annotation_root
#
#     def _bndbox(self, annotation_root, list_bndbox):
#         for i in range(0, len(list_bndbox), 9):
#             object_element = ET.Element('object')
#             name_element = SubElement(object_element, 'name')
#             name_element.text = list_bndbox[i]
#
#             bndbox_element = SubElement(object_element, 'bndbox')
#             xmin_element = SubElement(bndbox_element, 'xmin')
#             xmin_element.text = str(list_bndbox[i + 1])
#
#             ymin_element = SubElement(bndbox_element, 'ymin')
#             ymin_element.text = str(list_bndbox[i + 2])
#
#             xmax_element = SubElement(bndbox_element, 'xmax')
#             xmax_element.text = str(list_bndbox[i + 3])
#
#             ymax_element = SubElement(bndbox_element, 'ymax')
#             ymax_element.text = str(list_bndbox[i + 4])
#
#             pose_element = SubElement(object_element, 'pose')
#             pose_element.text = list_bndbox[i + 5]
#
#             truncated_element = SubElement(object_element, 'truncated')
#             truncated_element.text = list_bndbox[i + 6]
#
#             difficult_element = SubElement(object_element, 'difficult')
#             difficult_element.text = list_bndbox[i + 7]
#
#             flag_element = SubElement(object_element, 'occlusion')
#             flag_element.text = list_bndbox[i + 8]
#
#             annotation_root.append(object_element)
#
#         return annotation_root
#
#     def txt_to_xml(self, list_top, list_bndbox):
#         tree, annotation_root = self._imageinfo(list_top)
#         annotation_root = self._bndbox(annotation_root, list_bndbox)
#         self.__indent(annotation_root)
#         tree.write(list_top[0], encoding='utf-8', xml_declaration=True)
#
#
# def txt_2_xml(source_path, xml_save_dir, txt_dir):
#     COUNT = 0
#     for folder_path_tuple, folder_name_list, file_name_list in os.walk(source_path):
#         for file_name in file_name_list:
#             file_suffix = os.path.splitext(file_name)[-1]
#             if file_suffix != '.jpg':
#                 continue
#             list_top = []
#             list_bndbox = []
#             path = os.path.join(folder_path_tuple, file_name)
#             xml_save_path = os.path.join(xml_save_dir, file_name.replace(file_suffix, '.xml'))
#             txt_path = os.path.join(txt_dir, file_name.replace(file_suffix, '.txt'))
#             filename = os.path.splitext(file_name)[0]
#             im = Image.open(path)
#             im_w = im.size[0]
#             im_h = im.size[1]
#             width = str(im_w)
#             height = str(im_h)
#             depth = '3'
#             occlusion = '0'
#             pose = 'unknown'
#             truncated = '0'
#             difficult = '0'
#             list_top.extend([xml_save_path, folder_path_tuple, filename,
#                              width, height, depth])
#             for line in open(txt_path, 'r'):
#                 line = line.strip()
#                 if line == "% bbGt version=3":
#                     continue
#                 info = line.split(' ')
#                 name = info[0]
#                 xmin = float(info[1])
#                 ymin = float(info[2])
#                 xmax = float(info[3])
#                 ymax = float(info[4])
#                 # xmax = xmin + w
#                 # ymax = ymin + h
#                 # x_cen = float(info[1]) * im_w
#                 # y_cen = float(info[2]) * im_h
#                 # w = float(info[3]) * im_w
#                 # h = float(info[4]) * im_h
#                 # xmin = int(x_cen - w / 2)
#                 # ymin = int(y_cen - h / 2)
#                 # xmax = int(x_cen + w / 2)
#                 # ymax = int(y_cen + h / 2)
#                 list_bndbox.extend([name, str(xmin), str(ymin), str(xmax), str(ymax), pose, truncated, difficult,
#                                     occlusion])
#             Xml_make().txt_to_xml(list_top, list_bndbox)
#             COUNT += 1
#             print(COUNT, xml_save_path)
#
#
# if __name__ == '__main__':
#     source_path = r'D:\AAweixian\images\Dangerous Driving\images\Test第一次调整\BIFPN增长的测试集\images\Test'  # txt标注文件所对应的的图片
#     xml_save_dir = r'D:/AAweixian/images/Dangerous Driving/labels/Test-xml'  # 转换为xml标注文件的保存路径
#     txt_dir = r'D:\AAweixian\images\Dangerous Driving\images\Test第一次调整\BIFPN增长的测试集\lables\Test'  # 需要转换的txt标注文件
#     txt_2_xml(source_path, xml_save_dir, txt_dir)
#
