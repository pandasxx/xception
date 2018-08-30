import os
import random
import tensorflow as tf 
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import h5py

def read_pic_path(pic_src):
    # 得出子目录
    contents = os.listdir(pic_src)
    # 有子文件的目录为有效的
    classes = [each for each in contents if os.path.isdir(pic_src + each)]
    return classes

def get_all_files_path(pic_src):
    all_images = []
    all_labels = []
    
    classes = read_pic_path(pic_src)
    for index, class_name in enumerate(classes):
        class_path = pic_src + class_name
        for img_name in os.listdir(pic_src + class_name):
            img_path = class_path + '/' + img_name
            all_images.append(img_path)
            all_labels.append(index)

    return all_images, all_labels, classes

def shuffle_data_list(data_list, label_list):
    index = [i for i in range(len(data_list))]
    random.shuffle(index)
    data_list = [data_list[each] for each in index]
    label_list = [label_list[each] for each in index]
    return data_list, label_list

# 使用padding方式填充原图为正方形，并resize到目标大小
def resize_img(img, size):
    longer_side = max(img.size)
    horizontal_padding = (longer_side - img.size[0]) / 2
    vertical_padding = (longer_side - img.size[1]) / 2


#    img_croped = img.crop(
#        (
#            -horizontal_padding,
#            -vertical_padding,
#            img.size[0] + horizontal_padding,
#            img.size[1] + vertical_padding
#        )
#    )
#    img_resized = img_croped.resize((size, size))
#    return img_resized

    img_croped = img.crop(
        (
            -((299 - img.size[0]) / 2),
            -((299 - img.size[1]) / 2),
            img.size[0] + (299 - img.size[0]) / 2,
            img.size[1] + (299 - img.size[1]) / 2
        )
    )
    img_resized = img_croped.resize((size, size))
    return img_resized

# 读取图片到指定的tfrecord文件中
def read_img_2_folder(imgs_list, label_list, img_size, dst_folder):
        for index, img_path in enumerate(imgs_list):
                img = Image.open(img_path)
                # 有些图片是png（RGBA），会导致后面出错，在这里统一转换成RGB
                if (img.mode != 'RGB'):
                        img =img.convert("RGB")
                img_resized = resize_img(img, img_size)

                path_elements = img_path.split('/')
                new_sub_path = path_elements[-2] + '/' + path_elements[-1]
                new_sub_folder = path_elements[-2]
                new_path = dst_folder + new_sub_path
                new_folder = dst_folder + new_sub_folder

                if (False == os.path.exists(new_folder)):
                    os.makedirs(new_folder)
                img_resized.save(new_path, 'png')
#                img_resized.show()
                img.close()
                # img_resized 是 PIL对象，但可以这样直接赋值给numpy数组，且类型为float

def data_preprocess(pic_src, train_ratio, img_size, data_dst):
    # 读取所有路径，并乱序排列
    all_images, all_labels, classes = get_all_files_path(pic_src)
    all_images, all_labels = shuffle_data_list(all_images, all_labels)

    #获取切割num
    data_num = len(all_labels)
    train_num = int(data_num * train_ratio)
    
    print(data_num)
    print(train_num)
    
    #切割label
    train_labels = all_labels[:train_num]                                    # 切分出 从 0 到第train_num-1个（或者说前train_num个）形成新数组
    valid_labels  = all_labels[train_num:]                  		# 先切分出从第train_num个到最后一个

    print(data_num)
    print(train_num)
    print(len(valid_labels))
    
    #切割image
    train_images = all_images[:train_num]
    valid_images  = all_images[train_num:]

    #存储到h5py中
    read_img_2_folder(train_images, train_labels, img_size, data_dst[0])
    read_img_2_folder(valid_images, valid_labels, img_size, data_dst[1])

def submit_data_preprocess(pic_src, pic_size, data_dst):
    all_images = []
    for img_name in os.listdir(pic_src):
            img_path =  pic_src + img_name
            all_images.append(img_path)

    writer = tf.python_io.TFRecordWriter(data_dst)
    for index, img_path in enumerate(all_images):
        img = Image.open(img_path)
        # 有些图片是png（RGBA），会导致后面出错，在这里统一转换成RGB
        if (img.mode != 'RGB'):
            img =img.convert("RGB")
        img_resized = resize_img(img, pic_size)
#       img_resized.show()
        img.close()
        img_raw = img_resized.tobytes()
        #you de wenjian you 4channels
        example = tf.train.Example(features=tf.train.Features(feature={
            'img_raw': tf.train.Feature(bytes_list = tf.train.BytesList(value = [img_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()

    return len(all_images)

def test_auto_reload():
    print("pass")
    print("pass1")
    print("pass2")