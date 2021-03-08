from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from keras import backend as K
import tensorflow as tf

label_one = [128,128,128]
label_two = [128,0,0]
label_three = [192,192,128]
label_four = [128,64,128]

COLOR_DICT = np.array([label_one, label_two, label_three, label_four])

def adjustData(img,label,flag_multi_class,num_class):
    if (flag_multi_class):
        img = img/255.
        label = label[:,:,:,0] if (len(label.shape)==4) else label[:,:,0]
        new_label = np.zeros(label.shape+(num_class,))
        for i in range(num_class):
            new_label[label==i,i] = 1
        label = new_label
    elif (np.max(img)>1):
        img = img/255.
        label = label/255.
        label[label>0.5] = 1
        label[label<=0.5] = 0
    return (img,label)

def trainGenerator(batch_size,aug_dict,train_path,image_folder,label_folder,image_color_mode='rgb',
                   label_color_mode='rgb',image_save_prefix='image',label_save_prefix='label',
                   flag_multi_class=True,num_class=4,save_to_dir=None,target_size=(256,256),seed=1):
    image_datagen = ImageDataGenerator(**aug_dict)
    label_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix = image_save_prefix,
        seed = seed
    )
    label_generator = label_datagen.flow_from_directory(
        train_path,
        classes = [label_folder],
        class_mode = None,
        color_mode = label_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix = label_save_prefix,
        seed = seed
    )
    train_generator = zip(image_generator,label_generator)
    for img,label in train_generator:
        img,label = adjustData(img,label,flag_multi_class,num_class)
        yield img,label

def testGenerator(test_path,target_size=(256,256),flag_multi_class=True,as_gray=False):
    filenames = os.listdir(test_path)
    for filename in filenames:
        img = io.imread(os.path.join(test_path,filename),as_gray=as_gray)
        img = trans.resize(img,target_size,mode = 'constant')
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr

def saveResult(save_path,npyfile,flag_multi_class=True):
    for i,item in enumerate(npyfile):
        if flag_multi_class:
            img = item
            img_out = np.zeros(img[:, :, 0].shape + (3,))
            for row in range(img.shape[0]):
                for col in range(img.shape[1]):
                    index_of_class = np.argmax(img[row, col])
                    img_out[row, col] = COLOR_DICT[index_of_class]
            img = img_out.astype(np.uint8)
            io.imsave(os.path.join(save_path, '%s_predict.png' % i), img)
        else:
            img = item[:, :, 0]
            img[img > 0.5] = 1
            img[img <= 0.5] = 0
            img = img * 255.
            io.imsave(os.path.join(save_path, '%s_predict.png' % i), img)
