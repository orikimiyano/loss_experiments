from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
import tensorflow as tf

'''
Cross Entropy Loss
'''
def Cross_entropy_loss(y_true,y_pred):
    y_pred = K.clip(y_pred, K.epsilon(),1.0 - K.epsilon())
    crossEntropyLoss = -y_true * tf.log(y_pred)
 
    return tf.reduce_sum(crossEntropyLoss,-1)


'''
Focal Loss
'''

def binary_focal_loss(y_true, y_pred):
    gamma=2
    alpha=0.25
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
    
    p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
    focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
    return K.mean(focal_loss)


'''
Dice Loss
'''
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)



'''
Tversky Loss
'''
def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

'''
Restrictive Item
'''
#FP Item
def item_FP(y_true, y_pred):
	y_true_pos = K.flatten(y_true)
	y_pred_pos = K.flatten(y_pred)
	true_pos = K.sum(y_true_pos * y_pred_pos)
	false_pos = K.sum((1-y_true_pos)*y_pred_pos)
	gamma = 4
	precision = (true_pos + smooth)/(true_pos + false_pos + smooth)
	return K.pow((1-precision), (1/gamma))

#FN Item
def item_FN(y_true, y_pred):
	y_true_pos = K.flatten(y_true)
	y_pred_pos = K.flatten(y_pred)
	true_pos = K.sum(y_true_pos * y_pred_pos)
	false_neg = K.sum(y_true_pos * (1-y_pred_pos))
	gamma = 4
	recall = (true_pos + smooth)/(true_pos + false_neg + smooth)
	return K.pow((1-recall), (1/gamma))

'''
Item Plus with other Losses
'''
#tversky loss with	Restrictive Item
def fin_loss(y_true, y_pred):
	loss1 = tversky_loss(y_true, y_pred)
	loss2 = item_FP(y_true, y_pred)
	loss3 = item_FN(y_true, y_pred)
	rho = 0.8
	sigma = 2
	sigma_t = 0
	return ((rho*loss1)+((1-rho)*(sigma*loss2+(sigma_t*loss3))))
	
#dice loss with	Restrictive Item
def findice_loss(y_true, y_pred):
	loss1 = dice_coef_loss(y_true, y_pred)
	loss2 = item_FP(y_true, y_pred)
	loss3 = item_FN(y_true, y_pred)
	rho = 0.8
	sigma = 1.5
	sigma_t = 0
	return ((rho*loss1)+(1-rho)*(sigma*loss2+(sigma_t*loss3)))

#focal loss with Restrictive Item
def finfocal_loss(y_true, y_pred):
	loss1 = binary_focal_loss(y_true, y_pred)
	loss2 = item_FP(y_true, y_pred)
	loss3 = item_FN(y_true, y_pred)
	rho = 0.8
	sigma = 1.5
	sigma_t = 0
	return ((rho*loss1)+(1-rho)*(sigma*loss2+(sigma_t*loss3)))

#Cross Entropy loss with Restrictive Item
def finCE(y_true, y_pred):
	loss1 = Cross_entropy_loss(y_true, y_pred)
	loss2 = item_FP(y_true, y_pred)
	loss3 = item_FN(y_true, y_pred)
	rho = 0.8
	sigma = 1
	sigma_t = 0
	return ((rho*loss1)+(1-rho)*(sigma*loss2+(sigma_t*loss3)))

'''
combined losses in other studies
'''
def focal_dice(y_true, y_pred):
	t = 1/2
	
	return 1. - (dice_coef(y_true, y_pred)**t)


def focal_tver(y_true, y_pred):
	r = 3/4
	
	return (tversky_loss(y_true, y_pred)**r)
	
	

def dice_plus_focal(y_true, y_pred):
	s = 1/2
	return(dice_coef_loss(y_true, y_pred)+s*binary_focal_loss(y_true, y_pred))
