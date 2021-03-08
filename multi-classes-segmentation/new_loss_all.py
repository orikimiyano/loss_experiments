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
losses for two-class segmentation 
'''


#Cross_entropy_loss

def Cross_entropy_loss(y_true,y_pred):
    y_pred = K.clip(y_pred, K.epsilon(),1.0 - K.epsilon())
    crossEntropyLoss = -y_true * tf.log(y_pred)
 
    return tf.reduce_sum(crossEntropyLoss,-1)

#dice_loss
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

#focal_loss

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		#return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
		#return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
		return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed

smooth = 1.

#tversky_loss

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


#FP item
def item_FP(y_true, y_pred):
	y_true_pos = K.flatten(y_true)
	y_pred_pos = K.flatten(y_pred)
	true_pos = K.sum(y_true_pos * y_pred_pos)
	false_pos = K.sum((1-y_true_pos)*y_pred_pos)
	gamma = 4
	precision = (true_pos + smooth)/(true_pos + false_pos + smooth)
	return K.pow((1-precision), (1/gamma))

#FN item
def item_FN(y_true, y_pred):
	y_true_pos = K.flatten(y_true)
	y_pred_pos = K.flatten(y_pred)
	true_pos = K.sum(y_true_pos * y_pred_pos)
	false_neg = K.sum(y_true_pos * (1-y_pred_pos))
	gamma = 1
	recall = (true_pos + smooth)/(true_pos + false_neg + smooth)
	return K.pow((1-recall), (1/gamma))

#Optimal combination
def fin_loss(y_true, y_pred):
	loss1 = tversky_loss(y_true, y_pred)
	loss2 = item_FP(y_true, y_pred)
	loss3 = item_FN(y_true, y_pred)
	rho = 0.8
	sigma_1 = 2
	sigma_2 = 0
    
	return ((rho*loss1)+(1-rho)*((sigma_1*loss2)+(sigma_2*loss3)))


#Control group

##focal dice
def focal_dice(y_true, y_pred):
	t = 1/2
	
	return 1. - (dice_coef(y_true, y_pred)**t)

##focal tversky
def focal_tver(y_true, y_pred):
	r = 3/4
	
	return (tversky_loss(y_true, y_pred)**r)
	
##dice+focal
def focalloss(y_true, y_pred):
	gamma=2.
	alpha=.25
	pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
	pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))	
	return -0.5*K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
def dice_plus_focal(y_true, y_pred):
	s = 1/2
	return(dice_coef_loss(y_true, y_pred)+focalloss(y_true, y_pred))

##

'''
losses for muli-class segmentation
'''

#GDL
def generalized_dice(y_true, y_pred):
    
    y_true    = K.reshape(y_true,shape=(-1,4))
    y_pred    = K.reshape(y_pred,shape=(-1,4))
    sum_p     = K.sum(y_pred, -2)
    sum_r     = K.sum(y_true, -2)
    sum_pr    = K.sum(y_true * y_pred, -2)
    weights   = K.pow(K.square(sum_r) + K.epsilon(), -1)
    generalized_dice = (2 * K.sum(weights * sum_pr)) / (K.sum(weights * (sum_r + sum_p)))
    
    return generalized_dice
 
def generalized_dice_loss(y_true, y_pred):   
    return 1-generalized_dice(y_true, y_pred)



#FP item for each loss
def dice_loss_multi(y_true, y_pred):
	y_true_n = K.reshape(y_true,shape=(-1,4))
	y_pred_n = K.reshape(y_pred,shape=(-1,4))
	total_loss = 0
	wl = 0
    
	for i in range(y_pred_n.shape[1]):
		single_loss = dice_coef_loss(y_true_n[:, i], y_pred_n[:, i])
		num_all_int = tf.size(y_true_n[:, i])
		num_pos_int = tf.count_nonzero(y_true_n[:, i])

		num_pos = tf.to_float(num_pos_int)
		num_all = tf.to_float(num_all_int)
		wl = ((num_all - num_pos + smooth)/num_all + smooth)**3
		single_loss = wl*single_loss
		total_loss += single_loss
	
	return total_loss

def focal_loss_multi(y_true, y_pred):
	y_true_n = K.reshape(y_true,shape=(-1,4))
	y_pred_n = K.reshape(y_pred,shape=(-1,4))
	total_loss = 0
	wl = 0
    
	for i in range(y_pred_n.shape[1]):
		single_loss = focalloss(y_true_n[:, i], y_pred_n[:, i])
		num_all_int = tf.size(y_true_n[:, i])
		num_pos_int = tf.count_nonzero(y_true_n[:, i])

		num_pos = tf.to_float(num_pos_int)
		num_all = tf.to_float(num_all_int)
		wl = ((num_all - num_pos + smooth)/num_all + smooth)**3
		single_loss = wl*single_loss
		total_loss += single_loss
	
	return total_loss

def tversky_loss_multi(y_true, y_pred):
	y_true_n = K.reshape(y_true,shape=(-1,4))
	y_pred_n = K.reshape(y_pred,shape=(-1,4))
	total_loss = 0
	wl = 0
    
	for i in range(y_pred_n.shape[1]):
		single_loss = tversky_loss(y_true_n[:, i], y_pred_n[:, i])
		num_all_int = tf.size(y_true_n[:, i])
		num_pos_int = tf.count_nonzero(y_true_n[:, i])

		num_pos = tf.to_float(num_pos_int)
		num_all = tf.to_float(num_all_int)
		wl = ((num_all - num_pos + smooth)/num_all + smooth)**3
		single_loss = wl*single_loss
		total_loss += single_loss
	
	return total_loss

def fin_loss_multi(y_true, y_pred):
	y_true_n = K.reshape(y_true,shape=(-1,4))
	y_pred_n = K.reshape(y_pred,shape=(-1,4))
	total_loss = 0
	wl = 0
    
	for i in range(y_pred_n.shape[1]):
		single_loss = fin_loss(y_true_n[:, i], y_pred_n[:, i])
		num_all_int = tf.size(y_true_n[:, i])
		num_pos_int = tf.count_nonzero(y_true_n[:, i])

		num_pos = tf.to_float(num_pos_int)
		num_all = tf.to_float(num_all_int)
		wl = ((num_all - num_pos + smooth)/num_all + smooth)**3
		single_loss = wl*single_loss
		total_loss += single_loss
	
	return total_loss

#Sum Value for each loss
def tversky_add(y_true, y_pred):

	y_true_n = K.reshape(y_true,shape=(-1,4))
	y_pred_n = K.reshape(y_pred,shape=(-1,4))
	total_loss = 0
	wl = 0
    

	for i in range(y_pred_n.shape[1]):
		single_loss = tversky_loss(y_true_n[:, i], y_pred_n[:, i])
		single_loss = single_loss
		total_loss += single_loss
	
	return total_loss

def focal_add(y_true, y_pred):

	y_true_n = K.reshape(y_true,shape=(-1,4))
	y_pred_n = K.reshape(y_pred,shape=(-1,4))
	total_loss = 0
	wl = 0
    

	for i in range(y_pred_n.shape[1]):
		single_loss = focalloss(y_true_n[:, i], y_pred_n[:, i])
		single_loss = single_loss
		total_loss += single_loss
	
	return total_loss

def dice_add(y_true, y_pred):

	y_true_n = K.reshape(y_true,shape=(-1,4))
	y_pred_n = K.reshape(y_pred,shape=(-1,4))
	total_loss = 0
	wl = 0
    

	for i in range(y_pred_n.shape[1]):
		single_loss = dice_coef_loss(y_true_n[:, i], y_pred_n[:, i])
		single_loss = single_loss
		total_loss += single_loss
	
	return total_loss

#Mean Value for each loss	
def tversky_mean(y_true, y_pred):

	y_true_n = K.reshape(y_true,shape=(-1,4))
	y_pred_n = K.reshape(y_pred,shape=(-1,4))
	total_loss = 0
	wl = 0

	for i in range(y_pred_n.shape[1]):
		single_loss = tversky_loss(y_true_n[:, i], y_pred_n[:, i])
		single_loss = single_loss
		total_loss += single_loss
	
	nc = tf.to_float(y_pred.shape[-1])
	total_loss = total_loss/nc
	
	return total_loss
	
def focal_mean(y_true, y_pred):

	y_true_n = K.reshape(y_true,shape=(-1,4))
	y_pred_n = K.reshape(y_pred,shape=(-1,4))
	total_loss = 0
	wl = 0

	for i in range(y_pred_n.shape[1]):
		single_loss = focalloss(y_true_n[:, i], y_pred_n[:, i])
		single_loss = single_loss
		total_loss += single_loss
	
	nc = tf.to_float(y_pred.shape[-1])
	total_loss = total_loss/nc
	
	return total_loss
