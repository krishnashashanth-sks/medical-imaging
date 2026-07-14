import tensorflow.keras.backend as K
import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

def hybrid_loss(y_true, y_pred, lambda_dice=0.5, lambda_bce=0.5):
    return lambda_dice * dice_loss(y_true, y_pred) + lambda_bce * bce_loss(y_true, y_pred)

