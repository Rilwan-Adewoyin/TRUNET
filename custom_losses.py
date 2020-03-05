import tensorflow as tf

def lnormal_mse(obs,preds):
    loss =  tf.keras.losses.MSE( tf.math.log(obs) , tf.math.log(preds) )
    return loss
