import tensorflow as tf

def lnormal_mse(obs,preds):

    #loss = tf.keras.losses.MSE(tf.math.log(obs+1), tf.math.log(preds+1) )
    
    factor = 0.75
    loss = tf.keras.losses.MSE(obs**factor, preds**factor)
    
    return loss


def adjusted_log( vals, boundary=0.5):
    "This log does log transformation for all values above 0.5 and linear transformation for values below"
    vals = tf.where( vals>boundary, tf.math.log(vals), boundary + (vals-boundary)/boundary )

    return vals