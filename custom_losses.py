import tensorflow as tf

def lnormal_mse(obs,preds):

    loss = tf.keras.losses.MSE(tf.math.log(obs+1), tf.math.log(preds+1) )
    
    # factor = 0.75
    # loss = tf.keras.losses.MSE(obs**factor, preds**factor)

    #loss = scaler(obs, preds)

    
    return loss

def scaler(y,y_pred):

    scales = tf.where( y_pred > y, 1.5, 0.75 )

    diff = y - y_pred
    diff_2 = tf.math.square(diff)

    diff_scaled = scales * diff_2

    loss = tf.reduce_mean(diff_scaled )
    
    return loss

def adjusted_log( vals, boundary=0.5):

    "This log does log transformation for all values above 0.5 and linear transformation for values below"
    vals = tf.where( vals>boundary, tf.math.log(vals), boundary + (vals-boundary)/boundary )

    return vals

def combine_pp(vals, probs):

    round_probs = tf.math.round( probs)
    vals = vals* round_probs
    return vals