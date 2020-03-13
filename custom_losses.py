import tensorflow as tf

def lnormal_mse(obs,preds):
    # preds_filtr = tf.where( preds>0, True, False)
    # preds = tf.boolean_mask(preds,preds_filtr )
    # obs = tf.boolean_mask(obs, preds_filtr)
    loss =  tf.keras.losses.MSE( adjusted_log(obs) , adjusted_log(preds) )
    return loss

@tf.function
def adjusted_log( vals, boundary=0.5):
    "This log does log transformation for all values above 0.5 and linear transformation for values below"
    vals = tf.where( vals>boundary, tf.math.log(vals), boundary + (vals-boundary)/boundary )

    return vals