import tensorflow as tf
import numpy as np


"""
    Provides helper functions relating to losses

"""
def log_mse( obs, preds, count):
    """Calculates Log MSE

    Args:
        obs (np.arr/tensor): True values
        preds (np.arr/tensor): Predicted values
        count (Int, optional): Custom sample size for MSE calc. Defaults to None.

    Returns:
        loss (float32):
    """
    loss = mse( tf.math.log(obs+1), tf.math.log(preds+1), count )
        
    return loss

def mse( obs, preds, count=None):
    """Calculated MSE, but with a custom sample size (divisor)

    Args:
        obs (np.arr/tensor): True values
        preds (np.arr/tensor): Predicted values
        count (Int, optional): Custom sample size for MSE calc. Defaults to None.

    Returns:
        mse (float32): mean squared error value 
    """
    if count == None:
        mse = tf.keras.metrics.MSE( obs, preds)
    else:
        mse = tf.math.squared_difference( obs, preds)
        mse = mse / count
        mse = tf.math.reduce_sum( mse )
    return mse

def cond_rain(vals, probs):
    """
        If prob of event occuring is above 0.5 return predicted conditional event value,
        If it is below 0.5, then return 0
    """
    round_probs = tf.math.round( probs)
    vals = vals* round_probs
    return vals

# Stochastic loss, use kl loss in later training steps
def kl_loss_weighting_scheme( max_batch, curr_batch, var_model_type="dropout" ):

    if var_model_type in ["flipout"]:
        idx = max_batch-curr_batch+1
        weight = 1/(2**idx)
    else:
        weight = (1/max_batch)
        
    return weight*(1/10)

#TODO(akann-ade): let trainTruNet and testTruNet extend a common class, which contains this function
def central_region_bounds(region_grid_params ):
    """Returns the indexes defining the boundaries for the central regions for evaluation

    Args:
        region ([type]): [description]
        region_grid_params ([type]): [description]

    Returns:
        [type]: [description]
    """    

    central_hw_point = np.array(region_grid_params['outer_box_dims'])//2
    
    lower_hw_bound = central_hw_point - np.array(region_grid_params['inner_box_dims']) //2

    upper_hw_bound = lower_hw_bound + np.array(region_grid_params['inner_box_dims'] )
    

    return [lower_hw_bound[0], upper_hw_bound[0], lower_hw_bound[1], upper_hw_bound[1]]

def extract_central_region(tensor, bounds):
    """

    Args:
        tensor ([type]): [description]
        bounds ([type]): [description]
    """

    tensor = tensor[ :, :, bounds[0]:bounds[1],bounds[2]:bounds[3]  ]    
    
    return tensor

def water_mask( tensor, mask):

    mask = tf.broadcast_to( mask, tensor.shape )

    tensor = tf.where(mask, tensor, 0.0)

    return tensor