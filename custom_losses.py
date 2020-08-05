import tensorflow as tf
import numpy as np


"""
    Provides helper functions relating to losses

"""
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
        size = tf.size(obs, out_type=tf.int64)
        ratio = tf.cast( size/count, tf.float32)
        mse = tf.keras.metrics.MSE( obs, preds) * ratio

    return mse

def cond_rain(vals, probs, threshold=0.5):
    """
        If prob of event occuring is above 0.5 return predicted conditional event value,
        If it is below 0.5, then return 0
    """
    round_probs = tf.where(probs<=threshold, 0.0, 1.0)
    vals = vals* round_probs
    return vals

def central_region_bounds(region_grid_params ):
    """Returns the indexes defining the boundaries for the central regions for evaluation

    Args:
        region_grid_params (dict): information on formualation of the patches used in this ds 

    Returns:
        list: defines the vertices of the patch for extraction
    """    

    central_hw_point = np.array(region_grid_params['outer_box_dims'])//2
    
    lower_hw_bound = central_hw_point - np.array(region_grid_params['inner_box_dims']) //2

    upper_hw_bound = lower_hw_bound + np.array(region_grid_params['inner_box_dims'] )
    

    return [lower_hw_bound[0], upper_hw_bound[0], lower_hw_bound[1], upper_hw_bound[1]]

def extract_central_region(tensor, bounds):
    """
        Args:
            tensor ([type]): 4d or 5d tensor
            bounds ([type]): bounds defining the vertices of the patch to be extracted for evaluation
    """
    tensor = tensor[ :, :, bounds[0]:bounds[1],bounds[2]:bounds[3]  ]    
    return tensor

def water_mask( tensor, mask, mask_val=np.nan):
    """Mask out values in tensor by with mask value=0.0
    """
    #mask = tf.broadcast_to( mask, tensor.shape )

    tensor = tf.where(mask, tensor, mask_val)

    return tensor