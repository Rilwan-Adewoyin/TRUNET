#region imports
import os
import sys

import data_generators
import utility
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float16')
tf.keras.backend.set_epsilon(1e-3)
try:
    gpu_devices = tf.config.list_physical_devices('GPU')
except Exception as e:
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')

from tensorflow.keras.mixed_precision import experimental as mixed_precision
tf.config.set_soft_device_placement(True)

print(gpu_devices)
for idx, gpu_name in enumerate(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_name, True)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

def is_compatible_with(self, other):
    """Returns True if the `other` DType will be converted to this DType.
    The conversion rules are as follows:
    ```python
    DType(T)       .is_compatible_with(DType(T))        == True
    ```
    Args:
        other: A `DType` (or object that may be converted to a `DType`).
    Returns:
        True if a Tensor of the `other` `DType` will be implicitly converted to
        this `DType`.
    """
    other = tf.dtypes.as_dtype(other)
    if self._type_enum==19 and other.as_datatype_enum==1:
        return True

    return self._type_enum in (other.as_datatype_enum,
                                other.base_dtype.as_datatype_enum)
#from tensorflow.python.framework.dtypes import DType
tf.DType.is_compatible_with = is_compatible_with

import tensorflow_probability as tfp
try:
    import tensorflow_addons as tfa
except Exception as e:
    tfa = None

from tensorflow_probability import layers as tfpl
from tensorflow_probability import distributions as tfd
from tensorboard.plugins.hparams import api as hp
import pandas as pd

import math


import argparse 
from tqdm import tqdm
import traceback
import time

import models
import hparameters
import util_predict
import utility
import json
import ast
import itertools
from data_generators import Generator_rain
# endregion

def main( test_params, model_params):
    print("GPU Available: ", tf.test.is_gpu_available() )

    model, checkpoint_code = util_predict.load_model(test_params, model_params)

    if model_params['model_type_settings']['discrete_continuous'] == False:
         
        predict(model, test_params, model_params ,checkpoint_code )
    else:
        if model_params['model_type_settings']['prob_rain_thresh'] == "Tune":
            li_precip_range = np.arange(0.20, 0.55, 0.05 ).tolist()
        
            for precip_thrsh in li_precip_range: 
                predict(model, test_params, model_params, checkpoint_code, precip_thrsh )
        else:
            predict(model, test_params, model_params, checkpoint_code, model_params['model_type_settings']['prob_rain_thresh'] )


def predict( model, test_params, model_params ,checkpoint_no, precip_thrsh=0 ):
    
    # region ------ Setting up testing variables
    if model_params['model_type_settings']['location'] == 'region_grid' and 'location_test' not in model_params['model_type_settings'].keys():
        test_set_size_elements = int(test_params['test_set_size_elements']* np.prod(model_params['region_grid_params']['slides_v_h']) )
    else:
        test_set_size_elements = test_params['test_set_size_elements']
    
    test_set_size_batches = test_set_size_elements //( test_params['batch_size'] * model_params['data_pipeline_params']['lookback_target'] )

    if(model_params['model_name'] in ["SimpleGRU","THST", "SimpleConvGRU","SimpleDense"]):
        upload_size = test_set_size_batches #buffer size
    elif(model_params['model_name'] in ["DeepSD"]):
        upload_size = max( int( test_set_size_batches* test_params['dataset_pred_batch_reporting_freq']), 1 )

    li_predictions = [] #This will be a list of list of tensors, each list contain a set of (maybe stochastic) predictions for the corresponding ts
    li_timestamps = test_params['epochs']

    if model_params['model_name']=="DeepSD" :
        li_timestamps_chunked = [li_timestamps[i:i+test_params['batch_size']] for i in range(0, len(li_timestamps), test_params['batch_size'])]

    elif model_params['model_name'] in ["THST","SimpleConvGRU"]:
        if model_params['model_type_settings']['location'] == 'region_grid':
            li_timestamps_chunked = [li_timestamps[i:i+test_params['window_shift']] for i in range(0, len(li_timestamps), test_params['window_shift'])] 
            li_timestamps_chunked = list( itertools.chain.from_iterable( itertools.repeat(li_timestamps_chunked, x ) for x in range(int(np.prod(model_params['region_grid_params']['slides_v_h']))) ) )
        else:
            li_timestamps_chunked = [li_timestamps[i:i+test_params['window_shift']*test_params['batch_size']] for i in range(0, len(li_timestamps), test_params['window_shift']*test_params['batch_size'])] 
    
    if model_params['model_name'] in ["SimpleGRU","SimpleDense"]:
        li_timestamps_chunked = [li_timestamps[i:i+test_params['window_shift']*test_params['batch_size'] ] for i in range(0, len(li_timestamps), test_params['window_shift']*test_params['batch_size'])]

    li_true_values = []
    # endregion

    # region ----- Setting up datasets
    if(model_params['model_name']=="DeepSD"):
        ds = data_generators.load_data_vandal( test_params['starting_test_element'], 
                test_params, model_params,_num_parallel_calls=test_params['num_parallel_calls'], data_dir = test_params['data_dir'], drop_remainder=True  )
        ds = ds.take( test_set_size_batches )

    elif(model_params['model_name'] in [ "THST", "SimpleConvGRU"] ):
        if model_params['model_type_settings']['location'] =="whole_region" :

            ds, idx_city_in_whole = data_generators.load_data_ati(test_params, model_params, None, day_to_start_at=test_params['test_start_date'], data_dir=test_params['data_dir'] )
            idx_city_in_region = idx_city_in_whole
            ds = ds.take(test_set_size_batches)

        elif  model_params['model_type_settings']['location'] !="whole_region" and 'location_test' in model_params['model_type_settings'].keys() :

            ds, idx_city_in_region = data_generators.load_data_ati(test_params, model_params, None, day_to_start_at=test_params['test_start_date'], data_dir=test_params['data_dir'], _num_parallel_calls=1 )
            ds = ds.take( test_set_size_batches )
            cache_suffix = '_{}_bs_{}_loctest_{}_{}'.format( model_params['model_name'], test_params['batch_size'],model_params['model_type_settings']['location_test'],model_params['model_type_settings']['location']  ).strip('[]') 
            
            ds = ds.cache('data_cache/ds_test_cache'+cache_suffix ).repeat(1) 

    
    elif(model_params['model_name'] in ["SimpleGRU","SimpleDense"]):
        ds = data_generators.load_data_ati(test_params, model_params, None, day_to_start_at=test_params['test_start_date'], data_dir=test_params['data_dir'] )
        ds = ds.take( test_set_size_batches )

    iter_test = enumerate(ds)
    #endregion

    # region --- predictions
    
    for batch in range(1, int(1+test_set_size_batches) ):

        if model_params['model_type_settings']['location'] == 'region_grid' or model_params['model_type_settings']['twoD']==True :
            idx, (feature, target, mask) = next(iter_test)
        else:
            idx, (feature, target) = next(iter_test)
            pred = model.predict( feature, model_params['model_type_settings']['stochastic_f_pass'] ) # shape (bs ,156,352 )  

        if model_params['model_name'] == "DeepSD":
            pred = utility.water_mask( tf.squeeze(pred), test_params['bool_water_mask'])

            li_predictions.append(utility.standardize(pred,reverse=True,distr_type=model_params['model_type_settings']['distr_type']))
            li_true_values.append(utility.standardize(target,reverse=True,distr_type=model_params['model_type_settings']['distr_type']) )

        elif model_params['model_name'] in ["THST","SimpleConvGRU"] :
            if model_params['model_type_settings']['location'] == 'region_grid' or model_params['model_type_settings']['twoD']==True: 
                pass
            
            if model_params['model_type_settings']['stochastic'] == False:
                
                if model_params['model_type_settings']['discrete_continuous'] == False:
                    preds = model( tf.cast(feature,tf.float16),training=False )
                    preds = tf.squeeze(preds,axis=-1) # (1, bs, seq_len, h, w)
                    preds = tf.expand_dims(preds, axis=-1 )
                
                else:
                    preds = model( tf.cast(feature,tf.float16),training=False )
                    preds = tf.squeeze(preds,axis=-1) # (1, bs, seq_len, h, w)
                    preds, probs = tf.unstack(preds, axis=0)
                    #thresholding using probability
                    
                    preds = tf.where( probs > precip_thrsh, preds, utility.standardize_ati(0.0, test_params['normalization_shift']['rain'], test_params['normalization_scales']['rain'], reverse=False) )
                    preds = tf.expand_dims(preds, axis=-1 )

                if model_params['model_type_settings']['location'] == 'region_grid' or model_params['model_type_settings']['twoD']==True:
                    preds = preds[ :, :, 6:10, 6:10, :]
                    mask = mask[ :, :, 6:10, 6:10]
                    target = target[:, :, 6:10, 6:10]

                if 'location_test' in model_params['model_type_settings'].keys(): #location_test will be a city to indicate which city to focus prediction on
                    preds = preds[:, :, idx_city_in_region[0]-6, idx_city_in_region[1]-6,: ]
                    mask = mask[ :, :, idx_city_in_region[0]-6, idx_city_in_region[1]-6]
                    target = target[ :, :, idx_city_in_region[0]-6, idx_city_in_region[1]-6]
                
                #splitting in the time dimension
                preds_std = utility.standardize_ati(preds, test_params['normalization_shift']['rain'], test_params['normalization_scales']['rain'], reverse=True)
                preds_masked = utility.water_mask( preds_std, tf.expand_dims(mask,-1)  )
                target_masked = utility.water_mask(target, mask )
                
                if "location_test" in model_params['model_type_settings'].keys():
                        #combining the batch and seq_len axis to represent timesteps
                    preds_reshaped = tf.reshape(preds_masked, [ -1, 1]  ) #(samples, timesteps)
                    targets_reshaped = tf.reshape(target_masked, [-1, 1] ) 
                else:
                    preds_reshaped = tf.reshape(preds_masked, [ preds_masked.shape[0], -1] + preds_masked.shape.as_list()[-2:] ) #(samples, timesteps, h, w)
                    targets_reshaped = tf.reshape(target_masked, [-1]+target_masked.shape.as_list()[-2:] ) #(samples, timesteps, h, w)
            
            if model_params['model_type_settings']['stochastic'] == True:

                if model_params['model_type_settings']['var_model_type'] == "mc_dropout":
                    li_preds = model.predict( tf.cast(feature,tf.float16), model_params['model_type_settings']['stochastic_f_pass'],True )
                else:
                    li_preds = model.predict( tf.cast(feature,tf.float16), model_params['model_type_settings']['stochastic_f_pass'],False )

                preds = tf.concat(li_preds, axis=-1) #(bs,ts,h,w,samples) or #(2, bs,ts,h,w,samples)
                
                if model_params['model_type_settings']['discrete_continuous'] == True:
                    preds, probs = tf.unstack( preds, axis=0)

                    preds = tf.where( probs>precip_thrsh, preds, utility.standardize_ati(0.0, test_params['normalization_shift']['rain'], test_params['normalization_scales']['rain'], reverse=False) )

                if ( model_params['model_type_settings']['location']=='region_grid' ) or model_params['model_type_settings']['twoD']==True: #focusing on centre of square only
                    preds = preds[:, :, 6:10, 6:10, :]
                    mask = mask[:, :, 6:10, 6:10]
                    target = target[:, :, 6:10, 6:10]

                    if 'location_test' in model_params['model_type_settings'].keys() :
                        preds = preds[:, :, idx_city_in_region[0]-6, idx_city_in_region[1]-6,: ]
                        mask = mask[ :, :, idx_city_in_region[0]-6, idx_city_in_region[1]-6]
                        target = target[ :, :, idx_city_in_region[0]-6, idx_city_in_region[1]-6]

                preds_std = utility.standardize_ati( preds, test_params['normalization_shift']['rain'], test_params['normalization_scales']['rain'], reverse=True)
                preds_masked = utility.water_mask( preds_std, tf.expand_dims(mask,-1)  )
                target_masked = utility.water_mask(target, mask ) 

                if "location_test" in model_params['model_type_settings'].keys():
                        #combining the batch and seq_len axis to represent timesteps
                    preds_reshaped = tf.reshape(preds_masked, [ -1, preds_masked.shape[-1] ]  ) #(samples, timesteps)
                    targets_reshaped = tf.reshape(target_masked, [-1, target_masked.shape[-1] ] ) 

                else:                
                    _ = len(preds_std.shape) - 2
                    preds_reshaped = tf.reshape( preds_std, [-1] + preds_std.shape[ -_ ] )
                    targets_reshaped = tf.reshape( target_masked, [-1] + preds_std.shape[ -_ ] )

            li_predictions.append( preds_reshaped )
            li_true_values.append( targets_reshaped )

        elif model_params['model_name'] in ['SimpleGRU','SimpleDense']:
            target, mask = target

            if (model_params['model_type_settings']['stochastic'] == False or model_params['model_type_settings']['distr_type']=="None") and model_params['model_type_settings']['stochastic_f_pass']==1 :
                preds = model( tf.cast(feature,tf.float16),training=False ) # (bs, seq_len, 1)
                preds = tf.squeeze(preds) # (1, bs, seq_len,1 )
                preds = tf.expand_dims(preds, axis=-1 ) #(bs, seq_len, 1)

                preds_masked = utility.water_mask( preds, tf.expand_dims(mask,-1)  )
                target_masked = utility.water_mask(target, mask )

                preds_std = utility.standardize_ati(preds_masked, test_params['normalization_shift']['rain'] ,test_params['normalization_scales']['rain'], reverse=True)

                preds_reshaped = tf.reshape( preds_std, [-1, 1] )
                targets_reshaped = tf.reshape( target_masked, [-1, 1] )

            else:
                if model_params['model_type_settings']['var_model_type'] == "mc_dropout":
                    li_preds = model.predict( tf.cast(feature,tf.float16), model_params['model_type_settings']['stochastic_f_pass'],True )
                else:
                    li_preds = model.predict( tf.cast(feature,tf.float16), model_params['model_type_settings']['stochastic_f_pass'] ,False )

                preds = tf.concat(li_preds, axis=-1) #(bs,ts,samples)

                preds_masked = utility.water_mask( preds, tf.expand_dims(mask,-1)  )
                target_masked = utility.water_mask(target, mask ) 
                
                preds_std = utility.standardize_ati( preds, test_params['normalization_shift']['rain'], test_params['normalization_scales']['rain'], reverse=True)
                _ = len(preds_std.shape) - 2
                preds_reshaped = tf.reshape( preds_std, [-1] + preds_std.shape[ -_: ].as_list() )
                targets_reshaped = tf.reshape( target_masked, [-1,1] )

            li_predictions.append( preds_reshaped )
            li_true_values.append( targets_reshaped )
   
        if( len(li_predictions)>=upload_size ):

            util_predict.save_preds(test_params, model_params, li_predictions, li_timestamps_chunked[:len(li_predictions)], li_true_values, precip_thrsh )
            li_timestamps_chunked = li_timestamps_chunked[len(li_predictions):]
            li_predictions = []
            li_true_values = []
    
    if len(li_predictions) >0:
        util_predict.save_preds(test_params, model_params, li_predictions, li_timestamps_chunked[:len(li_predictions)], li_true_values, precip_thrsh )
        li_timestamps_chunked = li_timestamps_chunked[len(li_predictions):]
        li_predictions = []
        li_true_values = []

    try:
        next(iter_test)
    except (tf.errors.OutOfRangeError, StopIteration, StopAsyncIteration) as e:
        pass

            
    # endregion

if __name__ == "__main__":
    s_dir = utility.get_script_directory(sys.argv[0])
    
    args_dict = utility.parse_arguments(s_dir)

    test_params, model_params = utility.load_params_test_model(args_dict)  
    
    main(test_params(), model_params)
    
