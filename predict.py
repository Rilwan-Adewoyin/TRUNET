from data_generators import Generator_rain

import argparse
import ast
import itertools
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import data_generators
import hparameters
import models
import utility_predict
import utility
import custom_losses as cl


tf.keras.backend.set_floatx('float16')
tf.keras.backend.set_epsilon(1e-3)
try:
    gpu_devices = tf.config.list_physical_devices('GPU')
except Exception as e:
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')

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
tf.DType.is_compatible_with = is_compatible_with

try:
    import tensorflow_addons as tfa
except Exception as e:
    tfa = None
"""Example of how to use
"""
class TestTrueNet():
    """
        
    """    
    def __init__(self, t_params, m_params):
        self.t_params = t_params
        self.m_params = m_params

        print("GPU Available: ", tf.test.is_gpu_available() )
        # retreiving model data
        self.model, checkpoint_code = utility_predict.load_model(t_params, m_params)

        self.era5_eobs = data_generators.Era5_Eobs( self.t_params, self.m_params )

    def initialize_scheme_era5Eobs(self, location):
        """Initialization for the era5 and eobs datasets

        Args:
            location (list): [description]
        """        
        self.era5_eobs.location_size_calc(location) #Update the location the dataset generator will produce outputs for
        
        self.test_batches = self.t_params['test_batches'] * self.era5_eobs.loc_count 
        
        self.ds, self.idxs_loc_in_region = self.era5_eobs.load_data_era5eobs(batch_count=self.test_batches, start_date=self.t_params['start_date'])

        # region ------ Setting up timestamps, datasets, iterables
        self.buffer_size = self.test_batches 

        self.li_predictions = [] #list of list of tensors, each list contain a set of (maybe stochastic) predictions for the corresponding ts
        li_timestamps = self.t_params['timestamps'] #flat list of timestamps from start of test day to end 
    
        # Timestamps
        if self.era5_eobs.li_loc == ['All']:
            li_timestamps_chunked = [li_timestamps[i:i+self.t_params['window_shift']*self.t_params['batch_size']] 
                                        for i in range(0, len(li_timestamps), self.t_params['window_shift']*self.t_params['batch_size']) 
                                           if i+self.t_params['window_shift']*self.t_params['batch_size'] <= len(li_timestamps) ]

            #self.li_timestamps_chunked = list( itertools.chain.from_iterable( itertools.repeat(li_timestamps_chunked, self.era5_eobs.loc_count )) )
            self.li_timestamps_chunked = li_timestamps_chunked*self.era5_eobs.loc_count
        else:
            self.li_timestamps_chunked = [li_timestamps[i:i+self.t_params['window_shift']*self.t_params['batch_size']] for i in range(0, len(li_timestamps), self.t_params['window_shift']*self.t_params['batch_size'])] 
        
        self.li_true_values = []
        
        # Caching datasets, Creating iterable
        #if self.era5_eobs.li_loc != ['All']:
        
        if self.t_params['ctsm'] != "4ds_10years":
            cache_suffix = '{}_loctest_{}_bs_{}'.format(self.t_params['ctsm_test'] , "_".join(utility.loc_name_shrtner(self.era5_eobs.li_loc) ), t_params['batch_size'] )
        
        elif self.t_params['ctsm'] == "4ds_10years":
            cache_suffix = '{}_fyitest{}_loctest{}'.format( self.m_params['model_name'], str(self.t_params['fyi_test']), "_".join(utility.loc_name_shrtner(self.era5_eobs.li_loc) ) )
        
        cache_dir = self.t_params['data_dir']+"/data_cache/test/"
        os.makedirs( cache_dir, exist_ok=True )

        self.ds = self.ds.cache(cache_dir+cache_suffix)
        
        self.ds = self.ds.repeat(1) 
        
        self.iter_test = enumerate(self.ds)
        #endregion
    
    def predict(self, min_prob_for_rain=0.5 ):
        """Evaluates the trained model

        Args:
            model ([type]): trained model
            t_params : Dictionary containing parameters relevant to testing
            m_params : Dictionary containing parameters relevant to model
            min_prob_for_rain (float, optional): For discrete continuous, if the predicted probability of prediction is below 0, the prediction is fixed to zero. NOTE: this may be a mistake, 
        """
        
        # bounds for central region which we evaluate on 
        bounds = cl.central_region_bounds(self.m_params['region_grid_params']) 
                
        # region --- Generating predictions
        for batch in range(1, int(1+self.test_batches) ):
            
            # next batch of data
            idx, (feature, target, mask) = next(self.iter_test)

            #if region in datum is completely masked then skip to next training datum
            if( tf.reduce_any( cl.extract_central_region(mask, bounds) )==False ):
                continue
            
            if self.m_params['model_type_settings']['stochastic'] == False:
                    
                preds = self.model(feature,training=False )
                preds = tf.squeeze(preds,axis=-1)       #(bs, seq_len, h, w)
                
                if self.m_params['model_type_settings']['discrete_continuous'] == True:
                    preds, probs = tf.unstack(preds, axis=0)    #(bs, seq_len, h, w), (bs, seq_len, h, w)            
                    preds = tf.where( probs > min_prob_for_rain, preds, utility.standardize_ati(0.0, self.t_params['normalization_shift']['rain'], self.t_params['normalization_scales']['rain'], reverse=False) )
                                #thresholding using probability                    

                preds = tf.expand_dims(preds, axis=-1 )     #(bs, seq_len, h, w, 1)

                #Extracting central region of interest
                if self.era5_eobs.li_loc == ["All"] or self.t_params['t_settings'].get('region_pred',False) == True :
                    # For all we evaluate whole central regions not just the central location
                    preds   = cl.extract_central_region(preds, bounds)
                    mask    = cl.extract_central_region(mask, bounds)
                    target  = cl.extract_central_region(target, bounds)      
                                               #(bs, seq_len, h1, w1, 1)   
                elif self.era5_eobs.li_loc[0] in self.era5_eobs.rain_data.city_latlon.keys():                                    
                    preds   = preds[:, :, self.idxs_loc_in_region[0], self.idxs_loc_in_region[1],: ]
                    mask    = mask[ :, :, self.idxs_loc_in_region[0], self.idxs_loc_in_region[1]]
                    target  = target[ :, :, self.idxs_loc_in_region[0], self.idxs_loc_in_region[1]]     #(bs, seq_len, 1)
                
            elif self.m_params['model_type_settings']['stochastic'] == True:
                li_preds = self.model.predict( feature, self.m_params['model_type_settings']['stochastic_f_pass'], True )
                preds = tf.concat(li_preds, axis=-1) #(bs,ts,h,w,samples) or #(2, bs,ts,h,w,samples)
                
                if self.m_params['model_type_settings']['discrete_continuous'] == True:
                    preds, probs = tf.unstack( preds, axis=0)
                    #probs = tf.math.reduce_mean(preds, axis=-1, keepdims=True )
                    preds = tf.where( probs>min_prob_for_rain, preds, utility.standardize_ati(0.0, self.t_params['normalization_shift']['rain'], self.t_params['normalization_scales']['rain'], reverse=False) )
                        # rain thresholding

                # cropping
                if self.era5_eobs.li_loc == ["All"] or self.t_params['t_settings'].get('region_pred',False) == True :
                    #For all we evaluate whole central regions not just the central location
                    preds   = cl.extract_central_region(preds, bounds)              #(bs, seq_len, h1, w1 ,sample_size)
                    mask    = cl.extract_central_region(mask, bounds)
                    target  = cl.extract_central_region(target, bounds)             #(bs, seq_len, h1, w1)

                elif self.t_params['t_settings'].get('region_pred',False) == False: #self.era5_eobs.li_loc[0] in self.era5_eobs.rain_data.city_latlon.keys():                                        
                    preds   = preds[:, :, self.idxs_loc_in_region[0], self.idxs_loc_in_region[1],: ]
                    mask    = mask[ :, :, self.idxs_loc_in_region[0], self.idxs_loc_in_region[1]]
                    target  = target[ :, :, self.idxs_loc_in_region[0], self.idxs_loc_in_region[1]]     #(bs, seq_len, sample_size)
                
            # standardize
            preds_std = utility.standardize_ati(preds, self.t_params['normalization_shift']['rain'], self.t_params['normalization_scales']['rain'], reverse=True) #(bs, seq_len ,samples) or (bs, seq_len, h, w ,samples)

            # mask
            #TODO: make sure these masks add nan values for masked predictions, then add nanmeans to jupyter score scripts
            preds_masked = cl.water_mask(preds_std, tf.expand_dims(mask,-1), np.nan  )
            target_masked = cl.water_mask(target, mask, np.nan ) 

            #Combining the batch and seq_len dimensions into a timesteps dimension            
            preds_reshaped      = tf.reshape( preds_masked,[-1] + preds_masked.shape.as_list()[ 2: ] )    #(timesteps, ... , samples)
            targets_reshaped    = tf.reshape( target_masked,[-1] + target_masked.shape.as_list()[ 2: ] ) #(timesteps, ...)

            self.li_predictions.append( preds_reshaped )
            self.li_true_values.append( targets_reshaped )

            #Uploading predictions in batches
            bool_upload =  len(self.li_predictions)>=self.buffer_size or batch == self.test_batches
            if( bool_upload ):
                self.upload_pred(min_prob_for_rain)

        try:
            next(self.iter_test)
        except (tf.errors.OutOfRangeError, StopIteration, StopAsyncIteration) as e:
            pass

        # endregion
    
    def upload_pred(self, min_prob_for_rain):

        utility_predict.save_preds(self.t_params, self.m_params, self.li_predictions, self.li_timestamps_chunked[:len(self.li_predictions)], self.li_true_values, min_prob_for_rain, self.era5_eobs.li_loc )
        self.li_timestamps_chunked = self.li_timestamps_chunked[len(self.li_predictions):]
        self.li_predictions = []
        self.li_true_values = []


if __name__ == "__main__":
    s_dir = utility.get_script_directory(sys.argv[0])
    
    args_dict = utility.parse_arguments(s_dir)

    t_params, m_params = utility.load_params(args_dict,"test")  
    
    #main(t_params(), m_params)

    test_tru_net = TestTrueNet(t_params, m_params)
    mts = m_params['model_type_settings']
    locations = mts.get('location_test',None) if mts.get('location_test',None) != None  else mts.get('location') 

    for loc in locations:
        test_tru_net.initialize_scheme_era5Eobs(location=[loc])
        test_tru_net.predict()
        print(f"Completed Prediction for {loc}")