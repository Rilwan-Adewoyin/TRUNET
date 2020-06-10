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
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl
from tqdm import tqdm

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
#from tensorflow.python.framework.dtypes import DType
tf.DType.is_compatible_with = is_compatible_with

try:
    import tensorflow_addons as tfa
except Exception as e:
    tfa = None

class TestTrueNet():

    def __init__(self, t_params, m_params):
        self.t_params = t_params
        self.m_params = m_params

        print("GPU Available: ", tf.test.is_gpu_available() )
        # retreiving model data
        self.model, checkpoint_code = utility_predict.load_model(t_params, m_params)

        model_settings = self.m_params['model_type_settings']

        self.era5_eobs = data_generators.Era5_Eobs( self.t_params, self.m_params )

    def initialize_scheme_era5Eobs(self, location):
        """[summary]

        Args:
            location (list): [description]
        """        
        self.era5_eobs.location_size_calc(location) #Update the location the dataset generator will produce outputs for
        
        self.test_batches = self.t_params['test_batches'] * len( self.era5_eobs.loc_count )

        self.ds, self.idxs_loc_in_region = self.era5_eobs.load_data_era5eobs(  batch_count=self.test_batches)

        # region ------ Setting up timestamps and datasets  
        self.buffer_size = self.test_batches 

        self.li_predictions = [] #list of list of tensors, each list contain a set of (maybe stochastic) predictions for the corresponding ts
        li_timestamps = t_params['timestamps'] #flat list of timestamps from start of test day to end 
    
        # Timestamps
        if self.era5_eobs.li_loc == ['All']:

            li_timestamps_chunked = [li_timestamps[i:i+self.t_params['window_shift']*self.t_params['batch_size']] for i in range(0, len(li_timestamps), self.t_params['window_shift']*self.t_params['batch_size'])]
            self.li_timestamps_chunked = list( itertools.chain.from_iterable( itertools.repeat(li_timestamps_chunked, self.era5_eobs.loc_count )) )

        else:
            self.li_timestamps_chunked = [li_timestamps[i:i+self.t_params['window_shift']*self.t_params['batch_size']] for i in range(0, len(li_timestamps), self.t_params['window_shift']*self.t_params['batch_size'])] 

        self.li_true_values = []
        
        
        # Caching datasets
        if t_params['ctsm'] != "4ds_10years":
            cache_suffix = '_{}_bs_{}_loctest_{}_{}'.format( m_params['model_name'], t_params['batch_size'],m_params['model_type_settings']['location_test'],m_params['model_type_settings']['location']  ).strip('[]') 
        
        elif t_params['ctsm'] == "4ds_10years":
            cache_suffix = '{}_bs_fyitest{}_loctest{}'.format( m_params ['model_name'], str(t_params['fyi_test']), str(m_params['model_type_settings']['location_test']) )
        
        self.ds = self.ds.cache('Data/data_cache/ds_test_cache'+cache_suffix ).repeat(1) 

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
        
        bounds = cl.central_region_bounds(self.m_params['region_grid_params']) #bounds for central region which we evaluate on 
        # region --- Generating predictions
        
        for batch in range(1, int(1+self.test_batches) ):
            
            # Getting next batch of data
            idx, (feature, target, mask) = next(self.iter_test)

            if self.m_params['model_type_settings']['stochastic'] == False:
                    
                preds = self.model( tf.cast(feature,tf.float16),training=False )
                preds = tf.squeeze(preds,axis=-1)       #(bs, seq_len, h, w)
                
                if self.m_params['model_type_settings']['discrete_continuous'] == True:

                    preds, probs = tf.unstack(preds, axis=0)    #(bs, seq_len, h, w), (bs, seq_len, h, w)
                    
                    #thresholding using probability                    
                    preds = tf.where( probs > min_prob_for_rain, preds, utility.standardize_ati(0.0, t_params['normalization_shift']['rain'], t_params['normalization_scales']['rain'], reverse=False) )
                
                preds = tf.expand_dims(preds, axis=-1 )     #(bs, seq_len, h, w, 1)

                #Extracting central region of interest

                if self.era5_eobs.li_loc[0] in self.era5_eobs.rain_data.city_latlon.keys():
                                        
                    preds   = preds[:, :, self.idxs_loc_in_region[0], self.idxs_loc_in_region[1],: ]
                    mask    = mask[ :, :, self.idxs_loc_in_region[0], self.idxs_loc_in_region[1]]
                    target  = target[ :, :, self.idxs_loc_in_region[0], self.idxs_loc_in_region[1]]     #(bs, seq_len, 1)
                
                elif self.era5_eobs.li_loc == ["All"]:
                    #For all we evaluate whole central regions not just the central location
                    preds   = cl.extract_central_region(preds, bounds)
                    mask    = cl.extract_central_region(mask, bounds)
                    target  = cl.extract_central_region(target, bounds)                                 #(bs, seq_len, h1, w1, 1)              

            elif self.m_params['model_type_settings']['stochastic'] == True:

                li_preds = self.model.predict( tf.cast(feature,tf.float16), self.m_params['model_type_settings']['stochastic_f_pass'],True )

                preds = tf.concat(li_preds, axis=-1) #(bs,ts,h,w,samples) or #(2, bs,ts,h,w,samples)
                
                #handling dual output of discrete continuous
                if self.m_params['model_type_settings']['discrete_continuous'] == True:
                    preds, probs = tf.unstack( preds, axis=0)

                    # rain thresholding #NOTE: This need to be corrected
                    preds = tf.where( probs>min_prob_for_rain, preds, utility.standardize_ati(0.0, t_params['normalization_shift']['rain'], t_params['normalization_scales']['rain'], reverse=False) )

                # cropping
                if self.era5_eobs.li_loc[0] in self.era5_eobs.rain_data.city_latlon.keys():                                        
                    preds   = preds[:, :, self.idxs_loc_in_region[0], self.idxs_loc_in_region[1],: ]
                    mask    = mask[ :, :, self.idxs_loc_in_region[0], self.idxs_loc_in_region[1]]
                    target  = target[ :, :, self.idxs_loc_in_region[0], self.idxs_loc_in_region[1]]     #(bs, seq_len, sample_size)
                
                elif self.era5_eobs.li_loc == ["All"]:
                    #For all we evaluate whole central regions not just the central location
                    preds   = cl.extract_central_region(preds, bounds)              #(bs, seq_len, h1, w1 ,sample_size)
                    mask    = cl.extract_central_region(mask, bounds)
                    target  = cl.extract_central_region(target, bounds)             #(bs, seq_len, h1, w1)

            # standardize
            preds_std = utility.standardize_ati(preds, t_params['normalization_shift']['rain'], t_params['normalization_scales']['rain'], reverse=True) #(bs, seq_len ,samples) or (bs, seq_len, h, w ,samples)

            # mask
            preds_masked = cl.water_mask( preds_std, mask  )
            target_masked = cl.water_mask(target, mask ) 

            #Combining the batch and seq_len dimensions to get a timesteps dimension            
            preds_reshaped      = tf.reshape( preds_masked, [-1] + preds_masked.shape.as_list[ 2: ] )    #(timesteps, ... , samples)
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

    t_params, m_params = utility.load_params_test_model(args_dict)  
    
    #main(t_params(), m_params)

    test_tru_net = TestTrueNet(t_params(), m_params)

    locations = m_params.get('location_test',None) if m_params.get('location_test',None) != None  else m_params.get('location') 

    for loc in locations:
        test_tru_net.initialize_scheme_era5Eobs(location=loc)
        test_tru_net.predict()