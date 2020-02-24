#region imports
import os
import sys

import utility

import tensorflow as tf
try:
    gpu_devices = tf.config.list_physical_devices('GPU')
except Exception as e:
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
print(gpu_devices)
for idx, gpu_name in enumerate(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_name, True)

from tensorflow.keras.mixed_precision import experimental as mixed_precision
##comment the below two lines out if training DEEPSD
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

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
import numpy as np

import argparse 
from tqdm import tqdm
import traceback
import time

import models
import hparameters
import data_generators
import util_predict
import utility
import json
import ast
# endregion

def main( test_params, model_params):
    print("GPU Available: ", tf.test.is_gpu_available() )

    # 1) Instantiate Model
    model, checkpoint_code = util_predict.load_model(test_params, model_params)

    # 2) Form Predictions
    predict(model, test_params, model_params ,checkpoint_code )

def predict( model, test_params, model_params ,checkpoint_no ):
    
    # region ------ Setting up testing variables
    upload_size = int( (test_params['test_set_size_elements']/test_params['batch_size'] )* test_params['dataset_pred_batch_reporting_freq'] )
    li_predictions = [] #This will be a list of list of tensors, each list contain a set of (maybe stochastic) predictions for the corresponding ts
    li_timestamps = test_params['epochs']
    if model_params['model_name']=="DeepSD" :
        li_timestamps_chunked = [li_timestamps[i:i+test_params['batch_size']] for i in range(0, len(li_timestamps), test_params['batch_size'])] 
    elif model_params['model_name'] == "THST":
        li_timestamps_chunked = [li_timestamps[i:i+test_params['window_shift']] for i in range(0, len(li_timestamps), test_params['window_shift'])] 
    li_true_values = []
    # endregion

    # region ----- Setting up datasets
    if(model_params['model_name']=="DeepSD"):
        ds = data_generators.load_data_vandal( test_params['starting_test_element'], 
                test_params, model_params,_num_parallel_calls=test_params['num_parallel_calls'], data_dir = test_params['data_dir'], drop_remainder=True  )
        ds = ds.take( int(test_params['test_set_size_elements']/test_params['batch_size']) )

    elif(model_params['model_name']=="THST" ):
        ds = data_generators.load_data_ati(test_params, model_params, None,
                                        day_to_start_at=test_params['test_start_date'], data_dir=test_params['data_dir'] )
        ds = ds.take( test_params['total_datums'] )
    iter_test = enumerate(ds)
    #endregion

    # region --- predictions
    
    for batch in range(1, int(1+test_params['test_set_size_elements']/test_params['batch_size']) ):

        idx,(feature, target) = next(iter_test)

        if model_params['model_name'] == "DeepSD":
            
            pred = model.predict( feature, model_params['model_type_settings']['stochastic_f_pass'] ) # shape (bs ,156,352)
            pred = utility.water_mask( tf.squeeze(pred), test_params['bool_water_mask'])
            li_predictions.append(utility.standardize(pred,reverse=True,distr_type=model_params['model_type_settings']['distr_type']))
            li_true_values.append(utility.standardize(target,reverse=True,distr_type=model_params['model_type_settings']['distr_type']) )

        
        elif model_params['model_name'] == "THST":
            target, mask = target

            preds = model( tf.cast(feature,tf.float16),training=False )
            if model_params['model_type_settings']['stochastic'] == False:
                preds = tf.expand_dims(preds, axis=0 )

            preds = tf.squeeze(preds,axis=-1) # (pred_count, bs, seq_len, h, w)
            #splitting in the time dimension
            preds_masked = utility.water_mask( preds, mask  )
            target_masked = utility.water_mask(target, mask )

            preds_std = utility.standardize_ati(preds_masked, test_params['normalization_scales']['rain'], reverse=True)
            targets_std = utility.standardize_ati(target_masked, test_params['normalization_scales']['rain'], reverse=True)

            #combining the batch and seq_len axis to represent timesteps
            
            preds_reshaped = tf.reshape(preds_std, [ preds_std.shape[0], -1] + preds_std.shape.as_list()[-2:] )
            targets_reshaped = tf.reshape(targets_std, [-1]+preds_std.shape.as_list()[-2:] )

            li_predictions.append( preds_reshaped )
            li_true_values.append( targets_reshaped )
        

        if( len(li_predictions)>=upload_size ):

            util_predict.save_preds(test_params, model_params, li_predictions, li_timestamps_chunked[:len(li_predictions)], li_true_values )
            li_timestamps_chunked = li_timestamps_chunked[len(li_predictions):]
            li_predictions = []
            li_true_values = []
    
    if len(li_predictions) >0:
        util_predict.save_preds(test_params, model_params, li_predictions, li_timestamps_chunked[:len(li_predictions)], li_true_values )
        li_timestamps_chunked = li_timestamps_chunked[len(li_predictions):]
        li_predictions = []
        li_true_values = []

            
    # endregion

if __name__ == "__main__":
    s_dir = utility.get_script_directory(sys.argv[0])
    
    args_dict = utility.parse_arguments(s_dir)

    train_params, model_params = utility.load_params_train_model()  
    
    main(test_params(), model_params)
    
