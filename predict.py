#region imports
import os
import sys

import utility

import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
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
    li_timestamps = test_params['epochs'] #TODO: add this to ATI code
    li_timestamps_chunked = [li_timestamps[i:i+test_params['batch_size']] for i in range(0, len(li_timestamps), test_params['batch_size'])] 
    li_true_values = []
    # endregion

    # region ----- Setting up datasets
    if(model_params['model_name']=="DeepSD"):
        ds = data_generators.load_data_vandal( test_params['starting_test_element'], test_params, model_params,_num_parallel_calls=test_params['num_parallel_calls'], data_dir = test_params['data_dir'], drop_remainder=True  )
    elif(model_params['model_name']=="THST" ):
        ds = data_generators.load_data_ati(test_params, model_params, None,
                                        day_to_start_at=test_params['val_end_date'] )

    iter_test = iter(ds)
    #endregion

    # region --- predictions
    for batch in range(1, int(1+test_params['test_set_size_elements']/test_params['batch_size']) ):
        try:
            feature, target = next(iter_test)
        except StopIteration as e:
            break

        if model_params['model_name'] == "DeepSD":
            
            pred = model.predict( feature, model_params['model_type_settings']['stochastic_f_pass'] ) # shape (bs ,156,352)
            pred = utility.water_mask( tf.squeeze(pred), test_params['bool_water_mask'])
            li_predictions.append(utility.standardize(pred,reverse=True,distr_type=model_params['model_type_settings']['distr_type']))
            li_true_values.append(utility.standardize(target,reverse=True,distr_type=model_params['model_type_settings']['distr_type']) )

        
        elif model_params['model_name'] == "THST":
            target, mask = target

            preds = model(feature )
            preds = tf.squeeze(preds)

            preds_masked = utility.water_mask( preds, tf.logical_not(mask)  )
            target_masked = utility.water_mask(target, tf.logical_not(mask) )

            li_predictions.append( utility.standardize_ati(preds_masked, test_params['normalization_scales']['rain'], reverse=True) )
            li_true_values.append( utility.standardize_ati(target_masked, test_params['normalization_scales']['rain'], reverse=True) )
        

        if( len(li_predictions)>=upload_size ):

            util_predict.save_preds(test_params, model_params, li_predictions, li_timestamps_chunked[:len(li_predictions)], li_true_values )
            li_timestamps_chunked = li_timestamps_chunked[len(li_predictions):]
            li_predictions = []
            li_true_values = []
    
    util_predict.save_preds(test_params, model_params, li_predictions, li_timestamps_chunked[:len(li_predictions)], li_true_values )
    li_timestamps_chunked = li_timestamps_chunked[len(li_predictions):]
    li_predictions = []
    li_true_values = []

            
    # endregion

if __name__ == "__main__":
    s_dir = utility.get_script_directory(sys.argv[0])
    
    args_dict = utility.parse_arguments(s_dir)

    #stacked DeepSd methodology
    if( args_dict['model_name'] == "DeepSD" ):
        model_type_settings = ast.literal_eval( args_dict['model_type_settings'] )
        model_layers = { 'conv1_param_custom': json.loads(args_dict['conv1_param_custom']) ,
                         'conv2_param_custom': json.loads(args_dict['conv2_param_custom']) }
        del args_dict['model_type_settings']

        test_params = hparameters.test_hparameters( **args_dict )

        # model_type_settings = {'stochastic':True ,'stochastic_f_pass':10,
        #                 'distr_type':"LogNormal", 'discrete_continuous':True,
        #                 'precip_threshold':0.5, 'var_model_type':"horseshoefactorized" }

        init_params = {}
        input_output_dims = {"input_dims": [39, 88 ], "output_dims": [ 156, 352 ] } 
        model_layers
        init_params.update(input_output_dims)
        init_params.update({'model_type_settings': model_type_settings})
        init_params.update(model_layers)

        model_params = hparameters.model_deepsd_hparameters(**init_params)()
    
    elif(args_dict['model_name'] == "THST"):
        model_params = hparameters.model_THST_hparameters()()
        args_dict['lookback_target'] = model_params['data_pipeline_params']['lookback_target']
        test_params = hparameters.test_hparameters_ati( **args_dict )
    
    main(test_params(), model_params)
    
