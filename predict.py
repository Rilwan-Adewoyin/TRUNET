#region imports
import os
import sys

import utility

import tensorflow as tf
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
# endregion

def main( test_params, model_params):
    print("GPU Available: ", tf.test.is_gpu_available() )

    # 1) Instantiate Model
    model, checkpoint_code = util_predict.load_model(test_params, model_params)

    # 2) Form Predictions
    predict(model, test_params, checkpoint_code )

def predict( model, test_params, checkpoint_no ):
    
    # region ------ Setting up testing variables
    upload_size = int( (test_params['test_set_size_elements']/test_params['batch_size'] )* test_params['dataset_pred_batch_reporting_freq'] )
    li_predictions = [] #This will be a list of list of tensors, each list contain a set of stochastic predictions for the corresponding ts
    li_timestamps = test_params['dates_tss']
    li_timestamps_chunked = [li_timestamps[i:i+test_params['batch_size']] for i in range(0, len(li_timestamps), test_params['batch_size'])] 
    li_true_values = []
    # endregion

    # region ----- Setting up datasets

    ds = data_generators.load_data_vandal( test_params['starting_test_element'], test_params, _num_parallel_calls=test_params['num_parallel_calls']  )
    iter_test = iter(ds)
    #endregion

    # region --- predictions
    for batch in range(1, int(1+test_params['test_set_size_elements']/test_params['batch_size']) ):
        feature, target = next(iter_test)

        pred = model.predict( feature, test_params['num_preds'] ) # shape (bs ,156,352)
        pred = utility.water_mask( tf.squeeze(pred), test_params['bool_water_mask'])
        li_predictions.append(utility.standardize(pred,reverse=True))
        li_true_values.append(utility.standardize(target,reverse=True) )



        if( len(li_predictions)>=upload_size ):

            util_predict.save_preds(test_params, model_params, li_predictions, li_timestamps_chunked[:len(li_predictions)], li_true_values )
            li_timestamps_chunked = li_timestamps_chunked[len(li_predictions):]
            li_predictions = []
            li_true_values = []
            



    # endregion

if __name__ == "__main__":
    s_dir = utility.get_script_directory(sys.argv[0])
    
    args_dict = utility.parse_arguments(s_dir)

    test_params = hparameters.test_hparameters( **args_dict )

    #stacked DeepSd methodology
    li_input_output_dims = [ {"input_dims": [39, 88 ], "output_dims": [98, 220 ] , 'var_model_type':'guassian_factorized' } ,
                 {"input_dims": [98, 220 ] , "output_dims": [ 156, 352 ] , 'conv1_inp_channels':1, 'var_model_type':'guassian_factorized' }  ]

    model_params = [ hparameters.model_deepsd_hparameters(**_dict) for _dict in li_input_output_dims  ]
    model_params = [ mp() for mp in model_params]

    #TrajGRU Methodology

    main( test_params(), model_params )
