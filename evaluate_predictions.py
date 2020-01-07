import tensorflow as tf
import pandas as pd
import numpy as np
import models
import os
import pickle
import multiprocessing as mp
import scipy as sp
from statistics import NormalDist
import tensorflow_probability as tfp
impmort psutil

import util_predict

#This script needs to produce map plots of prediction quality and produce relevant statistics on performance

#Statistics to Estimate
#-----Normal----
##RMSE
##Bias

# ------Uncertainty Bands
##For all regions in every location calculate the frequency of predictions falling within a 95% confidence range
### Then calculate a subset of this focusing on the frequency of predictions for heavy rain days that fall into the prediction interval

#Method
##For RMSE and Bias produce one (width, height) matrix representing the average loss per spot on grid
##Then create plot function which maps the values on this matrix onto a map of USA and prints out results
## Then find out how to do the Climdex and SDII
##For uncertainty produce one array (preds, weight, height, 5%lq , 95% uq )

#Create plots comparing RMSE, bias to actual true precipitation

def main(test_params):

    

    

    core_count = 2 #max( psutil.cpu_count(), 4  ) 

    with mp.Pool(core_count ) as pool:
        _path_pred = test_params['scr_dir'] + "/Output/{}/{}/Predictions".format(model_params['model_name'], test_params['model_version'])
        gen_data_preds = util_predict.load_predictions_gen(_path_pred)
        res = pool.map_async( postproc_pipeline_evaluatepredictions, gen_data_preds, chunksize = 3  )
        res.wait()
        results = res.get()
        #This produces evaluations for a timestamps in the batch
    
    print("Completed Prediction Evaluation")
    
    _path_pred_eval = _path_pred = test_params['scr_dir'] + "/Output/{}/{}/EvaluatedPredictions".format(model_params['model_name'], test_params['model_version'])
    gen_data_eval_preds = util_predict.load_predictions_gen(_path)
    postproc_pipeline_compress_evaluations(postproc_pipeline_compress_evaluations)
        #averaging all evaluations across timesteps across different 
    
    print("Completed Summarisation of Prediction Evaluation")

    return True



    
    


def postproc_pipeline_evaluatepredictions(batch_datum, test_params):
    """
        datum list [ [timestamps], [ [preds_np], ... ], [true_np,...] ] 
        Each sublist represents the set of stochastic predictions for a timestep
    """
    ts = batch_datum[0] # [ ts ]
    preds = tf.stack( batch_datum[1], axis=0 ) #(batch_size, stochastic_pred_count, width, height)  [ [preds_np], ... ]
    true = tf.stack( batch_datum[2], axis=0) #(batch_size, width, height)   [true_np,...]

    #RMSE, bias Calc
    np_rmse, np_bias = rmse_calc(preds, true) #shape( width, hieght )

    #5% and 95% uncnertainty bands
    np_lower_bands, np_upper_bands, true_in_pred_range = uncertainty_bands(preds, true) 
        #shape( batch_size, width, height), shape( batch_size, width, height), shape( batch_size, width, height)

    data_tuple = (np_rmse, np_bias, np_lower_bands, np_upper_bands, true_in_pred_range)

    _path_pred_eval = _path_pred = test_params['scr_dir'] + "/Output/{}/{}/EvaluatedPredictions".format(model_params['model_name'], test_params['model_version'])
    fn = str(ts[0]) + "___" + str(ts[-1])

    pickle.dump( data_tuple, open( _path_pred_eval + "/" +fn ,"wb") )

    return True

def rmse_calc(preds, true):
    diff = preds - tf.expand_dims( true, axis=1)
    squared = np.square( diff )
    mean = np.reduce_mean( squared,axis=1 )
    res = np.sqrt( mean )
    return res, np.reduce_mean(diff, axis=1)

def uncertainty_bands(preds, true):

    shape = true.shape

    np_preds_means = np.reduce_mean( preds, axis=1)
    np_preds_stds = np.reduce_std( preds, axis=1)
    normal_distrs = tfp.distributions.Normal( loc=np_preds_mean, scale=np_preds_stds)

    lower_quantiles = normal_distrs.quantile(0.05)
    upper_quantiles = normal_distrs.quantile(0.95)

    tf_bool_true_in_range = tf.logical_and( tf.greater_equal( upper_quantiles , true) , tf.greater_equal(true, lower_quantiles) )
    tf_int_true_in_range = tf.where( tf_bool_true_in_range, 1.0,0.0) #1 indicates in range, 0 indicates out of range
    
    return lower_quantiles.numpy(), upper_quantiles.numpy(), tf_int_true_in_range.numpy()

def postproc_pipeline_compress_evaluations(postproc_pipeline_compress_evaluations):
    """
        This will operate on batch size amount of predictions
        Each invidual prediction represents an average already
        So This method averages across the batch sizes
    """
    avg_rmse = None
    avg_bias = None
    avg_true_in_pred_range = None
    elements_count = 0

    for idx, batch_datum in enumerate(gen_data_eval_preds):
        elements_count = elements_count + gen_data_eval_preds[0].shape[0]
        np_rmse = np.mean(batch_datum[0], axis=0)
        np_bias = np.mean(batch_datum[1], axis=0)
        true_in_pred_range = np.mean( batch_datum[4], axis=0)
         
        
        if idx==0:
            avg_rmse = np_rmse
            avg_bias = np_bias
            avg_true_in_pred_range = true_in_pred_range
        
        else:
            avg_rmse = np.average( np.stack( [avg_rmse, np_rmse],axis=-1 ), axi=-1, weights=[ elements_count , 1.0 ]  )
            avg_bias = np.average( np.stack( [avg_bias, np_bias],axis=-1 ), axi=-1, weights=[ elements_count , 1.0 ]  )
            avg_true_in_pred_range = np.average( np.stack( [avg_true_in_pred_range, true_in_pred_range],axis=-1 ), axi=-1, weights=[ elements_count , 1.0 ]  )
    
    _path_pred_eval = _path_pred = test_params['scr_dir'] + "/Output/{}/{}/SummarisedEvaluations".format(model_params['model_name'], test_params['model_version'])
    fn = str(ts[0]) + "___" + str(ts[-1])

    pickle8.dump( data_tuple, open( _path_pred_eval + "/" +fn ,"wb") )



        



        
        

    








if __name__ == "__main__":

    args_dict = utility.parse_arguments()

    train_params = hparameters.train_hparameters( **args_dict )
    model_params = hparameters.model_deepsd_hparameters()

    train_loop(train_params(), model_params() )
