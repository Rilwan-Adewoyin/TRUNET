import tensorflow as tf
import pandas as pd
import numpy as np
import models
import os
import pickle
import multiprocessing as mp
import scipy as sp
import tensorflow_probability as tfp
import psutil
import utility
import util_predict
import hparameters
import sys
from matplotlib import pyplot as plt
from PIL import Image
import utility
from skimage.transform import rescale, resize, downscale_local_mean

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

def main(test_params, model_params):
    """"
    This script does 2 things
        1) Evaluates Predictions: Saves (np_rmse, np_bias, np_lower_bands, np_upper_bands, true_in_pred_range) for each day in the test set predictions
        2) 

    """


    core_count = 2 #max( psutil.cpu_count(), 4  ) 

    if type(model_params == list):
        model_params = model_params[0]
    # with mp.Pool(core_count ) as pool:
    #     _path_pred = test_params['scr_dir'] + "/Output/{}/{}/Predictions".format(model_params['model_name'], model_params['model_version'])
    #     gen_data_preds = util_predict.load_predictions_gen(_path_pred)
    #     res = pool.map_async( postproc_pipeline_evaluatepredictions, gen_data_preds, chunksize = 3  )
    #     res.wait()
    #     results = res.get()
        #This produces evaluations for a timestamps in the batch
    
    _path_pred = test_params['script_dir'] + "/Output/{}/{}/Predictions".format(model_params['model_name'], model_params['model_version'])
    gen_data_preds = util_predict.load_predictions_gen(_path_pred)
    res =  [postproc_pipeline_evaluatepredictions(pred, test_params, model_params) for pred in gen_data_preds ]

    print("Completed Prediction Evaluation")
    
    _path_pred_eval = test_params['script_dir'] + "/Output/{}/{}/EvaluatedPredictions".format(model_params['model_name'], model_params['model_version'])
    gen_data_eval_preds = util_predict.load_predictions_gen(_path_pred_eval)
    postproc_pipeline_compress_evaluations(gen_data_eval_preds,test_params, model_params)
        #averaging all evaluations across timesteps across different 
    
    print("Completed Summarisation of Prediction Evaluation")

    _path_pred_eval_summ = test_params['script_dir'] + "/Output/{}/{}/SummarisedEvaluations".format(model_params['model_name'], model_params['model_version'])
    gen_data_eval_preds = util_predict.load_predictions_gen(_path_pred_eval_summ)
    postproc_pipeline_visualized_summary(gen_data_eval_preds, test_params, model_params)
    return True


def postproc_pipeline_evaluatepredictions(batch_datum, test_params, model_params):
    """
        datum list [ [[timestamps],... ] , [ [preds_np], ... ], [[true_nps], .. ] 
        Each sublist represents the set of stochastic predictions for a timestep
    """
    ts = np.concatenate( batch_datum[0], axis=0 ) # [ ts ]
    preds = np.squeeze( np.concatenate( batch_datum[1], axis=1 ) ) #(upload_batch_size, stochastic_pred_count, width, height)  [ [preds_np], ... ]
    true = np.concatenate( batch_datum[2], axis=0) #(upload_batch_size, width, height)   [true_np,...]

    #RMSE, bias Calc
    np_rmse, np_bias = rmse_calc(preds, true) #shape( width, hieght )

    #5% and 95% uncnertainty bands
    np_lower_bands, np_upper_bands, true_in_pred_range = uncertainty_bands(preds, true) 
        #shape( batch_size, width, height), shape( batch_size, width, height), shape( batch_size, width, height)

    data_tuple = (np_rmse, np_bias, np_lower_bands, np_upper_bands, true_in_pred_range)

    _path_pred_eval = _path_pred = test_params['script_dir'] + "/Output/{}/{}/EvaluatedPredictions".format(model_params['model_name'], model_params['model_version'])
    
    if(not os.path.exists(_path_pred_eval) ):
        os.makedirs(_path_pred_eval)

    fn = str(ts[0]) + "___" + str(ts[-1]) + ".dat"

    pickle.dump( data_tuple, open( _path_pred_eval + "/" +fn ,"wb") )

    return True

def rmse_calc(preds, true):
    diff = preds - np.expand_dims( true, axis=0)
    squared = np.square( diff )
    mean = np.mean( squared,axis=0 )
    res = np.sqrt( mean )
    return res, np.mean(diff, axis=0) #(150, 156, 352) (150, 156, 352)

def uncertainty_bands(preds, true):

    shape = true.shape

    np_preds_means = np.mean( preds, axis=0)
    np_preds_stds = np.std( preds, axis=0)
    normal_distrs = tfp.distributions.Normal( loc=np_preds_means, scale=np_preds_stds)

    lower_quantiles = normal_distrs.quantile(0.05)
    upper_quantiles = normal_distrs.quantile(0.95)

    tf_bool_true_in_range = tf.logical_and( tf.greater_equal( upper_quantiles , true) , tf.greater_equal(true, lower_quantiles) )
    tf_int_true_in_range = tf.where( tf_bool_true_in_range, 1.0,0.0) #1 indicates in range, 0 indicates out of range
    
    return lower_quantiles.numpy(), upper_quantiles.numpy(), tf_int_true_in_range.numpy()

def postproc_pipeline_compress_evaluations(gen_data_eval_preds, test_params, model_params):
    """
        This will operate on batch size amount of predictions
        Each invidual prediction represents an average already
        So This method averages for each position on the map across all timesteps
    """
    avg_rmse = None
    avg_bias = None
    avg_true_in_pred_range = None
    elements_count = 0

    for idx, batch_datum in enumerate(gen_data_eval_preds):

        np_rmse = np.mean(batch_datum[0], axis=0)
        np_bias = np.mean(batch_datum[1], axis=0)
        true_in_pred_range = np.mean( batch_datum[4], axis=0)
         
        if idx==0:
            avg_rmse = np_rmse
            avg_bias = np_bias
            avg_true_in_pred_range = true_in_pred_range
        
        else:
            avg_rmse = np.average( np.stack( [avg_rmse, np_rmse],axis=-1 ), axis=-1, weights=[ elements_count , batch_datum[0].shape[0] ]  )
            avg_bias = np.average( np.stack( [avg_bias, np_bias],axis=-1 ), axis=-1, weights=[ elements_count , batch_datum[0].shape[0] ]  )
            avg_true_in_pred_range = np.average( np.stack( [avg_true_in_pred_range, true_in_pred_range],axis=-1 ), axis=-1, weights=[ elements_count , batch_datum[0].shape[0] ]  )
        
        elements_count = elements_count + batch_datum[0].shape[0]

    data_tuple = {"average_rmse": avg_rmse , "average_bias": avg_bias , "hit_rate": avg_true_in_pred_range }

    _path_pred_eval_summ = _path_pred = test_params['script_dir'] + "/Output/{}/{}/SummarisedEvaluations".format(model_params['model_name'], model_params['model_version'])
    fn = "summarised_predictions.dat"

    if(not os.path.exists(_path_pred_eval_summ) ):
        os.makedirs(_path_pred_eval_summ)
    pickle.dump( data_tuple, open( _path_pred_eval_summ + "/" +fn ,"wb") )

def postproc_pipeline_visualized_summary(gen_data_eval_preds, test_params, model_params):
    
        
    try:
        image_usa = plt.imread(test_params['script_dir'] + '/Images/us_map.jpg')
    except Exception as e:
        _img = Image.open(test_params['script_dir'] + '/Images/us_map.png')
        if _img.mode in ('RGBA', 'LA'):
            fill_color=(0, 0, 0)
            background = Image.new(_img.mode[:-1], _img.size, fill_color)
            background.paste(_img, _img.split()[-1])
            _img = background

        _img.save(test_params['script_dir'] + '/Images/us_map.jpg','JPEG')
        image_usa = plt.imread(test_params['script_dir'] + '/Images/us_map.jpg')

    dict_summarised = next(gen_data_eval_preds)

    _path_visual = test_params['script_dir'] + "/Output/{}/{}/Visualisations".format(model_params['model_name'], model_params['model_version'])
    if(not os.path.exists(_path_visual) ):
        os.makedirs(_path_visual)

    for key,array in dict_summarised.items():
        plt.imshow( resize( image_usa, array.shape , anti_aliasing=False) )
        plt.imshow( utility.water_mask(array,mask=test_params['bool_water_mask'],mode=1) , cmap='Reds')
        plt.colorbar()
        fn = _path_visual + "/" + key + ".jpg"
        plt.savefig( fn )
        plt.clf()


        print("Heatmap for ",key," Created")



if __name__ == "__main__":

    s_dir = utility.get_script_directory(sys.argv[0])

    args_dict = utility.parse_arguments(s_dir)

    train_params = hparameters.train_hparameters( **args_dict )
    
    #stacked DeepSd methodology
    li_input_output_dims = [ {"input_dims": [39, 88 ], "output_dims": [98, 220 ] , 'var_model_type':'guassian_factorized' } ,
                 {"input_dims": [98, 220 ] , "output_dims": [ 156, 352 ] , 'conv1_inp_channels':1, 'var_model_type':'guassian_factorized' }  ]

    model_params = [ hparameters.model_deepsd_hparameters(**_dict) for _dict in li_input_output_dims  ]
    model_params = [ mp() for mp in model_params]

    main(train_params(), model_params )
