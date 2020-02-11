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
import json

#This script needs to produce map plots of prediction quality and produce relevant statistics on performance

#Statistics to Estimate
#-----Normal----
##RMSE
##Bias
##R20
##SPII

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
        1) Evaluates Predictions: Saves (np_rmse, np_bias, np_lower_bands, np_upper_bands, true_in_pred_range) for each day in the test set predictions: postproc_pipeline_evaluatepredictions
        2) Summarises Predictions: Saves the averages of all evaluations for a given timestep across timesteps 
        3) Creates a valisualization of Summarised Predictions

    """

    core_count = 2 #max( psutil.cpu_count(), 4  ) 

    if type(model_params) == list:
        model_params = model_params[0]
    # with mp.Pool(core_count ) as pool:
        #     _path_pred = test_params['scr_dir'] + "/Output/{}/{}/Predictions".format(model_params['model_name'], model_params['model_version'])
        #     gen_data_preds = util_predict.load_predictions_gen(_path_pred)
        #     res = pool.map_async( postproc_pipeline_evaluatepredictions, gen_data_preds, chunksize = 3  )
        #     res.wait()
        #     results = res.get()
            #This produces evaluations for a timestamps in the batch
    
    #region 1
    _path_pred = test_params['script_dir'] + "/Output/{}/{}_{}_{}/{}/Predictions".format(model_params['model_name'],model_params['model_type_settings']['var_model_type'],
                                                        model_params['model_type_settings']['distr_type'],str(model_params['model_type_settings']['discrete_continuous']), model_params['model_version'])
    gen_data_preds = util_predict.load_predictions_gen(_path_pred)
    res =  [postproc_pipeline_evaluatepredictions(pred, test_params, model_params) for pred in gen_data_preds ]

    print("Completed Prediction Evaluation")
    # endregion

    #region 2
    _path_pred_eval = test_params['script_dir'] + "/Output/{}/{}_{}_{}/{}/EvaluatedPredictions".format(model_params['model_name'],model_params['model_type_settings']['var_model_type'],
                                                        model_params['model_type_settings']['distr_type'],str(model_params['model_type_settings']['discrete_continuous']),model_params['model_version'])
    gen_data_eval_preds = util_predict.load_predictions_gen(_path_pred_eval)
    postproc_pipeline_compress_evaluations(gen_data_eval_preds,test_params, model_params)
        #averaging all evaluations across timesteps across different 
    
    print("Completed Summarisation of Prediction Evaluation")
    # endregion

    #region 3
    _path_pred_eval_summ = test_params['script_dir'] + "/Output/{}/{}_{}_{}/{}/SummarisedEvaluations".format(model_params['model_name'], model_params['model_type_settings']['var_model_type'],
                                                        model_params['model_type_settings']['distr_type'],str(model_params['model_type_settings']['discrete_continuous']) ,model_params['model_version'])
    gen_data_eval_preds = util_predict.load_predictions_gen(_path_pred_eval_summ)
    postproc_pipeline_visualized_summary(gen_data_eval_preds, test_params, model_params)
    # endregion

    return True


def postproc_pipeline_evaluatepredictions(batch_datum, test_params, model_params):
    """
        datum list [ [[timestamps],... ] , [ [preds_np], ... ], [[true_nps], .. ] 
        Each sublist represents the set of stochastic predictions for a timestep
    """
    ts = np.concatenate( batch_datum[0], axis=0 ) # [ ts ]
    preds = np.squeeze( np.concatenate( batch_datum[1], axis=1 ) ) #(stochastic_pred_count, upload_batch_size, width, height)  [ [preds_np], ... ]
    true = np.concatenate( batch_datum[2], axis=0) #(upload_batch_size, width, height)   [true_np,...]

    #RMSE, bias Calc
    np_rmse, np_bias = rmse_calc(preds, true) #shape( width, hieght )

    #5% and 95% uncnertainty bands
    np_lower_bands, np_upper_bands, true_in_pred_range = uncertainty_bands(preds, true) 
        #shape( batch_size, width, height), shape( batch_size, width, height), shape( batch_size, width, height)

    np_r20_err = r20_error_calc(preds, true)

    sdII_data = sdII_error_calc(preds, true, wd=0.5)  #shape( batch_size, width, height) #(total_obs_precip_wd-total_pred_precip_wd, element_count_in_batch) 

    data_tuple = (np_rmse, np_bias, np_lower_bands, np_upper_bands, true_in_pred_range, np_r20_err, sdII_data )

    _path_pred_eval = _path_pred = test_params['script_dir'] + "/Output/{}/{}_{}_{}/{}/EvaluatedPredictions".format(model_params['model_name'], model_params['model_type_settings']['var_model_type'],
                                                        model_params['model_type_settings']['distr_type'],str(model_params['model_type_settings']['discrete_continuous']),model_params['model_version'])
    
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

#def copy_his_plots_for_mse_bias

def r20_error_calc(preds, true):
    """R20 - Very heavy wet days ≥ 20mm
       This returns a matrices of differences for the predictions and observed occures of R20 events
    """

    bool_r20_true = tf.where( true >= 20, 1, 0 )
    bool_r20_pred = tf.where( tf.reduce_mean( preds, axis=0) >= 20, 1, 0 )

    return (bool_r20_true-bool_r20_pred).numpy()
    
def sdII_error_calc(preds, true, wd=0.5):
    """
    This calculates the unaverage sDII numbers for both preds and true, also includes a count of number of elements in batch
        These can be used in the next step to create a weighted mean of SDII error across all batches
    """

    bool_wd = true>=wd

    total_obs_precip_wd = np.sum( tf.boolean_mask( bool_wd, true ) )
    total_pred_precip_wd = np.sum( tf.boolean_mask( bool_wd, tf.reduce_mean( preds, axis=0)) )
    element_count_in_batch = np.count_nonzero( bool_wd )

    return [total_obs_precip_wd-total_pred_precip_wd,element_count_in_batch]


def pixel_calibration(preds, true):
    shape = true.shape

    np_preds_means = np.mean( preds, axis=0)
    np_preds_stds = np.std( preds, axis=0)
    normal_distrs = tfp.distributions.Normal( loc=np_preds_means, scale=np_preds_stds)

    lower_quantiles = normal_distrs.quantile(0.15)
    upper_quantiles = normal_distrs.quantile(0.85)


def postproc_pipeline_compress_evaluations(gen_data_eval_preds, test_params, model_params):
    """
        This will operate on batch size amount of predictions
        Each invidual prediction represents an average already
        So This method averages for each position on the map across all timesteps

        New Behaviour: Since we need to calculate standard deviations as well, all data must be held in memory
    """
    avg_rmse = None
    avg_bias = None
    avg_true_in_pred_range = None

    r20_diffs = None
    
    li_sdII_errors = []
    li_sdII_element_count = []

    elements_count = 0

    for idx, batch_datum in enumerate(gen_data_eval_preds):

        np_rmse = np.mean(batch_datum[0], axis=0)
        np_bias = np.mean(batch_datum[1], axis=0)
        true_in_pred_range = np.mean( batch_datum[4], axis=0)
         
        if idx==0:
            avg_rmse = np_rmse
            avg_bias = np_bias
            avg_true_in_pred_range = true_in_pred_range
            
            #new methods
            r20_diffs = batch_datum[5]

            li_sdII_errors.append( batch_datum[6][0] )
            li_sdII_element_count.append( batch_datum[6][1] )

        
        else:
            #old method
            avg_rmse = np.average( np.stack( [avg_rmse, np_rmse],axis=-1 ), axis=-1, weights=[ elements_count , batch_datum[0].shape[0] ]  )
            avg_bias = np.average( np.stack( [avg_bias, np_bias],axis=-1 ), axis=-1, weights=[ elements_count , batch_datum[0].shape[0] ]  )
            avg_true_in_pred_range = np.average( np.stack( [avg_true_in_pred_range, true_in_pred_range],axis=-1 ), axis=-1, weights=[ elements_count , batch_datum[0].shape[0] ]  )
            
            #new method
            r20_diffs = np.concatenate( [r20_diffs, batch_datum[5]] ,axis=0 )

            li_sdII_errors.append( batch_datum[6][0] )
            li_sdII_element_count.append( batch_datum[6][1] )
        
        elements_count = elements_count + batch_datum[0].shape[0]

    #final calculations
    r20_mean = np.mean(r20_diffs)
    r20_stds = np.std(r20_diffs)

    sdII_mean =np.sum(li_sdII_errors) / np.sum( li_sdII_element_count )
    sdII_std = np.std( np.array(li_sdII_errors) ) 
    

    data_tuple = {"rmse-mean": avg_rmse , "bias-mean": avg_bias , "hit-rate": avg_true_in_pred_range, "r20-mean":r20_mean,  "r20-stds":r20_stds, "sdII_mean":sdII_mean, "sdII_std": sdII_std }

    scores_for_table = {"rmse-mean-mean":np.mean(avg_rmse),  "bias-mean": np.mean(avg_bias) , "hit-rate": np.mean(avg_true_in_pred_range), "r20-mean":r20_mean,  "r20-stds":r20_stds, "sdII_mean":sdII_mean, "sdII_std": sdII_std  }
    # latex_table_maker(data_tuple)

    _path_pred_eval_summ = test_params['script_dir'] + "/Output/{}/{}_{}_{}/{}/SummarisedEvaluations".format(model_params['model_name'],  model_params['model_type_settings']['var_model_type'],
                                                        model_params['model_type_settings']['distr_type'],str(model_params['model_type_settings']['discrete_continuous']), model_params['model_version'])
    fn = "summarised_predictions.dat"

    if(not os.path.exists(_path_pred_eval_summ) ):
        os.makedirs(_path_pred_eval_summ)

    pickle.dump( data_tuple, open( _path_pred_eval_summ + "/" +fn ,"wb") )

    with open( _path_pred_eval_summ + "/" +"scores_for_table.json" ,"w") as f:
        json.dump(scores_for_table,f,sort_keys=True, indent=4)

    

# def latex_table_maker(data_tuple):
#     """ make the latex table containing results"""
#     tab1 = pd.DataFrame(tab1).set_index('name')
#     tab1.append(data_tuple)

#     tab1_latex = tab1[['rmse-mean','bias-mean','hit-rate','r20-mean','r20-stds',"sdII_mean","sdII_std"]].to_latex(float_format='{:,.3f}'.format)

    
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

    _dict = next(gen_data_eval_preds)
    keys_to_keep = ["rmse-mean","bias-mean","hit-rate"]
    dict_summarised = { k: _dict[k] for k in keys_to_keep }

    _path_visual = test_params['script_dir'] + "/Output/{}/{}_{}_{}/{}/Visualisations".format(model_params['model_name'], model_params['model_type_settings']['var_model_type'],
                                                        model_params['model_type_settings']['distr_type'],str(model_params['model_type_settings']['discrete_continuous']), model_params['model_version'])
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

    #stacked DeepSd methodology
    if( args_dict['model_name'] == "DeepSD" ):
        test_params = hparameters.test_hparameters( **args_dict )()

        model_type_settings = {'stochastic':True ,'stochastic_f_pass':10,
                        'distr_type':"LogNormal", 'discrete_continuous':True,
                        'precip_threshold':0.5, 'var_model_type':"horseshoefactorized" }

        input_output_dims = {"input_dims": [39, 88 ], "output_dims": [ 156, 352 ], 'model_type_settings': model_type_settings } 

        model_params = hparameters.model_deepsd_hparameters(**input_output_dims)()
    
    elif(args_dict['model_name'] == "THST"):
        model_params = hparameters.model_THST_hparameters()()
        args_dict['lookback_target'] = model_params['data_pipeline_params']['lookback_target']
        test_params = hparameters.test_hparameters_ati( **args_dict )()
    
    main(test_params, model_params)
