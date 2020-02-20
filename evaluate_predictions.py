import tensorflow as tf
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# print(gpu_devices)
# for idx, gpu_name in enumerate(gpu_devices):
#     tf.config.experimental.set_memory_growth(gpu_name, True)

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
import pywt
from skimage.transform import rescale, resize, downscale_local_mean
import json
import ast
import gc



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
        1) Evaluates Predictions: Saves (np_rmse, np_bias, np_lower_bands, np_upper_bands, true_in_pred_range) for each day for each position in the test set predictions: postproc_pipeline_evaluatepredictions
        2) Summarises Predictions: Saves the averages of all evaluations for a given timestep across timesteps 
        3) Creates a valisualization of Summarised Predictions

    """

    core_count = 2 #max( psutil.cpu_count(), 4  ) 

    if type(model_params) == list:
        model_params = model_params[0]

    
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
    #gen_data_eval_preds = util_predict.load_predictions_gen(_path_pred_eval)
    li_gen_data_eval_preds = [ util_predict.load_predictions_gen(_path_pred_eval) for idx in range(4) ]
    postproc_pipeline_compress_evaluations(li_gen_data_eval_preds,test_params, model_params)
        #averaging all evaluations across timesteps across different 
    
    print("Completed Summarisation of Prediction Evaluation")
    # endregion

    #region 3
    _path_pred_eval_summ = test_params['script_dir'] + "/Output/{}/{}_{}_{}/{}/SummarisedEvaluations".format(model_params['model_name'], model_params['model_type_settings']['var_model_type'],
                                                        model_params['model_type_settings']['distr_type'],str(model_params['model_type_settings']['discrete_continuous']) ,model_params['model_version'])
    
    li_gen_data_eval_preds = util_predict.load_predictions_gen(_path_pred_eval_summ)

    if model_params['model_name']=="DeepSD":
        postproc_pipeline_visualized_summary(li_gen_data_eval_preds, test_params, model_params)
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

    np_r20_err = r20_error_calc(preds, true)

    pc_binary_in_range = pixel_calibration(preds, true, model_params['model_type_settings']['distr_type'] ,rainy_threshold=0.5) #shape (bs, h, w, 100 ) , [list of indexes]

    data_tuple = (np_rmse, np_bias, np_r20_err, pc_binary_in_range   )

    _path_pred_eval = _path_pred = test_params['script_dir'] + "/Output/{}/{}_{}_{}/{}/EvaluatedPredictions".format(model_params['model_name'], model_params['model_type_settings']['var_model_type'],
                                                        model_params['model_type_settings']['distr_type'],str(model_params['model_type_settings']['discrete_continuous']),model_params['model_version'])
    
    if(not os.path.exists(_path_pred_eval) ):
        os.makedirs(_path_pred_eval)

    fn = str(ts[0]) + "___" + str(ts[-1]) + ".dat"

    pickle.dump( data_tuple, open( _path_pred_eval + "/" +fn ,"wb") )

    return True

def rmse_calc(preds, true):
    #pred #(stochastic_pred_count, upload_batch_size, width, height)
    diff = preds - np.expand_dims( true, axis=0)
    squared = np.square( diff )
    mean = np.mean( squared,axis=0 )
    res = np.sqrt( mean )
    return res.astype(np.float16), np.mean(diff, axis=0).astype(np.float16) #(bs, 156, 352) (bs, 156, 352)

def r20_error_calc(preds, true):
    """R20 - Very heavy wet days ≥ 20mm
       This returns a matrices of differences for the predictions and observed occures of R20 events
    """

    bool_r20_true = tf.where( true >= 20, 1, 0 )
    bool_r20_pred = tf.where( tf.reduce_mean( preds, axis=0) >= 20, 1, 0 )

    return (bool_r20_true-bool_r20_pred).numpy().astype(np.int8)
    
def pixel_calibration(preds, true, dist_name,rainy_threshold=0.5):
    """ 
    We calculate pixel calibration for all days for all poisitions, but include an 1D idxs array showing where the rainy days are 

    returns a list of ratios for each timestep 
    """

    p_range = np.arange(0,1.,0.01)

    distr = distribution_getter( preds, dist_name )

    bool_notrainydays = true<rainy_threshold
    idxs_not_rainydays = np.where( true>=rainy_threshold)[0] #This stores information on which timestep t and position h,w relate to positions of a rainy day from the batch (t,h,w) the metric belongs to

    # true_rainydays = true[idxs_rainydays]
    # np_preds_means_rainydays = np_preds_means[idxs_rainydays]
    # np_preds_stds_rainydays = np_preds_means[idxs_rainydays]
    
    ratios = uq_calibration(true, distr, p_range ) #shape (bs, h, w, 100 )


    #ratios[ np.tile( np.expand_dims( bool_notrainydays, -1 ), [1,1,1,100] ) ] = np.nan
    ratios[ np.tile( np.expand_dims( bool_notrainydays, -1 ), [1,1,1,100] ) ] = 127

    return ratios 

def distribution_getter( preds, dist_name):
    if dist_name=="Normal":
        distr = tfp.distributions.Normal( loc= np.mean(preds,axis=0), scale=np.stds(preds,axis=0) )
    elif dist_name =="LogNormal":
        distr = tfp.distributions.LogNormal( loc=tf.math.reduce_mean( tf.math.log(preds),axis=0 ), scale=tf.math.reduce_std( tf.math.log(preds),axis=0 ) )
    if dist_name =="None":
        distr = tfp.distributions.Normal( loc=np.mean(preds,axis=0), scale=0.05 )
    else:
        raise ValueError
    return distr

def uq_calibration(true, dist, p_range):
    """
        true should be the true values 
        distribution should be the distr based on the preds
        p_range the numbers to eval at
        
        This produces the uq_calibration for each position at each time 
            The uq_calibration is a list of 100 numbers indicating whether or not the true value fell within that confidence interval
            The confidence interval is also indicated by the p_range val
    """

    cdf = dist.cdf(true) # (bs, h, w)
    ratios = np.array([])
    zs = []
    for p_interval in p_range: #In this code for each p_range value he calcaultes the average rate that the values fall in the p_range
        plow = 0.5 - p_interval / 2.
        phigh = 0.5 + p_interval / 2.

        if ratios.size == 0:
            ratios = tf.logical_and( plow < cdf,  cdf < phigh ).numpy()
            ratios = np.expand_dims(ratios, -1)
        else:
            #ratios = np.concatenate( [ ratios, np.expand_dims(tf.logical_and( plow < cdf,  cdf < phigh ),axis=-1)  ], axis=-1 ).astype(np.float16)
            ratios = np.concatenate( [ ratios, np.expand_dims(tf.logical_and( plow < cdf,  cdf < phigh ),axis=-1)  ], axis=-1 ).astype(np.int8)

    return ratios #shape (bs, h, w, 100 )

def postproc_pipeline_compress_evaluations(li_gen_data_eval_preds, test_params, model_params):
    """
        This will operate on batch size amount of predictions
        Each invidual prediction represents an average already
        So This method averages for each position on the map across all timesteps

        New Behaviour: Since we need to calculate standard deviations as well, all data must be held in memory

        pc_binary_in_range: We concatenate all values together from all seperate chunks. Then we calc:
                                the per pixel average over time
                                the overall average over time
        
        Due to Memory Issues Each statistic will be calculated one by one

    """
    # rmse=  None
    # bias = None
    # r20_diffs = None
    # pc_binary_in_range = None

    for idx_stat, gen_data_eval_preds in enumerate(li_gen_data_eval_preds):
        for idx, batch_datum in enumerate( gen_data_eval_preds ):
            
            #true_in_pred_range = np.mean( batch_datum[4], axis=0)
            
            if idx==0:
                if idx_stat in [0,1,2]:
                    val = batch_datum[idx_stat]
                elif idx_stat in [3]:
                    np.save( "temp.npy",batch_datum[idx_stat])
                    prev_bs = batch_datum[idx_stat].shape[0]
            
            else:
                if idx_stat in [0,1,2]:
                    val = np.concatenate( [val, batch_datum[idx_stat]] , axis=0 )
                elif idx_stat in [3]:
                    mem_val = np.memmap("temp.npy", dtype='int8', mode='r+', shape=( prev_bs+batch_datum[idx_stat].shape[0] , batch_datum[idx_stat].shape[1], batch_datum[idx_stat].shape[2], batch_datum[idx_stat].shape[3] ), order='C' )
                    mem_val[ prev_bs:, :, :, :] = batch_datum[idx_stat]
                    prev_bs = mem_val.shape[0]

                # #new method
                # rmse = np.concatenate( [rmse, batch_datum[0]],axis=0 )
                # bias = np.concatenate( [rmse, batch_datum[1] ], axis=0 )

                # r20_diffs = np.concatenate( [r20_diffs, batch_datum[2]] ,axis=0 )

                # pc_binary_in_range = np.concatenate( [pc_binary_in_range, batch_datum[3] ],axis=0 )
        
        if idx_stat == 0:
            rmse_mean = np.mean(val, axis=0) #(h,w)
            rmse_mean_mean = np.mean(rmse_mean) 
            rmse_mean_std = np.std(rmse_mean)
            del val
            gc.collect()
        
        if idx_stat == 1:
            bias_mean =np.mean(val, axis=0)
            bias_mean_mean = np.mean(bias_mean)
            bias_mean_std = np.std(bias_mean)
            del val
            gc.collect()
        
        elif idx_stat ==2:
            r20_mean = np.mean(val,axis=0)
            r20_mean_mean = np.mean(r20_mean)
            r20_mean_std = np.std(r20_mean)
            del val
            gc.collect()

        if idx_stat == 3:
            gc.collect()
            val = np.memmap("temp.npy", dtype='int8', mode='r+', shape=( prev_bs , batch_datum[idx_stat].shape[1], batch_datum[idx_stat].shape[2], batch_datum[idx_stat].shape[3]), order='C' )

            ignore_bool = (val == 127)
            count_bool_perpixel = prev_bs - np.count_nonzero(ignore_bool, axis=0 )
            count_bool_all = np.prod( batch_datum[idx_stat].shape[:3],dtype=np.int64 ) - np.count_nonzero(ignore_bool, axis=(0,1,2) )
            val[ignore_bool] = 0

            # pc_binary_in_range_perpixel_average = np.sum( val, axis=0 )/count_bool_perpixel #(h,w,100)
            # pc_binary_in_range_all_average = np.sum(val, axis=(0,1,2) )/count_bool_all #(100)

            pc_binary_in_range_perpixel_average = np.divide( np.sum( val, axis=0, dtype=np.int32 ), count_bool_perpixel, where=count_bool_perpixel!=0  )#(h,w,100)
            pc_binary_in_range_all_average = np.divide( np.sum( val, axis=(0,1,2), dtype=np.int32 ), count_bool_all, where=count_bool_all!=0 )#(100)

            gc.collect()
            p_range = np.arange(0,1.,0.01)
            pc_binary_rmses_perpixel = np.sqrt( np.mean( (pc_binary_in_range_perpixel_average-p_range)**2  ,axis=-1 ) )    #(h,w)
            pc_binary_rmses_all = np.sqrt( np.mean( (pc_binary_in_range_all_average-p_range)**2  ) )    #(1)
            pc_binary_rmses_all_std = np.nanstd(pc_binary_in_range_all_average-p_range )


    data_tuple = {"rmse-mean": rmse_mean, "bias-mean": bias_mean ,
                    "r20-mean":r20_mean,  "r20-stds":r20_mean_std,
                    "pc-perpixel-rmse":pc_binary_rmses_perpixel, "pc_binary_in_range_all_average":pc_binary_in_range_all_average,
                    "pc-all-rmse":pc_binary_rmses_all, "pc_all_rmse_std":pc_binary_rmses_all_std }

    scores_for_table = {"rmse-mean-mean":float(rmse_mean_mean), "rmse_mean_std":float(rmse_mean_std) , "bias_mean_mean": float(bias_mean_mean) ,"bias_mean_std":float(bias_mean_std),
                            "r20-mean":float(r20_mean_mean),  "r20-std":float(r20_mean_std),
                            "pc-all-rmse":float(pc_binary_rmses_all), "pc-all-rmse_std":float(pc_binary_rmses_all_std)  }
    # latex_table_maker(data_tuple)

    _path_pred_eval_summ = test_params['script_dir'] + "/Output/{}/{}_{}_{}/{}/SummarisedEvaluations".format(model_params['model_name'],  model_params['model_type_settings']['var_model_type'],
                                                        model_params['model_type_settings']['distr_type'],str(model_params['model_type_settings']['discrete_continuous']), model_params['model_version'])
    fn = "summarised_predictions.dat"

    if(not os.path.exists(_path_pred_eval_summ) ):
        os.makedirs(_path_pred_eval_summ)


    pickle.dump( data_tuple, open( _path_pred_eval_summ + "/" +fn ,"wb") )

    with open( _path_pred_eval_summ + "/" +"scores_for_table.json" ,"w") as f:
        json.dump(scores_for_table,f,sort_keys=True, indent=4)

    try:
        os.remove("temp.npy")
    except OSError:
        pass
    
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
    
    #region map based images
    keys_to_keep = ["rmse-mean","bias-mean","pc-perpixel-rmse"]
    dict_vmin_max= { "rmse-mean":[0,6], "bias-mean":[-0.2,0.2], "pc-perpixel-rmse":[0,0.5] }
    dict_summarised = { k: _dict[k] for k in keys_to_keep }

    _path_visual = test_params['script_dir'] + "/Output/{}/{}_{}_{}/{}/Visualisations".format(model_params['model_name'], model_params['model_type_settings']['var_model_type'],
                                                        model_params['model_type_settings']['distr_type'],str(model_params['model_type_settings']['discrete_continuous']), model_params['model_version'])
    if(not os.path.exists(_path_visual) ):
        os.makedirs(_path_visual)

    for key,array in dict_summarised.items():
        plt.imshow( resize( image_usa, array.shape , anti_aliasing=False) )
        plt.imshow( utility.water_mask(array.astype(np.float32),mask=test_params['bool_water_mask'],mode=1) , cmap='bwr',vmin=dict_vmin_max[key][0], vmax=dict_vmin_max[key][1])
        cbar = plt.colorbar()
        cbar.set_label("mm/day")
        fn = _path_visual + "/" + key + ".jpg"
        plt.savefig( fn )
        plt.clf()


        print("Heatmap for ",key," Created")

    # endregion
    p = _dict['pc_binary_in_range_all_average']
    q = np.arange(0.0,1.0,0.01)
    #low = 
    #high =
    rmse = _dict['pc-all-rmse']
    rmse_std = _dict['pc_all_rmse_std']

    fig, axs = plt.subplots(1,1,figsize=(25,11))
    axs = np.ravel(axs)

    axs[0].plot(p, q, color='steelblue', label='RMSE=%0.3f $\pm$ %0.3f' % (rmse, rmse_std))
    axs[0].plot([0,1],[0,1], '--', color='black')
    #plt.plot(p, low)
    #plt.plot(p, high)
    #axs[i].fill_between(p, low, high, alpha=0.5, color='lightsteelblue')#colors[i])
    #axs[i].legend()
    #plt.plot(cals[:,0].T, cals[:,1].T, color='blue', alpha=0.05)
    axs[0].set_title( "{}_{}_{}_{}".format( model_params['model_name'],model_params['model_type_settings']['var_model_type'],
            model_params['model_type_settings']['distr_type'],str(model_params['model_type_settings']['discrete_continuous']) ) , fontsize=28)
    axs[0].set_xlabel("Probability", fontsize=22)
    axs[0].set_ylabel("Frequency", fontsize=22)

    fig.tight_layout()
    plt.savefig(_path_visual+"/uq-calibrations.pdf")
    plt.clf()

    #region calibration plots

    #endregion

if __name__ == "__main__":

    s_dir = utility.get_script_directory(sys.argv[0])
    
    args_dict = utility.parse_arguments(s_dir)

    #stacked DeepSd methodology
    if( args_dict['model_name'] == "DeepSD" ):
        test_params = hparameters.test_hparameters( **args_dict )()

        model_type_settings = ast.literal_eval( args_dict['model_type_settings'] )
        
        model_layers = { 'conv1_param_custom': json.loads(args_dict['conv1_param_custom']) ,
                         'conv2_param_custom': json.loads(args_dict['conv2_param_custom']) }

        del args_dict['model_type_settings']


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
        test_params = hparameters.test_hparameters_ati( **args_dict )()
    
    main(test_params, model_params)

