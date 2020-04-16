import numpy as np
import math
import tensorflow as tf
import sys
import os
import argparse
import json
import hparameters
import ast
import copy
import datetime
import re

# region Vandal
#precip data import - use in data pipeline
def read_prism_precip(bil_path, hdr_path=None, hdr_known=True, tensorf = True):
    """
        Read an array from ESRI BIL raster file using Info from the hdr file too
        https://pymorton.wordpress.com/2016/02/26/plotting-prism-bil-arrays-without-using-gdal/
    """
    if( tensorf ):
        bil_path = bil_path.numpy()

    if(hdr_known):
        NROWS = 621
        NCOLS = 1405
        NODATA_VAL = float(-9999)
    else:
        hdr_dict = read_hdr(hdr_path)
        NROWS = int(hdr_dict['NROWS'])
        NCOLS = int(hdr_dict['NCOLS'])
        NODATA_VAL = hdr_dict['NODATA']
    # For now, only use NROWS, NCOLS, and NODATA
    # Eventually use NBANDS, BYTEORDER, LAYOUT, PIXELTYPE, NBITS
 
    prism_array = np.fromfile(bil_path, dtype='<f4')
    prism_array = prism_array.astype( np.float32 )
    prism_array = prism_array.reshape( NROWS , NCOLS )
    prism_array[ prism_array == float(NODATA_VAL) ] = np.nan
    return prism_array

#prism data import
def read_prism_elevation(bil_path, hdr_path=None, hdr_known=True):
    if(hdr_known):
        NROWS = 6000
        NCOLS = 4800
        NODATA_VAL = float(-9999)
    else:
        hdr_dict = read_hdr(hdr_path)
        NROWS = int(hdr_dict['NROWS'])
        NCOLS = int(hdr_dict['NCOLS'])
        NODATA_VAL = int(hdr_dict['NODATA'])
    # For now, only use NROWS, NCOLS, and NODATA
    # Eventually use NBANDS, BYTEORDER, LAYOUT, PIXELTYPE, NBITS
 
    _array = np.fromfile(open(bil_path,"rb"), dtype='>i2')
    _array = _array.astype(dtype=np.float32)
    _array = _array.reshape( NROWS , NCOLS )
    _array[ _array == NODATA_VAL ] = np.nan
    return _array

def read_hdr(hdr_path):
    """Read an ESRI BIL HDR file"""
    with open(hdr_path, 'r') as input_f:
        header_list = input_f.readlines()
    return dict(item.strip().split() for item in header_list)

# Miscallaneous
def replace_inf_nan(_tensor):
    nan_bool_ind_tf = tf.math.is_nan( tf.dtypes.cast(_tensor,dtype=tf.float32 ) )
    inf_bool_ind_tf = tf.math.is_inf( tf.dtypes.cast( _tensor, dtype=tf.float32 ) )

    bool_ind_tf = tf.math.logical_or( nan_bool_ind_tf, inf_bool_ind_tf )
    
    _tensor = tf.where( bool_ind_tf, tf.constant(0.0,dtype=tf.float32), _tensor )
    return tf.dtypes.cast(_tensor, dtype=tf.float32)

def get_script_directory(_path):
    if(_path==None):
        _path = sys.argv[0]
    _path = os.path.realpath(_path)
    if os.path.isdir(_path):
        return _path
    else:
        return os.path.dirname(_path)

# Methods Related to Training
def update_checkpoints_epoch(df_training_info, epoch, train_metric_mse_mean_epoch, val_metric_mse_mean, ckpt_manager_epoch, train_params, model_params, train_metric_mse=None, val_metric_mse=None  ):
    """
        NOTE: Val_metric_mse_mean and train_metric_mse_mean_epoch; may not be mse as they are dependent on the loss function
        so for models that do not use mse, train_metric_mse, and val_metric_mse shold be passed values
    """
    df_training_info = df_training_info[ df_training_info['Epoch'] != epoch ] #rmv current batch records for compatability with code below
    
    if( ( val_metric_mse_mean.result().numpy() < min( df_training_info.loc[ : ,'Val_loss_MSE' ], default= val_metric_mse_mean.result().numpy()+1 ) ) ):
        #NOTE: To prevent model overffitng and a small scale term being approached, we changed this to be at least 

        print('Saving Checkpoint for epoch {}'.format(epoch)) 
        ckpt_save_path = ckpt_manager_epoch.save()

        
        # Possibly removing old non top5 records from end of epoch
        if( len(df_training_info.index) >= train_params['checkpoints_to_keep_epoch'] ):
            df_training_info = df_training_info.sort_values(by=['Val_loss_MSE'], ascending=True)
            df_training_info = df_training_info.iloc[:-1]
            df_training_info.reset_index(drop=True)

        
        if train_metric_mse==None:
            df_training_info = df_training_info.append( other={ 'Epoch':epoch,'Train_loss_MSE':train_metric_mse_mean_epoch.result().numpy(), 'Val_loss_MSE':val_metric_mse_mean.result().numpy(),
                                                            'Checkpoint_Path': ckpt_save_path, 'Last_Trained_Batch':-1}, ignore_index=True ) #A Train batch of -1 represents final batch of training step was completed
        else:
            df_training_info = df_training_info.append( other={ 'Epoch':epoch,'Train_loss_MSE':train_metric_mse_mean_epoch.result().numpy(), 'Val_loss_MSE':val_metric_mse_mean.result().numpy(),
                                                            'Checkpoint_Path': ckpt_save_path, 'Last_Trained_Batch':-1,
                                                            'Train_metric_mse':train_metric_mse.result().numpy(), 'Validation_metric_mse':val_metric_mse.result().numpy()
                                                            }, ignore_index=True ) #A Train batch of -1 represents final batch of training step was completed
            

        print("\nTop {} Performance Scores".format(train_params['checkpoints_to_keep_epoch']))
        df_training_info = df_training_info.sort_values(by=['Val_loss_MSE'], ascending=True)[:train_params['checkpoints_to_keep_epoch']]
        if train_metric_mse==None:
            print(df_training_info[['Epoch','Val_loss_MSE']] )
        else:
            print(df_training_info[['Epoch','Val_loss_MSE','Validation_metric_mse','Train_metric_mse']] )
        df_training_info.to_csv( path_or_buf="checkpoints/{}/checkpoint_scores.csv".format(model_name_mkr(model_params)),
                                    header=True, index=False ) #saving df of scores                      
    return df_training_info

def kl_loss_weighting_scheme( max_batch, curr_batch, var_model_type="dropout" ):

    if var_model_type in ["flipout"]:
        idx = max_batch-curr_batch+1
        weight = 1/(2**idx)
    else:
        weight = (1/max_batch)
        
    return weight*(1/10)

#standardising and de-standardizing 
def standardize( _array, reverse=False, distr_type="Normal" ):
    if distr_type=="Normal":
        SCALE = 2
    elif distr_type=="LogNormal":
        SCALE= 2

    if(reverse==False):
        _array = _array/SCALE
    
    else:
        _array = _array*SCALE
    
    return _array

#masking water
def water_mask(array, mask, mode=0):
    #need a mask for mse calculation, train and validation -> in this one just cast them to 0,
    
    if mode==0:
        array=tf.where( mask, array, 0.0 )

    #need a mask for image printing summary
    elif mode ==1:
        array = tf.where(mask, array, np.nan )
    return array

# endregion

# region passing arguments to script
def load_params_train_model(args_dict):
        
    if( args_dict['model_name'] == "DeepSD" ):
                      
        init_params = {}
        init_params.update({"input_dims": [39, 88 ], "output_dims": [ 156, 352 ] })
        init_params.update({'model_type_settings': ast.literal_eval( args_dict['model_type_settings'] )})
        init_params.update({ 'conv1_param_custom': json.loads(args_dict['conv1_param_custom']) ,
                         'conv2_param_custom': json.loads(args_dict['conv2_param_custom']) })

        model_params = hparameters.model_deepsd_hparameters(**init_params)()
        del args_dict['model_type_settings']
        train_params = hparameters.train_hparameters( **args_dict )
    
    elif(args_dict['model_name'] == "THST"):
        
        init_m_params = {}
        init_m_params.update( {'model_type_settings': ast.literal_eval( args_dict.pop('model_type_settings') ) } )
        model_params = hparameters.model_THST_hparameters( **init_m_params, **args_dict )()
        init_t_params = {}
        init_t_params.update( { 'lookback_target': model_params['data_pipeline_params']['lookback_target'] } )
        init_t_params.update( { 'lookback_feature': model_params['data_pipeline_params']['lookback_feature']})
        train_params = hparameters.train_hparameters_ati( **{ **args_dict, **init_t_params} )
    
    elif(args_dict['model_name'] in ["SimpleGRU"] ):
        
        init_m_params = {}
        init_m_params.update({'model_type_settings': ast.literal_eval( args_dict.pop('model_type_settings') ) } )
        model_params = hparameters.model_SimpleGRU_hparameters(**init_m_params, **args_dict)()
        init_t_params = {}
        init_t_params.update( { 'lookback_target': model_params['data_pipeline_params']['lookback_target'] } )
        init_t_params.update( { 'lookback_feature': model_params['data_pipeline_params']['lookback_feature']})
        init_t_params.update( {'loss_scales':ast.literal_eval( args_dict['loss_scales']) } )
        train_params = hparameters.train_hparameters_ati( **{ **args_dict, **init_t_params} )
    
    elif(args_dict['model_name'] in ["SimpleDense"] ):
        
        init_m_params = {}
        init_m_params.update({'model_type_settings': ast.literal_eval( args_dict.pop('model_type_settings') ) } )
        model_params = hparameters.model_SimpleDense_hparameters(**init_m_params)()
        init_t_params = {}
        init_t_params.update( { 'lookback_target': model_params['data_pipeline_params']['lookback_target'] } )
        init_t_params.update( { 'lookback_feature': model_params['data_pipeline_params']['lookback_feature']})
        
        train_params = hparameters.train_hparameters_ati( **{ **args_dict, **init_t_params} )

    elif(args_dict['model_name']=="SimpleConvGRU"):
        init_m_params = {}
        init_m_params.update({'model_type_settings': ast.literal_eval( args_dict.pop('model_type_settings') ) } )
        model_params = hparameters.model_SimpleConvGRU_hparamaters(**init_m_params, **args_dict)()
        init_t_params = {}
        init_t_params.update( { 'lookback_target': model_params['data_pipeline_params']['lookback_target'] } )
        init_t_params.update( { 'lookback_feature': model_params['data_pipeline_params']['lookback_feature']})
        
        train_params = hparameters.train_hparameters_ati( **{ **args_dict, **init_t_params} )

    #Other Checks
    ## 1) If training and if mc_dropout, change to Deterministic since mc_dropout is not really variational inference
    # if( model_params['model_type_settings']['var_model_type']=='mc_dropout' and model_params['model_type_settings']['stochastic']==False ):
    #     model_params['model_type_settings']['var_model_type'] ='Deterministic'

    save_model_settings( model_params, train_params() )

    return train_params, model_params

def load_params_test_model(args_dict):
    if( args_dict['model_name'] == "DeepSD" ):
                      
        init_params = {}
        init_params.update({"input_dims": [39, 88 ], "output_dims": [ 156, 352 ] })
        init_params.update({'model_type_settings': ast.literal_eval( args_dict['model_type_settings'] )})
        init_params.update({ 'conv1_param_custom': json.loads(args_dict['conv1_param_custom']) ,
                         'conv2_param_custom': json.loads(args_dict['conv2_param_custom']) })

        model_params = hparameters.model_deepsd_hparameters(**init_params)()
        del args_dict['model_type_settings']
        train_params = hparameters.test_hparameters( **args_dict )
    
    elif(args_dict['model_name'] == "THST"):
        
        init_m_params = {}
        init_m_params.update({'model_type_settings': ast.literal_eval( args_dict.pop('model_type_settings') ) } )
        model_params = hparameters.model_THST_hparameters(**init_m_params, **args_dict )()
        init_t_params = {}
        init_t_params.update( { 'lookback_target': model_params['data_pipeline_params']['lookback_target'] } )
        init_t_params.update( { 'lookback_feature': model_params['data_pipeline_params']['lookback_feature']})
        train_params = hparameters.test_hparameters_ati( **{ **args_dict, **init_t_params} )
    
    elif(args_dict['model_name'] == "SimpleGRU"):
        
        init_m_params = {}
        init_m_params.update({'model_type_settings': ast.literal_eval( args_dict.pop('model_type_settings') ) } )
        model_params = hparameters.model_SimpleGRU_hparameters(**init_m_params, **args_dict)()
        init_t_params = {}
        init_t_params.update( { 'lookback_target': model_params['data_pipeline_params']['lookback_target'] } )
        init_t_params.update( { 'lookback_feature': model_params['data_pipeline_params']['lookback_feature']})
        
        train_params = hparameters.test_hparameters_ati( **{ **args_dict, **init_t_params} )
    elif(args_dict['model_name'] == "SimpleDense"):
        
        init_m_params = {}
        init_m_params.update({'model_type_settings': ast.literal_eval( args_dict.pop('model_type_settings') ) } )
        model_params = hparameters.model_SimpleDense_hparameters(**init_m_params, **args_dict)()
        init_t_params = {}
        init_t_params.update( { 'lookback_target': model_params['data_pipeline_params']['lookback_target'] } )
        init_t_params.update( { 'lookback_feature': model_params['data_pipeline_params']['lookback_feature']})
        
        train_params = hparameters.test_hparameters_ati( **{ **args_dict, **init_t_params} )
    

    elif(args_dict['model_name']=="SimpleConvGRU"):
        init_m_params = {}
        init_m_params.update({'model_type_settings': ast.literal_eval( args_dict.pop('model_type_settings') ) } )
        model_params = hparameters.model_SimpleConvGRU_hparamaters(**init_m_params, **args_dict)()
        init_t_params = {}
        init_t_params.update( { 'lookback_target': model_params['data_pipeline_params']['lookback_target'] } )
        init_t_params.update( { 'lookback_feature': model_params['data_pipeline_params']['lookback_feature']})
        
        train_params = hparameters.test_hparameters_ati( **{ **args_dict, **init_t_params} )

    save_model_settings( model_params, train_params() )

    return train_params, model_params

def parse_arguments(s_dir=None):
    parser = argparse.ArgumentParser(description="Receive input params")

    parser.add_argument('-dd','--data_dir', type=str, help='the directory for the Data', required=False,
                        default='./Data')
    
    parser.add_argument('-mts','--model_type_settings', type=str, help="dictionary Defining type of model to use", required=True)

    parser.add_argument('-sdr','--script_dir', type=str, help="Directory for code", required=False, default=s_dir )

    parser.add_argument('-mn','--model_name', type=str, help='Name of model to use', required=False, default="DeepSD")                                      
        
    parser.add_argument('-ds','--distribution_strategy', type=str, help='The distribution strategy to be used by tensorflow', required=False, default="None" ) #TODO: Implement ability to train on multiple cores tensorflow

    parser.add_argument('-c1pc', '--conv1_param_custom', type=str, required=False, default='{}' )

    parser.add_argument('-c2pc', '--conv2_param_custom', type=str, required=False, default='{}')

    parser.add_argument('-opt','--optimizer_settings', type=str, required=False, default='{}')

    parser.add_argument('-bs','--batch_size', type=int, required=False, default=2)

    parser.add_argument('-od','--output_dir', type=str, required=False, default="./Output")

    parser.add_argument('-sdc','--strided_dataset_count',type=int, required=False, default=1, help="The number of datasets to create. Each dataset has stride equal to window size, so for large window sizes dataset becomes very small and overfitting is likely")
    
    parser.add_argument('-ls','--loss_scales',type=str, required=False, default="{}",help="The custom weighting to add to different parts of the loss function" )

    parser.add_argument('-do','--dropout',type=float, required=False, default=0.0)

    parser.add_argument('-ido','--inp_dropout',type=float, required=False, default=0.0)

    parser.add_argument('-rdo','--rec_dropout',type=float, required=False, default=0.0)

    parser.add_argument('-dif','--downscale_input_factor',type=int, required=False )
    
    args_dict = vars(parser.parse_args() )

    return args_dict

def save_model_settings(model_params,t_params):
    
    f_dir = "model_params/{}".format( model_name_mkr(model_params) )

    m_path = "model_params.json"
    
    if t_params['trainable']==True:
        t_path = "train_params.json"
    else:
        t_path = "test_params.json"

    if not os.path.isdir(f_dir):
        os.makedirs( f_dir, exist_ok=True  )
    with open( f_dir+"/"+m_path, "w" ) as fp:
        json.dump( model_params, fp, default=default )

    with open( f_dir+"/"+t_path, "w" ) as fp:
        json.dump( t_params, fp, default=default )

def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    elif type(obj).__module__ == datetime.__name__: 
        if isinstance(obj, datetime.date):
            return obj.isoformat()
        else:
            return obj.__str__()

    elif isinstance( obj, tf.keras.regularizers.Regularizer ):
        return obj.get_config() 
    
    elif isinstance(obj, tf.keras.layers.Layer):
        return obj.get_config()

    raise TypeError('Unknown type:', type(obj))

#endregion

def model_name_mkr(model_params, mode='Generic', load_save="load" ) : #change ordering of variables ehre
    if mode == "Generic":
        pass
    
    elif mode == "mc_dropout_test":
        model_params = copy.deepcopy(model_params)
        model_params['model_type_settings']['var_model_type'] = 'Deterministic'
        model_params['model_type_settings']['stochastic'] = False

    
    if  model_params['model_name'] == "THST":
        if model_params['model_type_settings']['deformable_conv'] == False:
            model_name = "{}_{}_{}_{}_{}_v{}".format( model_params['model_name'], model_params['model_type_settings']['var_model_type'],
                                            model_params['model_type_settings']['distr_type'], 
                                            str(model_params['model_type_settings']['discrete_continuous']),
                                            model_params['model_type_settings']['location'], model_params['model_type_settings']['model_version']   )
        else:
            model_name = "{}_{}_{}_{}_{}_v{}_dc".format( model_params['model_name'], model_params['model_type_settings']['var_model_type'],
                                model_params['model_type_settings']['distr_type'], 
                                str(model_params['model_type_settings']['discrete_continuous']),
                                model_params['model_type_settings']['location'], model_params['model_type_settings']['model_version']  )

    elif model_params['model_name'] == "DeepSD":
        model_name =    "{}_{}_{}_{}_v{}".format( model_params['model_name'], model_params['model_type_settings']['var_model_type'],
                                model_params['model_type_settings']['distr_type'], 
                                str(model_params['model_type_settings']['discrete_continuous']),
                                model_params['model_type_settings']['model_version'] )
   
    elif model_params['model_name'] in ["SimpleGRU","SimpleDense"]:
        model_name =    "{}_{}_{}_{}_{}_v{}".format( model_params['model_name'], model_params['model_type_settings']['var_model_type'],
                                model_params['model_type_settings']['distr_type'], 
                                str(model_params['model_type_settings']['discrete_continuous']),
                                model_params['model_type_settings']['location'],model_params['model_type_settings']['model_version'] )
   
    elif model_params['model_name'] == "SimpleConvGRU":
        model_name =  "{}_{}_{}_{}_{}_v{}".format( model_params['model_name'], model_params['model_type_settings']['var_model_type'],
                        model_params['model_type_settings']['distr_type'], 
                        str(model_params['model_type_settings']['discrete_continuous']),
                        model_params['model_type_settings']['location'],model_params['model_type_settings']['model_version'] )  
    
    if 'downscale_input_factor' in model_params:
        model_name  =  model_name + "dsf{}".format( model_params['downscale_input_factor'])
    
    if load_save == "save" and model_params['model_type_settings'].get('location_test',"London") != "London":
        model_name = model_name + model_params['model_type_settings']['location_test']

    model_name = re.sub("[ '\(\[\)\]]|ListWrapper",'',model_name )

    model_name = re.sub(",",'_',model_name )

    return model_name

# region ATI modules
def standardize_ati(_array, shift, scale, reverse):
    
    if(reverse==False):
        _array = (_array-shift)/scale
    elif(reverse==True):
        _array = (_array*scale)+shift
    
    return _array

# endregion
