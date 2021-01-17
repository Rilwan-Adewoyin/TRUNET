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
import pickle
from tensorflow.python.ops import variables as tf_variables


# region - Reporting
def update_checkpoints_epoch(df_training_info, epoch, train_loss_epoch, val_loss_epoch, ckpt_manager_epoch, t_params, m_params, train_metric_mse=None,
                                val_metric_mse=None,  objective="mse"  ):
    """Updates the checkpoint and epoch records associated with an instance of training

        Args:
            df_training_info (DataFrame): contains information on the scores achieved on each epoch/batch 
            epoch (int): current epoch
            train_loss_epoch (tf.keras.loss.Mean): aggregated loss from training batches within epoch
            val_loss_epoch (tf.keras.metric.Mean): aggregated loss from validation batches within epoch
            ckpt_manager_epoch (tf.train.CheckpointManager): ckpt manager for epoch 
            t_params (dict): params related to training/testing
            m_params (dict): params related to model
            train_metric_mse (tf.keras.metric.Mean): aggregated mse from train batches within epoch
            val_metric_mse (tf.keras.metric.Mean): aggregated mse from validation batches within epoch

        Returns:
            dictionary: Updated df_training_info
    """    
    # rm any pre-existing information from current epoch
    df_training_info = df_training_info[ df_training_info['Epoch'] != epoch ] 
    
    minimized = ( val_loss_epoch.result().numpy() <= min( df_training_info.loc[ : ,'Val_loss' ], default= val_loss_epoch.result().numpy()+1 ) )

    # if new val_los_epoch is less than existing Val_loss then update, else return unedited df_training info
    if( minimized  ):

        print('Saving Checkpoint for epoch {}'.format(epoch)) 
        ckpt_save_path = ckpt_manager_epoch.save()

        
        # Possibly removing old non top5 records from end of epoch
        if( len(df_training_info.index) >= t_params['checkpoints_to_keep'] ):
            
            df_training_info = df_training_info.sort_values(by=['Val_loss'], ascending=True)


            df_training_info = df_training_info.iloc[:-1]
            df_training_info.reset_index(drop=True)

    
        df_training_info = df_training_info.append( 
            other={ 'Epoch':epoch,'Train_loss':train_loss_epoch.result().numpy(), 'Train_mse':train_metric_mse.result().numpy(),
                'Val_loss':val_loss_epoch.result().numpy(),'Val_mse':val_metric_mse.result().numpy(),
                'Checkpoint_Path': ckpt_save_path, 'Last_Trained_Batch':-1
                }, ignore_index=True ) #A Train batch of -1 represents final batch of training step was completed


        print("\nTop {} Performance Scores".format(t_params['checkpoints_to_keep']))
        
        
        df_training_info = df_training_info.sort_values(by=['Val_loss'], ascending=True)[:t_params['checkpoints_to_keep']]

        print(df_training_info[['Epoch','Train_loss','Train_mse','Val_loss','Val_mse']] )

        df_training_info.to_csv( path_or_buf="checkpoints/{}/checkpoint_scores.csv".format(model_name_mkr(m_params, t_params=t_params,  htuning=m_params.get('htuning',False)),
                                    header=True, index=False) ) #saving df of scores                      
    
    return df_training_info

def tensorboard_record(writer, li_metrics, li_names, step, gradients=None, trainable_variables=None):
    """
        Updates tensorboard records

        Args:
            writer (tf.summary.Filewriter): tensorflow filewriter
            li_metrics (list{tf.keras.metric.Mean}): [description]
            li_names (list{str}): names of metrics for tensboard
            step (int): 
            gradients (list{float32}): Gradients of model at step. Defaults to None.
            trainable_variables (list{str}): names of weights associated with gradients. Defaults to None.
    """    
    with writer:
        for metric, name in zip( li_metrics, li_names) :
            tf.summary.scalar( name, metric , step =  step )
        
        if gradients != None:
            for grad, _tensor in zip( gradients, trainable_variables):
                if grad is not None:
                    tf.summary.histogram( "Grad:{}".format( _tensor.name ) , grad, step = step  )
                    tf.summary.histogram( "Weights:{}".format(_tensor.name), _tensor , step = step ) 
# endregion

# region - Loading params
def get_script_directory(_path):
    if(_path==None):
        _path = sys.argv[0]
    _path = os.path.realpath(_path)
    if os.path.isdir(_path):
        return _path
    else:
        return os.path.dirname(_path)

def load_params(args_dict, train_test="train"):
    """Returns t_params and m_params for specific models
    
        Returns:
            [tuple(dict,dict)]: t_params and m_params
    """    
    init_m_params = {}
    init_t_params = {}

    init_m_params.update( {'model_type_settings': ast.literal_eval( args_dict.pop('model_type_settings') ) } )
    init_t_params.update( {'t_settings': ast.literal_eval( args_dict.pop('t_settings') ) } )

    if(args_dict['model_name'] == "TRUNET"):
        m_params = hparameters.model_TRUNET_hparameters( **init_m_params, **args_dict )()
        init_t_params.update( { 'lookback_target': m_params['data_pipeline_params']['lookback_target'] } )
        init_t_params.update( { 'lookback_feature': m_params['data_pipeline_params']['lookback_feature']})

    elif(args_dict['model_name']=="SimpleConvGRU"):
        m_params = hparameters.model_SimpleConvGRU_hparamaters(**init_m_params, **args_dict)()
        init_t_params.update( { 'lookback_target': m_params['data_pipeline_params']['lookback_target'] } )
        init_t_params.update( { 'lookback_feature': m_params['data_pipeline_params']['lookback_feature']})

    elif(args_dict['model_name']=="UNET"):
        m_params = hparameters.model_UNET_hparamaters(**init_m_params, **args_dict)()
        init_t_params.update( { 'lookback_target': 1 } )
        init_t_params.update( { 'lookback_feature': 4 })

    if train_test == "train":
        t_params = hparameters.train_hparameters_ati( **{ **args_dict, **init_t_params} )
    else:
        t_params = hparameters.test_hparameters_ati( **{ **args_dict, **init_t_params} )
    
    #if train_test == "train":
    save_model_settings( m_params, t_params() )

    return t_params(), m_params

def parse_arguments(s_dir=None):
    """ Set up argument parser"""
    parser = argparse.ArgumentParser(description="Receive input params")

    parser.add_argument('-dd','--data_dir', type=str, help='the directory for the Data', required=False,
                        default='./Data')
    
    parser.add_argument('-mts','--model_type_settings', type=str, help="m_params", required=True, default={})

    parser.add_argument('-ts','--t_settings',type=str, help="dictioary of custom settings for training/testing", required=False, default='{}')

    parser.add_argument('-mprm','--m_params', type=str, help="m_params", required=False, default=argparse.SUPPRESS )

    parser.add_argument('-tprm','--t_params', type=str, help="t_params", required=False, default=argparse.SUPPRESS )

    parser.add_argument('-sdr','--script_dir', type=str, help="Directory for code", required=False, default=s_dir )

    parser.add_argument('-mn','--model_name', type=str, help='Name of model to use', required=False, default="TRUNET")                                      
        
    parser.add_argument('-bs','--batch_size', type=int, required=False, default=5)

    parser.add_argument('-od','--output_dir', type=str, required=False, default="./Output")
    
    parser.add_argument('-ctsm','--ctsm', type=str, required=True, default="1979_1982_1983_1984", help="how to split dataset for training and validation") 

    parser.add_argument('-ctsm_test','--ctsm_test', type=str, required=False, default=argparse.SUPPRESS, help="dataset for testing") 

    parser.add_argument('-pc','--parallel_calls', type=int, required=False)

    parser.add_argument('-ep','--epochs', default=100, type=int, required=False)
       
    args_dict = vars(parser.parse_args() )

    return args_dict


#endregion

# region - Saving model / settings / params
def save_model_settings(m_params,t_params):
    """Saves the m_params and t_params dicts to file

    """    
    f_dir = "saved_params/{}".format( model_name_mkr(m_params, t_params=t_params, htuning=m_params.get('htuning',False)) )

    
    
    if t_params['trainable']==True:
        t_path = "train_params.json"
        m_path = "m_params.json"
    else:
        t_path = "test_params.json"
        m_path = "test_m_params.json"

    if not os.path.isdir(f_dir):
        os.makedirs( f_dir, exist_ok=True  )
    with open( f_dir+"/"+m_path, "w" ) as fp:
        json.dump( m_params, fp, default=default_pkl )

    with open( f_dir+"/"+t_path, "w" ) as fp:
        json.dump( t_params, fp, default=default_pkl )

def default_pkl(obj):
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

def model_name_mkr(m_params, train_test="train", t_params={}, custom_test_loc=None, htuning=False ) : 
    """Creates file names for models based on the variants used to train them

        Args:
            m_params (dict): params for model
            train_test (str, optional): Defaults to "train".
            t_params (dict, optional): params for training/testin. Defaults to {}.
            custom_test_loc ([type], optional): custom location for testing, opposing that found in t_param['location_test']. Defaults to None.

        Returns:
            [type]: [description]
    """    
     
    model_name = "{}_{}_{}_{}_{}".format( m_params['model_name'], m_params['model_type_settings'].get('var_model_type',''),
                        m_params['model_type_settings'].get('distr_type',"Normal"), 
                        str(m_params['model_type_settings']['discrete_continuous']),
                        "_".join(loc_name_shrtner(m_params['model_type_settings']['location']) ) )


    # if m_params['model_type_settings'].get('conv_ops_qk',False) == True:
    #     model_name = model_name + "convopsqk"


    if train_test=="train": 
        model_name = model_name + "_" + str( t_params['ctsm'] )

    if train_test == "test":
        if custom_test_loc != None:
            pass

        elif m_params.get('location_test', None) != None:
            custom_test_loc = m_params.get('location_test')

        elif m_params.get('location', None) == None:
            custom_test_loc = m_params.get('location')

        model_name = model_name +"_" + '_'.join(custom_test_loc)

        model_name = model_name + "_train" + str( t_params['ctsm'] ) +"_test" + str( t_params['ctsm_test'] )

    
    # Addons
    
    if m_params['model_type_settings'].get('attn_ablation',0) != 0:
        model_name = model_name + "_ablation" + str(m_params['model_type_settings']['attn_ablation'])

    if m_params['model_type_settings'].get('heads',8) != 8:
        model_name = model_name + "_heads_{}".format( str(m_params['model_type_settings']['heads']) )
    
    if htuning==True:
        model_name = model_name + f"_htune_v{m_params['htune_version']:03d}"
    
    model_name = re.sub(",",'_',model_name )

    return model_name

def loc_name_shrtner(li_locs):
    li_locs = [ name[:3] for name in li_locs]
    return li_locs

def cache_suffix_mkr(m_params, t_params):
    """Creates the cache suffix for training datasets

    Args:
        m_params ([type]): [description]
        t_params ([type]): [description]

    Returns:
        [type]: [description]
    """        
    if t_params['ctsm'] == "4ds_10years":
        cache_suffix ='_{}_bs_{}_fyitrain_{}_loc_{}'.format( m_params['model_name'], t_params['model_name'], str(t_params['fyi_train']), loc_name_shrtner(m_params['model_type_settings']['location'])  )
        
    else: 
        cache_suffix = '_{}_bs_{}_{}_{}'.format(m_params['model_name'], t_params['batch_size'],
                            loc_name_shrtner(m_params['model_type_settings']['location']),
                            m_params['ctsm']  )    
        
    return cache_suffix

def location_getter(model_settings):

    if model_settings.get('location_test', None) == None:
        # If training use train locations
        # If testing but no location_test passed, used training location
        li_loc = model_settings['location']
    else:
        # Use test location specified
        li_loc = model_settings.get('location_test')
    return li_loc
#endregion

# region data standardization

#@tf.function( experimental_relax_shapes=True )
def standardize_ati(_array, shift, scale, reverse):
    
    if(reverse==False):
        _array = (_array-shift)/scale
    elif(reverse==True):
        _array = (_array*scale)+shift
    
    return _array

# endregion

