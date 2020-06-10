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


def get_script_directory(_path):
    if(_path==None):
        _path = sys.argv[0]
    _path = os.path.realpath(_path)
    if os.path.isdir(_path):
        return _path
    else:
        return os.path.dirname(_path)


# region - Reporting
def update_checkpoints_epoch(df_training_info, epoch, train_metric_mse_mean_epoch, val_metric_mse_mean, ckpt_manager_epoch, t_params, m_params, train_metric_mse=None, val_metric_mse=None  ):
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
        if( len(df_training_info.index) >= t_params['checkpoints_to_keep_epoch'] ):
            df_training_info = df_training_info.sort_values(by=['Val_loss_MSE'], ascending=True)
            df_training_info = df_training_info.iloc[:-1]
            df_training_info.reset_index(drop=True)

        
        if train_metric_mse==None:
            df_training_info = df_training_info.append( other={ 'Epoch':epoch,'Train_loss':train_metric_mse_mean_epoch.result().numpy(), 'Val_loss':val_metric_mse_mean.result().numpy(),
                                                            'Checkpoint_Path': ckpt_save_path, 'Last_Trained_Batch':-1}, ignore_index=True ) #A Train batch of -1 represents final batch of training step was completed
        else:
            df_training_info = df_training_info.append( other={ 'Epoch':epoch,'Train_loss':train_metric_mse_mean_epoch.result().numpy(), 'Val_loss':val_metric_mse_mean.result().numpy(),
                                                            'Checkpoint_Path': ckpt_save_path, 'Last_Trained_Batch':-1,
                                                            'Train_mse':train_metric_mse.result().numpy(), 'Val_mse':val_metric_mse.result().numpy()
                                                            }, ignore_index=True ) #A Train batch of -1 represents final batch of training step was completed
            

        print("\nTop {} Performance Scores".format(t_params['checkpoints_to_keep_epoch']))
        df_training_info = df_training_info.sort_values(by=['Val_loss'], ascending=True)[:t_params['checkpoints_to_keep_epoch']]
        if train_metric_mse==None:
            print(df_training_info[['Epoch','Val_loss']] )
        else:
            print(df_training_info[['Epoch','Val_loss','Val_mse','Train_loss','Train_mse']] )
        df_training_info.to_csv( path_or_buf="checkpoints/{}/checkpoint_scores.csv".format(model_name_mkr(m_params, t_params=t_params)),
                                    header=True, index=False ) #saving df of scores                      
    return df_training_info

def tensorboard_record(writer, li_metrics, li_names, step, gradients=None, trainable_variables=None):
    
    #with writer.as_default():
    with writer:
        for name, metric in zip( li_metrics, li_names) :
            tf.summary.scalar( name, metric , step =  step )
        
        if gradients != None:
            for grad, _tensor in zip( gradients, trainable_variables):
                if grad is not None:
                    tf.summary.histogram( "Grad:{}".format( _tensor.name ) , grad, step = step  )
                    tf.summary.histogram( "Weights:{}".format(_tensor.name), _tensor , step = step ) 
# endregion

# region - loading params
def load_params_train_model(args_dict):
        
    if( args_dict['model_name'] == "DeepSD" ):
                      
        init_params = {}
        init_params.update({"input_dims": [39, 88 ], "output_dims": [ 156, 352 ] })
        init_params.update({'model_type_settings': ast.literal_eval( args_dict['model_type_settings'] )})
        init_params.update({ 'conv1_param_custom': json.loads(args_dict['conv1_param_custom']) ,
                         'conv2_param_custom': json.loads(args_dict['conv2_param_custom']) })

        m_params = hparameters.model_deepsd_hparameters(**init_params)()
        del args_dict['model_type_settings']
        t_params = hparameters.train_hparameters( **args_dict )
    
    elif(args_dict['model_name'] == "THST"):
        
        init_m_params = {}
        init_m_params.update( {'model_type_settings': ast.literal_eval( args_dict.pop('model_type_settings') ) } )
        m_params = hparameters.model_THST_hparameters( **init_m_params, **args_dict )()
        init_t_params = {}
        init_t_params.update( { 'lookback_target': m_params['data_pipeline_params']['lookback_target'] } )
        init_t_params.update( { 'lookback_feature': m_params['data_pipeline_params']['lookback_feature']})
        t_params = hparameters.train_hparameters_ati( **{ **args_dict, **init_t_params} )
    
    elif(args_dict['model_name'] in ["SimpleGRU"] ):
        
        init_m_params = {}
        init_m_params.update({'model_type_settings': ast.literal_eval( args_dict.pop('model_type_settings') ) } )
        m_params = hparameters.model_SimpleGRU_hparameters(**init_m_params, **args_dict)()
        init_t_params = {}
        init_t_params.update( { 'lookback_target': m_params['data_pipeline_params']['lookback_target'] } )
        init_t_params.update( { 'lookback_feature': m_params['data_pipeline_params']['lookback_feature']})
        init_t_params.update( {'loss_scales':ast.literal_eval( args_dict['loss_scales']) } )
        t_params = hparameters.train_hparameters_ati( **{ **args_dict, **init_t_params} )
    
    elif(args_dict['model_name'] in ["SimpleDense"] ):
        
        init_m_params = {}
        init_m_params.update({'model_type_settings': ast.literal_eval( args_dict.pop('model_type_settings') ) } )
        m_params = hparameters.model_SimpleDense_hparameters(**init_m_params)()
        init_t_params = {}
        init_t_params.update( { 'lookback_target': m_params['data_pipeline_params']['lookback_target'] } )
        init_t_params.update( { 'lookback_feature': m_params['data_pipeline_params']['lookback_feature']})
        
        t_params = hparameters.train_hparameters_ati( **{ **args_dict, **init_t_params} )

    elif(args_dict['model_name']=="SimpleConvGRU"):
        init_m_params = {}
        init_m_params.update({'model_type_settings': ast.literal_eval( args_dict.pop('model_type_settings') ) } )
        m_params = hparameters.model_SimpleConvGRU_hparamaters(**init_m_params, **args_dict)()
        init_t_params = {}
        init_t_params.update( { 'lookback_target': m_params['data_pipeline_params']['lookback_target'] } )
        init_t_params.update( { 'lookback_feature': m_params['data_pipeline_params']['lookback_feature']})
        
        t_params = hparameters.train_hparameters_ati( **{ **args_dict, **init_t_params} )

    #Other Checks
    ## 1) If training and if mc_dropout, change to Deterministic since mc_dropout is not really variational inference
    # if( m_params['model_type_settings']['var_model_type']=='mc_dropout' and m_params['model_type_settings']['stochastic']==False ):
    #     m_params['model_type_settings']['var_model_type'] ='Deterministic'

    save_model_settings( m_params, t_params() )

    return t_params(), m_params

def load_params_test_model(args_dict):
    if( args_dict['model_name'] == "DeepSD" ):
                      
        init_params = {}
        init_params.update({"input_dims": [39, 88 ], "output_dims": [ 156, 352 ] })
        init_params.update({'model_type_settings': ast.literal_eval( args_dict['model_type_settings'] )})
        init_params.update({ 'conv1_param_custom': json.loads(args_dict['conv1_param_custom']) ,
                         'conv2_param_custom': json.loads(args_dict['conv2_param_custom']) })

        m_params = hparameters.model_deepsd_hparameters(**init_params)()
        del args_dict['model_type_settings']
        t_params = hparameters.test_hparameters( **args_dict )
    
    elif(args_dict['model_name'] == "THST"):
        
        init_m_params = {}
        init_m_params.update({'model_type_settings': ast.literal_eval( args_dict.pop('model_type_settings') ) } )
        m_params = hparameters.model_THST_hparameters(**init_m_params, **args_dict )()
        init_t_params = {}
        init_t_params.update( { 'lookback_target': m_params['data_pipeline_params']['lookback_target'] } )
        init_t_params.update( { 'lookback_feature': m_params['data_pipeline_params']['lookback_feature']})
        t_params = hparameters.test_hparameters_ati( **{ **args_dict, **init_t_params} )
    
    elif(args_dict['model_name'] == "SimpleGRU"):
        
        init_m_params = {}
        init_m_params.update({'model_type_settings': ast.literal_eval( args_dict.pop('model_type_settings') ) } )
        m_params = hparameters.model_SimpleGRU_hparameters(**init_m_params, **args_dict)()
        init_t_params = {}
        init_t_params.update( { 'lookback_target': m_params['data_pipeline_params']['lookback_target'] } )
        init_t_params.update( { 'lookback_feature': m_params['data_pipeline_params']['lookback_feature']})
        
        t_params = hparameters.test_hparameters_ati( **{ **args_dict, **init_t_params} )
    elif(args_dict['model_name'] == "SimpleDense"):
        
        init_m_params = {}
        init_m_params.update({'model_type_settings': ast.literal_eval( args_dict.pop('model_type_settings') ) } )
        m_params = hparameters.model_SimpleDense_hparameters(**init_m_params, **args_dict)()
        init_t_params = {}
        init_t_params.update( { 'lookback_target': m_params['data_pipeline_params']['lookback_target'] } )
        init_t_params.update( { 'lookback_feature': m_params['data_pipeline_params']['lookback_feature']})
        
        t_params = hparameters.test_hparameters_ati( **{ **args_dict, **init_t_params} )
    

    elif(args_dict['model_name']=="SimpleConvGRU"):
        init_m_params = {}
        init_m_params.update({'model_type_settings': ast.literal_eval( args_dict.pop('model_type_settings') ) } )
        m_params = hparameters.model_SimpleConvGRU_hparamaters(**init_m_params, **args_dict)()
        init_t_params = {}
        init_t_params.update( { 'lookback_target': m_params['data_pipeline_params']['lookback_target'] } )
        init_t_params.update( { 'lookback_feature': m_params['data_pipeline_params']['lookback_feature']})
        
        t_params = hparameters.test_hparameters_ati( **{ **args_dict, **init_t_params} )

    save_model_settings( m_params, t_params() )

    return t_params, m_params

def parse_arguments(s_dir=None):
    parser = argparse.ArgumentParser(description="Receive input params")

    parser.add_argument('-dd','--data_dir', type=str, help='the directory for the Data', required=False,
                        default='./Data')
    
    parser.add_argument('-mts','--model_type_settings', type=str, help="m_params", required=True)

    parser.add_argument('-mprm','--m_params', type=str, help="m_params", required=False)

    parser.add_argument('-tprm','--t_params', type=str, help="t_params", required=False)

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

    parser.add_argument('-di','--downscaled_input',type=bool, required=False, default=False )

    parser.add_argument('-tst','--train_set_size', type=float,required=False, default=0.6 )

    parser.add_argument('-iim','--input_interpolation_method', type=str, required=False, default='linear' )

    parser.add_argument('-ctsm','--ctsm', type=str, required=False, default=None) #custom training scheme method

    parser.add_argument('-fyi_train','--fyi_train',type=int, required=False, default=1 )

    parser.add_argument('-fyi_test','--fyi_test',type=int, required=False, default=1 )
    
    args_dict = vars(parser.parse_args() )

    return args_dict

#endregion

# region - Saving model / settings / params
def save_model_settings(m_params,t_params):
    
    f_dir = "m_params/{}".format( model_name_mkr(m_params, t_params=t_params) )

    m_path = "m_params.json"
    
    if t_params['trainable']==True:
        t_path = "train_params.json"
    else:
        t_path = "test_params.json"

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

def model_name_mkr(m_params, mode='Generic', load_save="load", t_params={}, custom_test_loc=None ) : 
    """Creates names for vairants of models

    Args:
        m_params ([type]): [description]
        mode (str, optional): [description]. Defaults to 'Generic'.
        load_save (str, optional): [description]. Defaults to "load".
        t_params (dict, optional): [description]. Defaults to {}.
        custom_test_loc ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """    
    if  m_params['model_name'] == "THST":
        model_name = "{}_{}_{}_{}_{}_v{}_dc".format( m_params['model_name'], m_params['model_type_settings']['var_model_type'],
                            m_params['model_type_settings']['distr_type'], 
                            str(m_params['model_type_settings']['discrete_continuous']),
                            m_params['model_type_settings']['location'], m_params['model_type_settings']['model_version']  )

    elif m_params['model_name'] == "SimpleConvGRU":
        model_name =  "{}_{}_{}_{}_{}_v{}".format( m_params['model_name'], m_params['model_type_settings']['var_model_type'],
                        m_params['model_type_settings']['distr_type'], 
                        str(m_params['model_type_settings']['discrete_continuous']),
                        m_params['model_type_settings']['location'],m_params['model_type_settings']['model_version'] )  

    if m_params['model_type_settings'].get('conv_ops_qk',False) == True:
        model_name = model_name + "convopsqk"

    if load_save == "save":
        if custom_test_loc != None:
            pass

        elif m_params.get('location_test', None) != None:
            custom_test_loc = m_params.get('location_test')

        elif m_params.get('location', None) == None:
            custom_test_loc = m_params.get('location')

        model_name = model_name + "_" + custom_test_loc
    
    if t_params.get('ctsm',None) != "Rolling_eval" :
        model_name = model_name + "_tst_" +str( t_params['train_set_size'] )
    
    if t_params.get('ctsm') == "4ds_10years":
        model_name = model_name + "4ds_{}".format(str( t_params['fyi_train']) )
    
    if m_params['model_type_settings'].get('attn_ablation',0) != 0:
        model_name = model_name + "_" + str(m_params['model_type_settings']['attn_ablation'])
    
    model_name = re.sub("[ '\(\[\)\]]|ListWrapper",'',model_name )

    model_name = re.sub(",",'_',model_name )

    return model_name

def cache_suffix_mkr(m_params, t_params):
    """Creates the cache suffix for training datasets

    Args:
        m_params ([type]): [description]
        t_params ([type]): [description]

    Returns:
        [type]: [description]
    """    
    if t_params['ctsm'] == None:
        cache_suffix = '_{}_bs_{}_tst_{}_{}'.format(m_params['model_name'], t_params['batch_size'], t_params.get('train_set_size', 0.6) ,str(m_params['model_type_settings']['location'] ).strip('[]') )
    
    elif t_params['ctsm'] == "4ds_10years":
        cache_suffix ='_{}_bs_{}_fyitrain_{}_loc_{}'.format( m_params['model_name'], t_params['model_name'], str(t_params['fyi_train']), str(m_params['model_type_settings']['location'] ).strip("[]\'") )
        cache_suffix = re.sub("[ '\(\[\)\]]|ListWrapper",'',cache_suffix )
        cache_suffix = re.sub(",",'_',cache_suffix )
    
    return cache_suffix

#endregion

# region data standardization
def standardize_ati(_array, shift, scale, reverse):
    
    if(reverse==False):
        _array = (_array-shift)/scale
    elif(reverse==True):
        _array = (_array*scale)+shift
    
    return _array

# endregion

