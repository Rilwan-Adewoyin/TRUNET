# BNN - Uncertainty SCRNN - ATI Project - PhD Computer Science
#region imports
import os
import sys

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import data_generators
import utility
#os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
tf.keras.backend.set_floatx('float16')
tf.keras.backend.set_epsilon(1e-3)


#DType.is_compatible_with = is_compatible_with
from tensorflow.keras.mixed_precision import experimental as mixed_precision
#tf.config.set_soft_device_placement(True)
#tf.debugging.set_log_device_placement(True)
try:
    gpu_devices = tf.config.list_physical_devices('GPU')
except Exception as e:
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')

print("GPU Available: {}\n GPU Devices:{} ".format(tf.test.is_gpu_available(), gpu_devices) )
for idx, gpu_name in enumerate(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_name, True)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

def is_compatible_with(self, other):
    """Returns True if the `other` DType will be converted to this DType.
    The conversion rules are as follows:
    ```python
    DType(T)       .is_compatible_with(DType(T))        == True
    ```
    Args:
        other: A `DType` (or object that may be converted to a `DType`).
    Returns:
        True if a Tensor of the `other` `DType` will be implicitly converted to
        this `DType`.
    """
    other = tf.dtypes.as_dtype(other)
    if self._type_enum==19 and other.as_datatype_enum==1:
        return True

    return self._type_enum in (other.as_datatype_enum,
                                other.base_dtype.as_datatype_enum)
#from tensorflow.python.framework.dtypes import DType
tf.DType.is_compatible_with = is_compatible_with

import tensorflow_probability as tfp
try:
    import tensorflow_addons as tfa
except Exception as e:
    tfa = None
from tensorflow_probability import layers as tfpl
from tensorflow_probability import distributions as tfd
from tensorboard.plugins.hparams import api as hp

import math

import argparse 
import time
import ast

import models
import hparameters
import gc
import itertools
import json

import custom_losses
import copy
#tf.random.set_seed(seed)
# endregion

def train_loop(train_params, model_params): 

    # region --- Setting up training parameters - to be moved to hparams file
    if model_params['model_type_settings']['location'] == "region_grid":
        train_set_size_batches= int((train_params['train_set_size_elements_b4_sdc_multlocation']//train_params['batch_size']) * np.prod(model_params['region_grid_params']['slides_v_h']) )
        val_set_size_batches = int((train_params['train_set_size_elements_b4_sdc_multlocation']//train_params['batch_size']) * np.prod(model_params['region_grid_params']['slides_v_h']))
    
    elif type( model_params['model_type_settings']['location'][:] ) == list :
        train_set_size_batches = (train_params['train_set_size_elements_b4_sdc_multlocation']//train_params['batch_size']) *train_params['strided_dataset_count'] - (train_params['strided_dataset_count'] - 1)
        val_set_size_batches = (train_params['val_set_size_elements_b4_sdc_multlocation']//train_params['batch_size']) *train_params['strided_dataset_count'] - (train_params['strided_dataset_count'] - 1)

        _city_count = len(model_params['model_type_settings']['location'])
        train_set_size_batches= int(train_set_size_batches * _city_count )
        val_set_size_batches = int(val_set_size_batches * _city_count )
    
    elif model_params['model_type_settings']['location'] == "whole_region" :
        train_set_size_batches= (train_params['train_set_size_elements_b4_sdc_multlocation']//train_params['batch_size'])
        val_set_size_batches = (train_params['train_set_size_elements_b4_sdc_multlocation']//train_params['batch_size'])
    
    else: # in case its a city
        train_set_size_batches = (train_params['train_set_size_elements_b4_sdc_multlocation']//train_params['batch_size']) *train_params['strided_dataset_count'] - (train_params['strided_dataset_count'] - 1)
        val_set_size_batches = (train_params['val_set_size_elements_b4_sdc_multlocation']//train_params['batch_size']) *train_params['strided_dataset_count'] - (train_params['strided_dataset_count'] - 1)
    
    train_batch_reporting_freq = max( int(train_set_size_batches*train_params['dataset_trainval_batch_reporting_freq'] ), 1 )
    val_batch_reporting_freq = max( int(val_set_size_batches*2*train_params['dataset_trainval_batch_reporting_freq'] ), 1)
    #endregion

    # region ----- Defining Model / Optimizer / Losses / Metrics / Records
    model = models.model_loader(train_params, model_params)
    if type(model_params) == list:
        model_params = model_params[0]
    
    if tfa==None:
        optimizer = tf.keras.optimizers.Adam( learning_rate=1e-4, beta_1=0.1, beta_2=0.99, epsilon=1e-5 )
    else:
        total_steps = train_set_size_batches        
        radam = tfa.optimizers.RectifiedAdam( **model_params['rec_adam_params'], total_steps=total_steps*30 ) 
        optimizer = tfa.optimizers.Lookahead(radam, **model_params['lookahead_params'])
    
    optimizer = mixed_precision.LossScaleOptimizer( optimizer, loss_scale=tf.mixed_precision.experimental.DynamicLossScale() )

    if model_params['model_type_settings']['discrete_continuous'] == True:
        #Trying 2 optimizers for discrete_continuious LSTM
        if model_params['model_type_settings']['model_version'] in ["54","55","56","155"]:

            optimizer_rain      = tfa.optimizers.RectifiedAdam( **{"learning_rate":5e-3, "warmup_proportion":0.25, "min_lr":1e-3, "beta_1":0.5, "beta_2":0.85, "decay":0.006, "amsgrad":True, "epsilon":1e-3} , total_steps=total_steps*20 ) 
            optimizer_nonrain   = tfa.optimizers.RectifiedAdam( **{"learning_rate":4e-3, "warmup_proportion":0.25, "min_lr":1e-3, "beta_1":0.5, "beta_2":0.85, "decay":0.006, "amsgrad":True,"epsilon":1e-3} , total_steps=total_steps*20 ) 
            optimizer_dc        = tfa.optimizers.RectifiedAdam( **{"learning_rate":5e-3, "warmup_proportion":0.25, "min_lr":1e-3, "beta_1":0.5, "beta_2":0.85, "decay":0.006, "amsgrad":True,"epsilon":1e-3} , total_steps=total_steps*20 )  
                #copy.deepcopy( optimizer )

            # optimizer_nonrain = tf.keras.optimizers.Nadam( **{"learning_rate":1e-4,"beta_1":0.25, "beta_2":0.30, "epsilon":1e-2, "schedule_decay": (30*train_set_size_batches/3)**-1 }  ) 
            # optimizer_rain = tf.keras.optimizers.Nadam( **{"learning_rate":1e-3, "beta_1":0.25, "beta_2":0.30, "epsilon":1e-2, "schedule_decay": (30*train_set_size_batches/3)**-1  } ) #Every 30 epochs
            # optimizer_dc = tf.keras.optimizers.Nadam(  **{"learning_rate":1e-4,"beta_1":0.25, "beta_2":0.30, "epsilon":1e-2, "schedule_decay": (30*train_set_size_batches/3)**-1 } ) 
            
            optimizers          = [ optimizer_rain, optimizer_nonrain, optimizer_dc ]
            optimizers          = [ mixed_precision.LossScaleOptimizer(_opt, loss_scale=tf.mixed_precision.experimental.DynamicLossScale() ) for _opt in optimizers ]
            optimizer_ready     = [ False ]*len( optimizers )
        else:
            _optimizer = optimizer
            ##monkey patch so optimizer works with mixed precision
    
    train_loss_mean_groupbatch = tf.keras.metrics.Mean(name='train_loss_mse_obj')
    train_loss_mean_epoch = tf.keras.metrics.Mean(name="train_loss_obj_epoch")
    train_mse_metric_epoch = tf.keras.metrics.Mean(name='train_mse_metric')
    train_loss_var_free_nrg_mean_groupbatch = tf.keras.metrics.Mean(name='train_loss_var_free_nrg_obj')
    train_loss_var_free_nrg_mean_epoch = tf.keras.metrics.Mean(name="train_loss_var_free_nrg_obj_epoch")
    val_metric_loss = tf.keras.metrics.Mean(name='val_metric_obj')
    val_metric_mse = tf.keras.metrics.Mean(name='val_metric_mse_obj')

    try:
        df_training_info = pd.read_csv( "checkpoints/{}/checkpoint_scores.csv".format(utility.model_name_mkr(model_params)),
                            header=0, index_col =False   )
        print("Recovered checkpoint scores model csv")
    except Exception as e:
        df_training_info = pd.DataFrame(columns=['Epoch','Train_loss_MSE','Val_loss_MSE','Checkpoint_Path', 'Last_Trained_Batch'] ) #key: epoch number #Value: the corresponding loss #TODO: Implement early stopping
        print("Did not recover checkpoint scores model csv")
  
    # endregion

    # region ----- Setting up Checkpoints 
        #  (For Epochs)
    checkpoint_path_epoch = "checkpoints/{}/epoch".format(utility.model_name_mkr(model_params))
    os.makedirs(checkpoint_path_epoch,exist_ok=True)
        
    ckpt_epoch = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager_epoch = tf.train.CheckpointManager(ckpt_epoch, checkpoint_path_epoch, 
                max_to_keep=train_params['checkpoints_to_keep_epoch'], keep_checkpoint_every_n_hours=None)    
     
        # (For Batches)
    checkpoint_path_batch = "checkpoints/{}/batch".format(utility.model_name_mkr(model_params))
    os.makedirs(checkpoint_path_batch,exist_ok=True)
        #Create the checkpoint path and the checpoint manager. This will be used to save checkpoints every n epochs
    ckpt_batch = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager_batch = tf.train.CheckpointManager(ckpt_batch, checkpoint_path_batch, max_to_keep=train_params['checkpoints_to_keep_batch'], keep_checkpoint_every_n_hours=None)

        # restoring checkpoint from last batch if it exists
    if ckpt_manager_batch.latest_checkpoint: #restoring last checkpoint if it exists
        ckpt_batch.restore(ckpt_manager_batch.latest_checkpoint)
        print ('Latest checkpoint restored from {}'.format(ckpt_manager_batch.latest_checkpoint  ) )

    else:
        print (' Initializing from scratch')

    # endregion     

    # region Logic for setting up resume location
    starting_epoch =  int(max( df_training_info['Epoch'], default=0 )) 
    df_batch_record = df_training_info.loc[ df_training_info['Epoch'] == starting_epoch,'Last_Trained_Batch' ]

    if( len(df_batch_record)==0 ):
        batches_to_skip = 0
    elif (df_batch_record.iloc[0]==-1 ):
        starting_epoch = starting_epoch + 1
        batches_to_skip = 0
    else:
        batches_to_skip = int(df_batch_record.iloc[0])
        if batches_to_skip >= train_set_size_batches :
            starting_epoch = starting_epoch + 1
            batches_to_skip = train_set_size_batches
    
    print("batches to skip", batches_to_skip)

    #batches_to_skip_on_error = 2
    # endregion

    # region --- Tensorboard
    os.makedirs("log_tensboard/{}".format(utility.model_name_mkr(model_params)), exist_ok=True ) 
    writer = tf.summary.create_file_writer( "log_tensboard/{}".format(utility.model_name_mkr(model_params) ) )
    # endregion

    # region ---- Making Datasets
    if model_params['model_name'] == "DeepSD":
        #ds_train = data_generators.load_data_vandal( batches_to_skip*train_params['batch_size'], train_params, model_params, data_dir=train_params['data_dir'] )
        ds_val = data_generators.load_data_vandal( train_set_size_batches*train_params['batch_size'], train_params, model_params, data_dir=train_params['data_dir'] )
            #temp fix to the problem where if we init ds_train at batches_to_skip, then every time we reuse ds_train then it will inevitably start from that skipped to region on the next iteration 
        ds_train = data_generators.load_data_vandal( batches_to_skip*train_params['batch_size'], train_params, model_params, data_dir=train_params['data_dir'] )

    elif model_params['model_name'] in ["THST", "SimpleLSTM", "SimpleGRU" ,"SimpleConvLSTM", "SimpleConvGRU","SimpleDense"]:
    
        # version 2  for iters, will be used when you want to increase the size of training set, while doing stateful training
        li_start_days_train = np.arange( train_params['train_start_date'], 
                                    train_params['train_start_date'] + np.timedelta64(train_params['lookback_target'],'D'),
                                    np.timedelta64(train_params['lookback_target']//train_params['strided_dataset_count'],'D'), dtype='datetime64[D]')[:train_params['strided_dataset_count']]
        
        li_start_days_val = np.arange( train_params['val_start_date'], 
                                    train_params['val_start_date'] + np.timedelta64(train_params['lookback_target'],'D'),
                                    np.timedelta64(train_params['lookback_target']//train_params['strided_dataset_count'],'D'), dtype='datetime64[D]')[:train_params['strided_dataset_count']]

        li_ds_trains = [ data_generators.load_data_ati( train_params, model_params, day_to_start_at=sd, data_dir=train_params['data_dir']) for sd in li_start_days_train ]
        li_ds_vals = [ data_generators.load_data_ati( train_params, model_params, day_to_start_at=sd, data_dir=train_params['data_dir']) for sd in li_start_days_val ]
        
        li_ds_trains = [ _ds.take( math.ceil( train_set_size_batches/train_params['strided_dataset_count'] ) )  if idx==0 
                            else _ds.take( train_set_size_batches//train_params['strided_dataset_count'] )  #This ensures that the for loops switch between validation and train sets at the right counts
                            for idx,_ds in  enumerate(li_ds_trains) ] #Only the first ds takes the full amount
       
        li_ds_vals = [ _ds.take( math.ceil( val_set_size_batches/train_params['strided_dataset_count'] ) )  if idx==0 
                    else _ds.take( val_set_size_batches//train_params['strided_dataset_count'] )  #This ensures that the for loops switch between validation and train sets at the right counts
                    for idx,_ds in  enumerate(li_ds_vals) ] #Only the first ds takes the full amount

        ds_train = li_ds_trains[0]
        for idx in range(1,len(li_ds_trains ) ):
            ds_train = ds_train.concatenate( li_ds_trains[idx] )

        ds_val = li_ds_vals[0]
        for idx in range(1,len(li_ds_vals ) ):
            ds_val = ds_val.concatenate( li_ds_vals[idx] )

        #Version that doesnt work on warwick desktop
        # ds_train_val = ds_train.concatenate(ds_val).repeat(train_params['epochs']-starting_epoch)
        # ds_train_val = ds_train_val.skip(batches_to_skip)
        # iter_val_train = enumerate(ds_train_val)
        # iter_train = iter_val_train
        # iter_val = iter_val_train

        #Version that ensures validation and train set are well defined
        ds_train = ds_train.repeat(train_params['epochs']-starting_epoch)
        ds_val = ds_val.repeat(train_params['epochs']-starting_epoch)

        ds_train = ds_train.skip(batches_to_skip)

        iter_train = enumerate(ds_train)
        iter_val = enumerate(ds_val)



    #endregion

    #region Setting up points at which we must reset states for training on a particular location
    if train_params['strided_dataset_count'] > 1:
        if type( model_params['model_type_settings']['location'][:] ) == list:
            #Note: In this scenario The train dataset is
                # 1) a series: dst1, dst2, dstN where N is the number of sdc
                # 2) each ds in series contains a series relating to a single location eg. dst1 = dst1_1, dst1_2, dst_l3...
            _city_count =  len(model_params['model_type_settings']['location'] )
            
            bc_ds_in_dst1_train = math.ceil( train_set_size_batches/( train_params['strided_dataset_count'] *_city_count ) ) #batch_count
            bc_ds_in_others_train = (train_set_size_batches/_city_count) //( train_params['strided_dataset_count'])

            bc_ds_in_dst1_val = math.ceil( val_set_size_batches/( train_params['strided_dataset_count'] *_city_count ) ) #batch_count
            bc_ds_in_others_val = (val_set_size_batches/_city_count) //( train_params['strided_dataset_count'] )

            reset_idxs_training_dst1 = np.cumsum( [bc_ds_in_dst1_train]*_city_count )
            reset_idxs_validation_dst1 = np.cumsum( [bc_ds_in_dst1_val]*_city_count )

            reset_idxs_training_others = np.arange( reset_idxs_training_dst1[-1], train_set_size_batches ,bc_ds_in_others_train, dtype=int  )
            reset_idxs_validation_others = np.arange( reset_idxs_validation_dst1[-1], val_set_size_batches ,bc_ds_in_others_val, dtype=int  )

            reset_idxs_training = reset_idxs_training_dst1.tolist() + reset_idxs_training_others.tolist()
            reset_idxs_validation = reset_idxs_validation_dst1.tolist() + reset_idxs_validation_others.tolist()

        else:
            reset_idxs_training = np.arange( math.ceil( train_set_size_batches/train_params['strided_dataset_count'] ), train_set_size_batches,  train_set_size_batches//train_params['strided_dataset_count'] ).tolist()
            reset_idxs_validation = np.arange(math.ceil( val_set_size_batches/train_params['strided_dataset_count'] ), val_set_size_batches,  train_set_size_batches//train_params['strided_dataset_count'] ).tolist()
    else:
        reset_idxs_training = [ ]
        reset_idxs_validation = [ ]

    # endregion

    # region --- Train and Val
    if model_params['model_type_settings']['var_model_type'] in ['horseshoefactorized','horseshoestructured'] :
        tf.config.experimental_run_functions_eagerly(True)

    for epoch in range(starting_epoch, int(train_params['epochs']) ):
        #region metrics, loss, dataset, and standardization
        train_loss_mean_groupbatch.reset_states()
        train_loss_var_free_nrg_mean_groupbatch.reset_states()
        train_loss_mean_epoch.reset_states()
        train_mse_metric_epoch.reset_states()
        train_loss_var_free_nrg_mean_epoch.reset_states()
        val_metric_loss.reset_states()
        val_metric_mse.reset_states()
                        
        df_training_info = df_training_info.append( { 'Epoch':epoch, 'Last_Trained_Batch':0 }, ignore_index=True )
        
        start_epoch = time.time()
        start_epoch_val = None
        inp_time = None
        start_batch_time = time.time()
        
        #endregion 

        batch=0
        print("\n\nStarting EPOCH {} Batch {}/{}".format(epoch, batches_to_skip+1, train_set_size_batches))
        
        #region Train
        model.reset_states()
        for batch in range(batches_to_skip,train_set_size_batches):
            
            if batch in reset_idxs_training :
                model.reset_states()

            step = batch + (epoch)*train_set_size_batches
            #if model_params['model_type_settings']['location'] == 'region_grid':
            if model_params['model_name'] in ["SimpleConvLSTM","SimpleConvGRU","THST"]:
                idx, (feature, target, mask) = next(iter_train)
            else:
                idx, (feature, target) = next(iter_train)

            with tf.GradientTape(persistent=False) as tape:
                if model_params['model_name'] == "DeepSD":
                    #region stochastic fward passes
                    if model_params['model_type_settings']['stochastic_f_pass']>1:
                        
                        li_preds = model.predict(feature, model_params['model_type_settings']['stochastic_f_pass'], pred=False )

                        preds_stacked = tf.concat( li_preds,axis=-1)
                        preds_mean = tf.reduce_mean( preds_stacked, axis=-1)
                        preds_scale = tf.math.reduce_std( preds_stacked, axis=-1)

                            #masking for water/sea predictions
                        preds_mean = tf.where( train_params['bool_water_mask'] , preds_mean, 0 )
                        preds_scale = tf.where( train_params['bool_water_mask'] , preds_scale, 1 )

                    elif model_params['model_type_settings']['stochastic_f_pass']==1:
                        raise NotImplementedError("haven't handled case for mean of logarithms of predictoins") 

                        preds = model( feature, tape=tape ) #shape batch_size, output_h, output_w, 1 #TODO Debug, remove tape variable from model later

                        #noise_std = tfd.HalfNormal(scale=5)     #TODO(akanni-ade): remove (mask) eror for predictions that are water i.e. null, through water_mask
                        preds = utility.water_mask( tf.squeeze(preds), train_params['bool_water_mask'])
                        preds = tf.reshape( preds, [train_params['batch_size'], -1] )       #TODO:(akanni-ade) This should decrease exponentially during training #RESEARCH: NOVEL Addition #TODO:(akanni-ade) create tensorflow function to add this
                        target = tf.reshape( target, [train_params['batch_size'], -1] )     #NOTE: In the original Model Selection paper they use Guassian Likelihoods for loss with a precision (noise_std) that is Gamma(6,6)
                        preds_mean = preds
                        preds_scale = 0.1
                    # endregion
                    
                    #region Discrete continuous or not
                    if( model_params['model_type_settings']['discrete_continuous']==False ):                                                            
                        #note - on discrete_continuous==False, there is a chance that the preds_scale term takes value 0 i.e. relu output is 0 all times. 
                            #  So for this scenario just use really high variance to reduce the effect of this loss
                        preds_scale = tf.where(tf.equal(preds_scale,0.0), 5, preds_scale)

                        if(model_params['model_type_settings']['distr_type']=="Normal" ):
                            preds_distribution = tfd.Normal( loc=preds_mean, scale= preds_scale)
                            
                            _1 = tf.where(train_params['bool_water_mask'], target, 0) 
                            _2 = preds_distribution.log_prob( _1)
                            _3 = tf.boolean_mask( _2, train_params['bool_water_mask'],axis=1 )
                            log_likelihood = tf.reduce_mean( _3)
                                #This represents the expected log_likelihood corresponding to each target y_i in the mini batch

                        kl_loss_weight = utility.kl_loss_weighting_scheme(train_set_size_batches, batch, model_params['model_type_settings']['var_model_type'] ) #TODO: Implement scheme where kl loss increases during training
                        kl_loss = tf.cast( tf.math.reduce_sum( model.losses ) * kl_loss_weight * (1/model_params['model_type_settings']['stochastic_f_pass']), tf.float32)  #This KL-loss is already normalized against the number of samples of weights drawn #TODO: Later implement your own Adam type method to determine this
                        
                        var_free_nrg_loss = kl_loss  - log_likelihood

                        l  = var_free_nrg_loss

                    elif( model_params['model_type_settings']['discrete_continuous']==True ):
                        #get classification labels & predictions, true/1 means it has rained
                        labels_true = tf.cast( tf.greater( target, utility.standardize( model_params['model_type_settings']['precip_threshold'],reverse=False,distr_type=model_params['model_type_settings']['distr_type'] ) ), tf.float32 )
                        labels_pred = tf.cast( tf.greater( preds_mean, utility.standardize(model_params['model_type_settings']['precip_threshold'],reverse=False, distr_type= model_params['model_type_settings']['distr_type']) ),tf.float32 )

                        #  gather predictions which are conditional on rain
                        bool_indices_cond_rain = tf.where(tf.equal(labels_true,1),True,False )
                        bool_water_mask = train_params['bool_water_mask']

                        bool_cond_rain=  tf.math.logical_and(bool_indices_cond_rain, bool_water_mask )

                        _preds_cond_rain_mean = tf.boolean_mask( preds_mean, bool_cond_rain)
                        _preds_cond_rain_scale = tf.boolean_mask(preds_scale, bool_cond_rain)
                        _target_cond_rain = tf.boolean_mask( target, bool_cond_rain )

                        # making distributions
                        if( model_params['model_type_settings']['distr_type'] =="Normal" ):
                            preds_distribution_condrain = tfd.Normal( loc=_preds_cond_rain_mean, scale= tf.where( _preds_cond_rain_scale==0, 1, _preds_cond_rain_scale  ) )

                        elif(model_params['model_type_settings']['distr_type'] == "LogNormal" ):
                            epsilon = tf.random.uniform( preds_stacked.shape.as_list(), minval=1e-10, maxval=1e-7 )
                            preds_stacked_adj = tf.where( preds_stacked==0, epsilon, preds_stacked )
                            log_vals = tf.math.log( preds_stacked_adj )
                            log_distr_mean = tf.math.reduce_mean( log_vals, axis=-1 )
                            log_distr_std = tf.math.reduce_std(log_vals, axis=-1 )
                            #Filtering out value 

                            preds_distribution_condrain = tfd.LogNormal( loc=tf.boolean_mask(log_distr_mean, bool_cond_rain) , 
                                                                                scale=tf.boolean_mask( log_distr_std, bool_cond_rain) ) 
                            
                        else:
                            raise ValueError

                        # calculating log-likehoods
                        log_cross_entropy_rainclassification = tf.reduce_mean( tf.boolean_mask(
                                            tf.keras.backend.binary_crossentropy( labels_true, labels_pred, from_logits=True),train_params['bool_water_mask'],axis=1 ) )

                        log_likelihood_cond_rain =  tf.reduce_sum( preds_distribution_condrain.log_prob( _target_cond_rain ) ) / tf.size( tf.boolean_mask( target, train_params['bool_water_mask'], axis=1 ) , out_type=tf.float32) 
                        log_likelihood = log_likelihood_cond_rain - log_cross_entropy_rainclassification

                        kl_loss_weight = utility.kl_loss_weighting_scheme(train_set_size_batches, batch, model_params['model_type_settings']['var_model_type'] ) 
                        kl_loss = tf.cast(tf.math.reduce_sum( model.losses ) * kl_loss_weight * (1/model_params['model_type_settings']['stochastic_f_pass'] ), tf.float32)  #This KL-loss is already normalized against the number of samples of weights drawn #TODO: Later implement your own Adam type method to determine this

                        var_free_nrg_loss = kl_loss  - log_likelihood
                        l = var_free_nrg_loss
                        

                        loss_mse_condrain = tf.reduce_mean( tf.keras.losses.MSE( _target_cond_rain , _preds_cond_rain_mean) )
                    #endregion

                    target_filtrd = tf.reshape( tf.boolean_mask(       target , train_params['bool_water_mask'], axis=1 ),       [train_params['batch_size'], -1] )
                    preds_mean_filtrd = tf.reshape( tf.boolean_mask( preds_mean, train_params['bool_water_mask'],axis=1 ),  [train_params['batch_size'], -1] )

                    metric_mse = tf.reduce_mean( tf.keras.losses.MSE( target_filtrd , preds_mean_filtrd)  )

                    scaled_loss = optimizer.get_scaled_loss(l)
                    scaled_gradients = tape.gradient( scaled_loss, model.trainable_variables )
                    gradients = optimizer.get_unscaled_gradients(scaled_gradients) 
                    gc.collect()
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    
                elif( model_params['model_name'] in ["SimpleLSTM","SimpleGRU","SimpleDense"]):
                    if( model_params['model_type_settings']['stochastic']==False):
                        target, mask = target # (bs, seq_len)

                        preds = model( tf.cast(feature,tf.float16), train_params['trainable'] ) #( bs, tar_seq_len)
                        preds = tf.squeeze( preds )

                        preds_filtrd = tf.boolean_mask( preds, mask )
                        target_filtrd = tf.boolean_mask( target, mask )

                        #scaling them back up
                        preds_filtrd = utility.standardize_ati( preds_filtrd, train_params['normalization_shift']['rain'], 
                                                                train_params['normalization_scales']['rain'], reverse=True)

                        if model_params['model_type_settings']['discrete_continuous'] == False:
                            loss_mse = tf.keras.losses.MSE(target_filtrd, preds_filtrd)
                            metric_mse = loss_mse
                            _l1 = loss_mse/3
                            _l2 = _l1
                            _l3 = _l2

                        elif model_params['model_type_settings']['discrete_continuous'] == True:
                            #get classification labels & predictions, true/1 means it has rained   
                            
                            labels_true = tf.where( target_filtrd>model_params['model_type_settings']['precip_threshold'], 1.0, 0.0)
                            labels_pred = tf.where( preds_filtrd>model_params['model_type_settings']['precip_threshold'], 1.0, 0.0)
                            alpha = 1000
                            labels_pred_cont_approx = tf.math.sigmoid( alpha*preds_filtrd - alpha/2 )  #Label calculation allowing back-prop 
                                #Note in tf.float32 tf.math.sigmoid(16.7)==1 and  

                            rain_count = tf.math.count_nonzero( labels_true,dtype=tf.float32 )
                            all_count = tf.size( target_filtrd,out_type=tf.float32 )

                            #  gather predictions which are conditional on rain
                            bool_cond_rain = tf.where(tf.equal(labels_true,1.0),True,False )

                            preds_cond_rain_mean = tf.boolean_mask( preds_filtrd, bool_cond_rain)
                            target_cond_rain = tf.boolean_mask( target_filtrd, bool_cond_rain )

                            preds_cond_no_rain_mean = tf.boolean_mask( preds_filtrd, tf.math.logical_not(bool_cond_rain) )
                            target_cond_no_rain = tf.boolean_mask( target_filtrd, tf.math.logical_not(bool_cond_rain) )
                            
                            if model_params['model_type_settings']['distr_type'] == 'Normal': #These two below handle dc cases of normal and log_normal
                                
                                loss_mse = (rain_count/all_count)*tf.keras.losses.MSE(target_cond_rain, preds_cond_rain_mean)
                                metric_mse = loss_mse
                            
                            elif model_params['model_type_settings']['distr_type'] == 'LogNormal':

                                train_mse_cond_rain = (rain_count/all_count) * tf.keras.metrics.MSE(target_cond_rain, preds_cond_rain_mean)                            
                                metric_mse =  train_mse_cond_rain
                                _l1 = (rain_count/all_count) * custom_losses.lnormal_mse(target_cond_rain, preds_cond_rain_mean) #loss1 conditional on rain
                                
                                #_l1 = train_mse_cond_rain
                                loss_mse = _l1  

                                #temp: adding an mse loss to the values which are under 0.5
                                if(model_params['model_type_settings']['model_version'] in ["3","4","44","46","54","55","155"] ):
                                    
                                    train_mse_cond_no_rain = ((all_count-rain_count)/all_count)*tf.keras.metrics.MSE(target_cond_no_rain, preds_cond_no_rain_mean ) #loss2 conditional on no rain
                                    _l2 = train_mse_cond_no_rain
                                    loss_mse += train_mse_cond_no_rain
                                    metric_mse += train_mse_cond_no_rain
                                else:
                                    train_mse_cond_no_rain = ((all_count-rain_count)/all_count)*tf.keras.metrics.MSE(target_cond_no_rain, preds_cond_no_rain_mean ) #loss2 conditional on no rain
                                    _l2 = 0
                                    loss_mse += 0
                                    metric_mse += train_mse_cond_no_rain
                            
                            if(model_params['model_type_settings']['model_version'] in ["3","4","44","45","145","47","48","49","50","51","52","53","54","56"] ):

                                log_cross_entropy_rainclassification = tf.reduce_mean( 
                                                tf.keras.backend.binary_crossentropy( labels_true, labels_pred, from_logits=False) )                         #loss3 conditional on no rain v2
                                _l3 = tf.reduce_mean( 
                                                tf.keras.backend.binary_crossentropy( labels_true, labels_pred_cont_approx, from_logits=False) )  #differentiable version     #loss3 conditional on no rain v2
                                loss_mse += log_cross_entropy_rainclassification
                            
                            else:
                                log_cross_entropy_rainclassification = 0
                                _l3 = 0
                                loss_mse += log_cross_entropy_rainclassification

                        if(model_params['model_type_settings']['model_version'] in ["54","55","56"] ): #multiple optimizers 
                            
                            #optm_idx = step % len(optimizers)
                            if(model_params['model_type_settings']['model_version'] in ["54"]):
                                optm_idx = (step // int(train_set_size_batches/4) ) % len(optimizers)
                                losses = [_l1, _l2, _l3  ]

                            elif(model_params['model_type_settings']['model_version'] in ["55","155"]):
                                optm_idx = (step // int(train_set_size_batches/3) ) % 2
                                losses = [_l1, _l2 ]
                            
                            elif(model_params['model_type_settings']['model_version'] in ["56"]):
                                optm_idx = (step // int(train_set_size_batches/3) ) % 2
                                losses = [_l1, _l3 ]

                            _optimizer = optimizers[optm_idx]

                            scaled_loss = _optimizer.get_scaled_loss( losses[optm_idx] + sum(model.losses) )
                        else:
                            _optimizer = optimizer
                            scaled_loss = _optimizer.get_scaled_loss(_l1 + _l2 + _l3 + sum(model.losses) )

                    elif(model_params['model_type_settings']['stochastic']==True):
                        raise NotImplementedError("mc_dropout model uses the deterministically trained model")

                    scaled_gradients = tape.gradient( scaled_loss, model.trainable_variables )
                    gradients = _optimizer.get_unscaled_gradients(scaled_gradients)
                    
                    if(model_params['model_type_settings']['model_version'] in ["54","55","56","155"] ): #multiple optimizers 
                        gradients, _ = tf.clip_by_global_norm( gradients, 30.0 )

                        #insert code here to handle ensuring all loss functions start at the same time, e.g. when all optimizers have stopped producing nans

                        if optimizer_ready[optm_idx]==False:
                            if tf.math.is_finite( _ ):
                                optimizer_ready[optm_idx]=True
                            else:
                                _optimizer.loss_scale.update(gradients)
                                    #make optimizer reduce loss scaling
                        
                        if tf.reduce_all(optimizer_ready):
                          _optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                    else:
                        _optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                elif (model_params['model_name'] in ["SimpleConvLSTM","SimpleConvGRU","THST"]):
                    if model_params['model_type_settings']['location'] == 'region_grid'  or model_params['model_type_settings']['twoD']==True:
                        if( tf.reduce_any( mask[:, :, 6:10, 6:10] )==False ):
                            continue
                    else:
                        target, mask = target # (bs, h, w) 

                    if( model_params['model_type_settings']['stochastic']==False):

                        preds = model( tf.cast(feature,tf.float16), train_params['trainable'] ) #( bs, tar_seq_len, h, w)
                        preds = tf.squeeze(preds)

                        if (model_params['model_type_settings']['location']=='region_grid' )  or model_params['model_type_settings']['twoD']==True :  #focusing on centre of square only
                            preds = preds[:, :, 6:10, 6:10]
                            mask = mask[:, :, 6:10, 6:10]
                            target = target[:, :, 6:10, 6:10]

                        preds_filtrd = tf.boolean_mask( preds, mask )
                        target_filtrd = tf.boolean_mask( target, mask )

                        preds_filtrd = utility.standardize_ati( preds_filtrd, train_params['normalization_shift']['rain'], 
                                                                train_params['normalization_scales']['rain'], reverse=True)

                        loss_mse = tf.keras.losses.MSE(target_filtrd, preds_filtrd) 
                        metric_mse = loss_mse
                        scaled_loss = optimizer.get_scaled_loss(loss_mse + sum(model.losses))

                    elif(model_params['model_type_settings']['stochastic']==True):
                        raise NotImplementedError
                    scaled_gradients = tape.gradient( scaled_loss, model.trainable_variables )
                    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    
                gc.collect()
            #region Tensorboard Update
            
            with writer.as_default():
                if( model_params['model_type_settings']['stochastic']==True ):
                    tf.summary.scalar('train_loss_var_free_nrg', var_free_nrg_loss , step =  step )
                    tf.summary.scalar('kl_loss', kl_loss, step=step )
                    tf.summary.scalar('neg_log_likelihood', -log_likelihood, step=step )
                    tf.summary.scalar('train_metric_mse', metric_mse , step = step )

                    if model_params['model_type_settings']['discrete_continuous'] == True:
                        tf.summary.scalar('train_mse_cond_rain', train_mse_cond_rain, step=step )
                        tf.summary.scalar('train_mse_cond_no_rain',train_mse_cond_no_rain, step = step)
                        tf.summary.scalar('cross_entropy_rainclassification',log_cross_entropy_rainclassification, step=step)
                
                elif( model_params['model_type_settings']['stochastic']==False ):
                    tf.summary.scalar('train_loss', loss_mse , step = step )
                    tf.summary.scalar('train_metric_mse', metric_mse , step = step )

                if model_params['model_type_settings']['discrete_continuous'] == True:
                        tf.summary.scalar('train_mse_cond_rain', train_mse_cond_rain, step=step )
                        tf.summary.scalar('train_mse_cond_no_rain',train_mse_cond_no_rain, step = step)
                        tf.summary.scalar('cross_entropy_rainclassification',log_cross_entropy_rainclassification, step=step)
    
                for grad, _tensor in zip( gradients, model.trainable_variables):
                    if grad is not None:
                        tf.summary.histogram( "Grad:{}".format( _tensor.name ) , grad, step = step  )
                        tf.summary.histogram( "Weights:{}".format(_tensor.name), _tensor , step = step ) 
            #endregion

            #region training Reporting and Metrics updates
            if( model_params['model_type_settings']['stochastic']==True ):
                train_loss_var_free_nrg_mean_groupbatch( var_free_nrg_loss )
                train_loss_var_free_nrg_mean_epoch( var_free_nrg_loss )
                train_loss_mean_groupbatch( loss_mse )
                train_loss_mean_epoch( loss_mse )
                train_mse_metric_epoch( metric_mse)

            elif( model_params['model_type_settings']['stochastic']==False ):
                train_loss_mean_groupbatch( loss_mse )
                train_loss_mean_epoch( loss_mse )
                train_mse_metric_epoch( metric_mse )
                                        
            ckpt_manager_batch.save()
            if( (batch+1)%train_batch_reporting_freq==0 or batch+1 == train_set_size_batches):
                batches_report_time =  time.time() - start_batch_time

                est_completion_time_seconds = (batches_report_time/train_params['dataset_trainval_batch_reporting_freq']) * (train_set_size_batches - batch)/train_set_size_batches
                est_completion_time_mins = est_completion_time_seconds/60

                print("\t\tBatch:{}/{}\tTrain Loss: {:.8f} \t Batch Time:{:.4f}\tEpoch mins left:{:.1f}".format(batch, train_set_size_batches, train_loss_mean_groupbatch.result(), batches_report_time, est_completion_time_mins ) )
                train_loss_mean_groupbatch.reset_states()
                start_batch_time = time.time()

                # Updating record of the last batch to be operated on in training epoch
            df_training_info.loc[ ( df_training_info['Epoch']==epoch) , ['Last_Trained_Batch'] ] = batch
            df_training_info.to_csv( path_or_buf="checkpoints/{}/checkpoint_scores.csv".format(utility.model_name_mkr(model_params)), header=True, index=False )

        start_epoch_val = time.time()
        start_batch_time = time.time()

        print("\tStarting Validation")
        
        #endregion
        # endregion

        #region Valid
        model.reset_states()
        for batch in range(val_set_size_batches):

            if batch in reset_idxs_validation :
                model.reset_states()

            if model_params['model_name'] in ["SimpleConvLSTM","SimpleConvGRU","THST"]:
                idx, (feature, target, mask) = next(iter_val)
            else:
                idx, (feature, target) = next(iter_val)

            if model_params['model_name'] == "DeepSD":
                if model_params['model_type_settings']['stochastic'] ==True: #non stochastic version

                    preds = model( feature, training=False )
                    preds = utility.water_mask( tf.squeeze(preds), train_params['bool_water_mask'])
                    
                    target_filtrd = tf.reshape( tf.boolean_mask(  target , train_params['bool_water_mask'], axis=1 ), [train_params['batch_size'], -1] )
                    preds_filtrd = tf.reshape( tf.boolean_mask( preds, train_params['bool_water_mask'],axis=1 ), [train_params['batch_size'], -1] )
                    val_metric_loss( tf.reduce_mean( tf.keras.metrics.MSE( target_filtrd , preds_filtrd ) )  ) #TODO: Ensure that both preds and target are reshaped prior 
                    #TODO: Add Discrete Continuous Metric here is wrong, should be same as other version

            elif model_params['model_name'] in ["SimpleLSTM", "SimpleGRU" ,"SimpleDense"]:
                if model_params['model_type_settings']['stochastic'] == False:
                    target, mask = target
                    preds = model(tf.cast(feature,tf.float16),training=False )
                    preds = tf.squeeze(preds)

                    preds_filtrd = tf.boolean_mask( preds, mask )
                    target_filtrd = tf.boolean_mask( target, mask )

                    preds_filtrd = utility.standardize_ati( preds_filtrd, train_params['normalization_shift']['rain'], 
                                                            train_params['normalization_scales']['rain'], reverse=True)

                    if model_params['model_type_settings']['discrete_continuous'] == False:
                        _ = tf.reduce_mean(tf.keras.metrics.MSE( target_filtrd , preds_filtrd ) ) 
                        val_metric_loss( _ )
                        val_metric_mse( _ )

                    elif model_params['model_type_settings']['discrete_continuous'] == True:
                        #get classification labels & predictions, true/1 means it has rained   
                        labels_true = tf.cast( tf.greater( target_filtrd, model_params['model_type_settings']['precip_threshold'] ), tf.float32 )
                        labels_pred = tf.cast( tf.greater( preds_filtrd, model_params['model_type_settings']['precip_threshold'] ) ,tf.float32 )
                        
                        rain_count = tf.math.count_nonzero( target_filtrd, dtype=tf.float32 )
                        all_count = tf.size( target_filtrd, out_type=tf.float32 )

                        #  gather predictions which are conditional on rain
                        bool_cond_rain = tf.where(tf.equal(labels_true,1),True,False )

                        preds_cond_rain_mean = tf.boolean_mask( preds_filtrd, bool_cond_rain)
                        target_cond_rain = tf.boolean_mask( target_filtrd, bool_cond_rain )
                        
                        preds_cond_no_rain_mean = tf.boolean_mask( preds_filtrd, tf.math.logical_not(bool_cond_rain) )
                        target_cond_no_rain = tf.boolean_mask( target_filtrd, tf.math.logical_not(bool_cond_rain) )
                        

                        if model_params['model_type_settings']['distr_type'] == 'Normal': #These two below handle dc cases of normal and log_normal

                            loss_mse = (rain_count/all_count)*tf.reduce_mean(tf.keras.metrics.MSE( target_cond_rain , preds_cond_rain_mean ) )  #NOTE: currently the val_metric_loss represents a different target for different combinations of distr_type and stochastic
                            val_mse = val_metric_loss( tf.reduce_mean(tf.keras.metrics.MSE( target_filtrd , preds_filtrd ) )  )

                        elif model_params['model_type_settings']['distr_type'] == 'LogNormal':
                            val_mse = (rain_count/all_count)*tf.reduce_mean(tf.keras.metrics.MSE( target_filtrd , preds_filtrd ) ) 

                            loss_mse = (rain_count/all_count)*tf.reduce_mean(tf.keras.metrics.MSE( target_filtrd , preds_filtrd ) ) # (rain_count/all_count) * custom_losses.lnormal_mse(target_cond_rain, preds_cond_rain_mean)

                            if(model_params['model_type_settings']['model_version'] in ["3","4","44","46","54","55","155"] ):
                                val_mse_cond_no_rain = ((all_count-rain_count)/all_count)*tf.keras.metrics.MSE(target_cond_no_rain, preds_cond_no_rain_mean)
                                loss_mse += val_mse_cond_no_rain
                                val_mse += val_mse_cond_no_rain
                            else:
                                val_mse_cond_no_rain = ((all_count-rain_count)/all_count)*tf.keras.metrics.MSE(target_cond_no_rain, preds_cond_no_rain_mean)
                                val_mse +=val_mse_cond_no_rain
                            

                            if(model_params['model_type_settings']['model_version'] in ["3","4","44","45","145","47","48","49","50","51","52",'53','54',"56"] ):
                                log_cross_entropy_rainclassification = tf.reduce_mean( 
                                        tf.keras.backend.binary_crossentropy( labels_true, labels_pred, from_logits=False) )
                            
                            else:
                                log_cross_entropy_rainclassification = 0

                            loss_mse +=     log_cross_entropy_rainclassification
                        
                        val_metric_loss(loss_mse)
                        val_metric_mse(val_mse)

                elif model_params['model_type_settings']['stochastic'] == True:
                    raise NotImplementedError
            
            elif model_params['model_name'] in ["SimpleConvLSTM", "SimpleConvGRU","THST"]:
                
                if model_params['model_type_settings']['location'] == 'region_grid' or model_params['model_type_settings']['twoD']==True:
                    if tf.reduce_any( mask[:, :, 6:10, 6:10] )==False  :
                        continue
                else:
                    target, mask = target # (bs, h, w) 

                if model_params['model_type_settings']['stochastic'] == False:
                    preds = model(tf.cast(feature,tf.float16), training=False )
                    preds = tf.squeeze(preds)

                    if model_params['model_type_settings']['location']=='region_grid'  or model_params['model_type_settings']['twoD']==True : #focusing on centre of square only
                        preds = preds[:, :, 6:10, 6:10]
                        mask = mask[:, :, 6:10, 6:10]
                        target = target[:, :, 6:10, 6:10]
 
                    preds_filtrd = tf.boolean_mask( preds, mask )
                    target_filtrd = tf.boolean_mask( target, mask )
                    preds_filtrd = utility.standardize_ati( preds_filtrd, train_params['normalization_shift']['rain'], 
                                                            train_params['normalization_scales']['rain'], reverse=True)
                    _mse = tf.reduce_mean(tf.keras.metrics.MSE( target_filtrd , preds_filtrd ) )
                    val_metric_loss(  _mse )
                    val_metric_mse( _mse )

                elif(model_params['model_type_settings']['stochastic']==True):
                    raise NotImplementedError                
                
            if ( (batch+1) % val_batch_reporting_freq) ==0 or batch+1==val_set_size_batches :
                batches_report_time =  time.time() - start_batch_time
                est_completion_time_seconds = (batches_report_time/train_params['dataset_trainval_batch_reporting_freq']) *( 1 -  ((batch)/val_set_size_batches ) )
                est_completion_time_mins = est_completion_time_seconds/60

                print("\t\tCompleted Validation Batch:{}/{} \t Time:{:.4f} \tEst Time Left:{:.1f}".format( batch, val_set_size_batches ,batches_report_time,est_completion_time_mins ))
                                            
                start_batch_time = time.time()
                #iter_train = None
                if( batch +1 == val_set_size_batches  ):
                    batches_to_skip = 0
        model.reset_states()

        print("\tEpoch:{}\t Train Loss:{:.8f}\tValidation Loss:{:.5f}\t Train MSE:{:.5f}\t Val MSE:{:.5f}\t Time:{:.5f}".format(epoch, train_loss_mean_epoch.result(), val_metric_loss.result(), train_mse_metric_epoch.result(), val_metric_mse.result() ,time.time()-start_epoch_val  ) )
        if( model_params['model_type_settings']['stochastic']==True ):
            print('\t\tVar_Free_Nrg: {:.5f} '.format(train_loss_var_free_nrg_mean_epoch.result()  ) )

            # endregion
        
        with writer.as_default():
            tf.summary.scalar('Validation Loss', val_metric_loss.result() , step =  epoch )
            try:
                tf.summary.scalar('Validation MSE', val_metric_mse.result(), step=epoch)
                tf.summary.scalar('Validation MSE cond_no_rain', val_mse_cond_no_rain, step=epoch)
            except Exception as e:
                pass
            
        df_training_info = utility.update_checkpoints_epoch(df_training_info, epoch, train_loss_mean_epoch, val_metric_loss, ckpt_manager_epoch, train_params, model_params, train_mse_metric_epoch, val_metric_mse )
        
            
        #region Early iteration Stop Check
        if epoch > ( max( df_training_info.loc[:, 'Epoch'], default=0 ) + train_params['early_stopping_period']) :
            print("Model Early Stopping at EPOCH {}".format(epoch))
            print(df_training_info)
            break
        #endregion

    # endregion

    print("Model Training Finished")

if __name__ == "__main__":
    s_dir = utility.get_script_directory(sys.argv[0])
    args_dict = utility.parse_arguments(s_dir)
    train_params, model_params = utility.load_params_train_model(args_dict)
    
    train_loop(train_params(), model_params )

    

