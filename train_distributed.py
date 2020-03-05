# BNN - Uncertainty SCRNN - ATI Project - PhD Computer Science
#region imports
import os
import sys

import data_generators
import utility
#os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import tensorflow as tf
import horovod.tensorflow as hvd


tf.keras.backend.set_floatx('float16')
tf.keras.backend.set_epsilon(1e-3)


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
#tf.random.set_seed(seed)
# endregion

def train_loop(train_params, model_params): 
    
    # region ----- Defining Model / Optimizer / Losses / Metrics / Records
        # model = models.model_loader(train_params, model_params)
        # if type(model_params) == list:
        #     model_params = model_params[0]
        
        # if tfa==None:
        #     optimizer = tf.keras.optimizers.Adam( learning_rate=1e-4, beta_1=0.1, beta_2=0.99, epsilon=1e-5 )
        # else:
        #     if model_params['model_type_settings']['var_model_type']=="flipout":
        #         model_params['rec_adam_params']['learning_rate'] = 1e-8
        #         model_params['rec_adam_params']['min_lr'] = 1e-9
        #     if model_params['model_type_settings']['location'] == 'region_grid':
        #         total_steps = int(train_params['train_set_size_batches']* np.prod(model_params['region_grid_params']['slides_v_h']) *0.55)
        #     else:
        #         total_steps =int(train_params['train_set_size_batches'] *0.55)
            
        #     radam = tfa.optimizers.RectifiedAdam( **model_params['rec_adam_params'], total_steps=total_steps ) 
        #     optimizer = tfa.optimizers.Lookahead(radam, **model_params['lookahead_params'])
        
        # optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale=tf.mixed_precision.experimental.DynamicLossScale() )
    ##monkey patch so optimizer works with mixed precision
    
    # train_metric_mse_mean_groupbatch = tf.keras.metrics.Mean(name='train_loss_mse_obj')
    # train_metric_mse_mean_epoch = tf.keras.metrics.Mean(name="train_loss_mse_obj_epoch")
    # train_loss_var_free_nrg_mean_groupbatch = tf.keras.metrics.Mean(name='train_loss_var_free_nrg_obj ')
    # train_loss_var_free_nrg_mean_epoch = tf.keras.metrics.Mean(name="train_loss_var_free_nrg_obj_epoch")
    # val_metric_mse_mean = tf.keras.metrics.Mean(name='val_metric_mse_obj')

    # try:
    #     df_training_info = pd.read_csv( "checkpoints/{}/checkpoint_scores.csv".format(utility.model_name_mkr(model_params)),
    #                         header=0, index_col =False   )
    #     print("Recovered checkpoint scores model csv")
    # except Exception as e:
    #     df_training_info = pd.DataFrame(columns=['Epoch','Train_loss_MSE','Val_loss_MSE','Checkpoint_Path', 'Last_Trained_Batch'] ) #key: epoch number #Value: the corresponding loss #TODO: Implement early stopping
    #     print("Did not recover checkpoint scores model csv")
  
    # endregion

    # region ----- Setting up Checkpoints 
        #  (For Epochs)
    # checkpoint_path_epoch = "checkpoints/{}/epoch".format(utility.model_name_mkr(model_params))
    # os.makedirs(checkpoint_path_epoch,exist_ok=True)
        
    # ckpt_epoch = tf.train.Checkpoint(model=model, optimizer=optimizer)
    # ckpt_manager_epoch = tf.train.CheckpointManager(ckpt_epoch, checkpoint_path_epoch, 
    #             max_to_keep=train_params['checkpoints_to_keep_epoch'], keep_checkpoint_every_n_hours=None)    
     
    #     # (For Batches)
    # checkpoint_path_batch = "checkpoints/{}/batch".format(utility.model_name_mkr(model_params))
    # os.makedirs(checkpoint_path_batch,exist_ok=True)
    #     #Create the checkpoint path and the checpoint manager. This will be used to save checkpoints every n epochs
    # ckpt_batch = tf.train.Checkpoint(model=model, optimizer=optimizer)
    # ckpt_manager_batch = tf.train.CheckpointManager(ckpt_batch, checkpoint_path_batch, max_to_keep=train_params['checkpoints_to_keep_batch'], keep_checkpoint_every_n_hours=None)

    #     # restoring checkpoint from last batch if it exists
    # if ckpt_manager_batch.latest_checkpoint: #restoring last checkpoint if it exists
    #     ckpt_batch.restore(ckpt_manager_batch.latest_checkpoint)
    #     print ('Latest checkpoint restored from {}'.format(ckpt_manager_batch.latest_checkpoint  ) )

    # else:
    #     print (' Initializing from scratch')

    # endregion     

    # region --- Setting up training parameters - to be moved to hparams file
    # if model_params['model_type_settings']['location'] == "region_grid":
    #     train_set_size_batches= int(train_params['train_set_size_batches'] * np.prod(model_params['region_grid_params']['slides_v_h']) )
    #     val_set_size_batches = int(train_params['val_set_size_batches'] * np.prod(model_params['region_grid_params']['slides_v_h']))
    # else:
    #     train_set_size_batches= train_params['train_set_size_batches'] 
    #     val_set_size_batches = train_params['val_set_size_batches'] 
    
    # train_batch_reporting_freq = max( int(train_set_size_batches*train_params['dataset_trainval_batch_reporting_freq'] ), 1 )
    # val_batch_reporting_freq = max( int(val_set_size_batches*2*train_params['dataset_trainval_batch_reporting_freq'] ), 1)
    #endregion

    # region Logic for setting up resume location
    # starting_epoch =  int(max( df_training_info['Epoch'], default=0 )) 
    # df_batch_record = df_training_info.loc[ df_training_info['Epoch'] == starting_epoch,'Last_Trained_Batch' ]

    # if( len(df_batch_record)==0 ):
    #     batches_to_skip = 0
    # elif (df_batch_record.iloc[0]==-1 ):
    #     starting_epoch = starting_epoch + 1
    #     batches_to_skip = 0
    # else:
    #     batches_to_skip = int(df_batch_record.iloc[0])
    #     if batches_to_skip >= train_set_size_batches :
    #         starting_epoch = starting_epoch + 1
    #         batches_to_skip = train_set_size_batches
    
    # print("batches to skip", batches_to_skip)

    # endregion

    # region --- Tensorboard Setup
    # os.makedirs("log_tensboard/{}".format(utility.model_name_mkr(model_params)), exist_ok=True ) 
    # writer = tf.summary.create_file_writer( "log_tensboard/{}".format(utility.model_name_mkr(model_params) ) )
    # endregion

    # region ---- Making Datasets
    #TODO: Don't share datasets across workers. Instead Give each worker a specific number of datasets.
    #           - Note -sdc should be a multiple of the number of GPU's available

    # if model_params['model_name'] in ["THST"]:
    #     options = tf.data.Options()
    #     options.experimental_distribute.auto_shard_policy = AutoShardPolicy.OFF

    
    #     # version 2  for iters, will be used when you want to increase the size of training set, while doing stateful training
    #     li_start_days_train = np.arange( train_params['train_start_date'], 
    #                                 train_params['train_start_date'] + np.timedelta64(train_params['lookback_target'],'D'),
    #                                 np.timedelta64(train_params['lookback_target']//train_params['strided_dataset_count'],'D'), dtype='datetime64[D]')[:train_params['strided_dataset_count']]
        
    #     li_start_days_val = np.arange( train_params['val_start_date'], 
    #                                 train_params['val_start_date'] + np.timedelta64(train_params['lookback_target'],'D'),
    #                                 np.timedelta64(train_params['lookback_target']//train_params['strided_dataset_count'],'D'), dtype='datetime64[D]')[:train_params['strided_dataset_count']]

    #     li_ds_trains = [ data_generators.load_data_ati( train_params, model_params, day_to_start_at=sd, data_dir=train_params['data_dir']) for sd in li_start_days_train ]
    #     li_ds_vals = [ data_generators.load_data_ati( train_params, model_params, day_to_start_at=sd, data_dir=train_params['data_dir']) for sd in li_start_days_val ]
        
    #     li_ds_trains = [ _ds.take( math.ceil( train_set_size_batches/train_params['strided_dataset_count'] ) )  if idx==0 
    #                         else _ds.take( train_set_size_batches//train_params['strided_dataset_count'] )  #This ensures that the for loops switch between validation and train sets at the right counts
    #                         for idx,_ds in  enumerate(li_ds_trains) ] #Only the first ds takes the full amount
       
    #     li_ds_vals = [ _ds.take( math.ceil( val_set_size_batches/train_params['strided_dataset_count'] ) )  if idx==0 
    #                 else _ds.take( val_set_size_batches//train_params['strided_dataset_count'] )  #This ensures that the for loops switch between validation and train sets at the right counts
    #                 for idx,_ds in  enumerate(li_ds_vals) ] #Only the first ds takes the full amount

    #     ds_train = li_ds_trains[0]
    #     for idx in range(1,len(li_ds_trains[1:]) ):
    #         ds_train = ds_train.concatenate( li_ds_trains[idx] )

    #     ds_val = li_ds_vals[0]
    #     for idx in range(1,len(li_ds_vals[1:]) ):
    #         ds_val = ds_val.concatenate( li_ds_vals[idx] )

    #     ds_train = strategy.experimental_distribute_dataset(ds_train.repeat( train_params['epochs']-starting_epoch) ).with_options(options)
    #     ds_val = strategy.experimental_distribute_dataset( ds_val.repeat( train_params['epochs']-starting_epoch)  ).with_options(options)
    
    #     iter_train = enumerate(ds_train)
    #     iter_val = enumerate(ds_val)
    #endregion
    ## region --- Train and Val

    
    # region ---  Distributed Params
    # BUFFER_SIZE = len(train_images)
    # BATCH_SIZE_PER_REPLICA = train_params['batch_size'] #TODO: multiple train_set_size_batches in hparameters by strategy.num_replciase
    # GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    
    for epoch in range(starting_epoch, int(train_params['epochs']) ):
        
        #region metrics, loss, dataset, and standardization
        train_metric_mse_mean_groupbatch.reset_states()
        train_loss_var_free_nrg_mean_groupbatch.reset_states()
        train_metric_mse_mean_epoch.reset_states()
        train_loss_var_free_nrg_mean_epoch.reset_states()
        val_metric_mse_mean.reset_states()
                        
        df_training_info = df_training_info.append( { 'Epoch':epoch, 'Last_Trained_Batch':0 }, ignore_index=True )
        
        start_epoch = time.time()
        start_epoch_val = None
        inp_time = None
        start_batch_time = time.time()
        
        #endregion 

        batch=0
        print("\n\nStarting EPOCH {} Batch {}/{}".format(epoch, batches_to_skip+1, train_set_size_batches))
        
        # train_mse_loss_distributed, train_metric_loss_distributed = experimental_run_v2( train_loop, args=() , kwargs= )
        # train_mse_loss =    tf.reduce( tf.distribute.ReduceOp.Mean, train_mse_loss_distributed, axis=0 )
        # train_metric_loss_distributed = tf.reduce( tf.distribute.ReduceOp.Mean, train_metric_loss_distributed, axis=0 )

        model.reset_states()

        val_mse_loss_distributed , val_metric_loss_distributed = experimental_run_v2( val_loop , args=(), kwargs= )
        val_mse_loss = tf.reduce(  tf.distribute.ReduceOp.Mean, val_mse_loss_distributed, axis=0 )
        val_mse_metric_loss = tf.reduce( tf.distribute.ReduceOp.Mean, val_metric_loss_distributed, axis=0 )

        #region Valid
        model.reset_states()
        for batch in range(val_set_size_batches):

            if model_params['model_type_settings']['location'] == 'region_grid':
                idx, (feature, target, mask) = next(iter_val)
            else:
                idx, (feature, target) = next(iter_val)

            if model_params['model_name'] == "THST":
                if model_params['model_type_settings']['location'] == 'region_grid':
                    if( tf.reduce_any( mask[:, :, 6:10, 6:10] )==False ):
                        continue
                else:
                    target, mask = target # (bs, h, w) 

                if model_params['model_type_settings']['stochastic'] == False: #non stochastic version
                    preds = model(tf.cast(feature,tf.float16), training=False )
                    preds = tf.squeeze(preds)

                    if (model_params['model_type_settings']['location']=='region_grid' ): #focusing on centre of square only
                        preds = preds[:, :, 6:10, 6:10]
                        mask = mask[:, :, 6:10, 6:10]
                        target = target[:, :, 6:10, 6:10]

                    preds_filtrd = tf.boolean_mask( preds, mask )
                    target_filtrd = tf.boolean_mask( target, mask )
                    preds_filtrd = utility.standardize_ati( preds_filtrd, train_params['normalization_shift']['rain'], 
                                        train_params['normalization_scales']['rain'], reverse=True)
                    
                    val_metric_mse_mean( tf.reduce_mean(tf.keras.metrics.MSE( target_filtrd , preds_filtrd ) )  )
                
                elif model_params['model_type_settings']['stochastic'] ==True:
                    raise NotImplementedError 
            

            if ( (batch+1) % val_batch_reporting_freq) ==0 or batch+1==val_set_size_batches :
                batches_report_time =  time.time() - start_batch_time
                est_completion_time_seconds = (batches_report_time/train_params['dataset_trainval_batch_reporting_freq']) *( 1 -  ((batch)/val_set_size_batches ) )
                est_completion_time_mins = est_completion_time_seconds/60

                print("\tCompleted Validation Batch:{}/{} \t Time:{:.4f} \tEst Time Left:{:.1f}".format( batch, val_set_size_batches ,batches_report_time,est_completion_time_mins ))
                                            
                start_batch_time = time.time()
                #iter_train = None
                if( batch +1 == val_set_size_batches  ):
                    batches_to_skip = 0
        model.reset_states()

        print("\tEpoch:{}\t Train MSE:{:.8f}\tValidation Loss: MSE:{:.5f}\tTime:{:.5f}".format(epoch, train_metric_mse_mean_epoch.result(), val_metric_mse_mean.result(), time.time()-start_epoch_val  ) )
        if( model_params['model_type_settings']['stochastic']==True ):
            print('\t\tVar_Free_Nrg: {:.5f} '.format(epoch, train_loss_var_free_nrg_mean_epoch.result()  ) )

            # endregion
        
        with writer.as_default():
            tf.summary.scalar('Validation Loss MSE', val_metric_mse_mean.result() , step =  epoch )
        df_training_info = utility.update_checkpoints_epoch(df_training_info, epoch, train_metric_mse_mean_epoch, val_metric_mse_mean, ckpt_manager_epoch, train_params, model_params )
        
            
        #region Early iteration Stop Check
        if epoch > ( max( df_training_info.loc[:, 'Epoch'], default=0 ) + train_params['early_stopping_period']) :
            print("Model Early Stopping at EPOCH {}".format(epoch))
            print(df_training_info)
            break
        #endregion

    # endregion

    print("Model Training Finished")


def main(train_params,model_params):
    # Stateful Training can be done - it will require special tinkering with bactch shaping and ordering example repository her https://github.com/visionscaper/stateful_multi_gpu
    
    mirroredstrategy = tf.distribute.MirroredStrategy()
    # region ----- Defining Model / Optimizer / Losses / Metrics / Records    
    with mirroredstrategy.scope():
        model = models.model_loader(train_params, model_params)
        if type(model_params) == list:
            model_params = model_params[0]
        
        if tfa==None:
            optimizer = tf.keras.optimizers.Adam( learning_rate=1e-4, beta_1=0.1, beta_2=0.99, epsilon=1e-5 )
        else:
            if model_params['model_type_settings']['var_model_type']=="flipout":
                model_params['rec_adam_params']['learning_rate'] = 1e-8
                model_params['rec_adam_params']['min_lr'] = 1e-9
            if model_params['model_type_settings']['location'] == 'region_grid':
                total_steps = int(train_params['train_set_size_batches']* np.prod(model_params['region_grid_params']['slides_v_h']) *0.55)
            else:
                total_steps =int(train_params['train_set_size_batches'] *0.55)
            
            radam = tfa.optimizers.RectifiedAdam( **model_params['rec_adam_params'], total_steps=total_steps ) 
            optimizer = tfa.optimizers.Lookahead(radam, **model_params['lookahead_params'])
        
        optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale=tf.mixed_precision.experimental.DynamicLossScale() )
    
    train_metric_mse_mean_groupbatch = tf.keras.metrics.Mean(name='train_loss_mse_obj')
    train_metric_mse_mean_epoch = tf.keras.metrics.Mean(name="train_loss_mse_obj_epoch")
    train_loss_var_free_nrg_mean_groupbatch = tf.keras.metrics.Mean(name='train_loss_var_free_nrg_obj ')
    train_loss_var_free_nrg_mean_epoch = tf.keras.metrics.Mean(name="train_loss_var_free_nrg_obj_epoch")
    val_metric_mse_mean = tf.keras.metrics.Mean(name='val_metric_mse_obj')

    try:
    df_training_info = pd.read_csv( "checkpoints/{}/checkpoint_scores.csv".format(utility.model_name_mkr(model_params)),
                        header=0, index_col =False   )
    print("Recovered checkpoint scores model csv")
    except Exception as e:
        df_training_info = pd.DataFrame(columns=['Epoch','Train_loss_MSE','Val_loss_MSE','Checkpoint_Path', 'Last_Trained_Batch'] ) #key: epoch number #Value: the corresponding loss #TODO: Implement early stopping
        print("Did not recover checkpoint scores model csv")
    
    #endregion

    # region ----- Setting up Checkpoints 
        #  (For Epochs)
    checkpoint_path_epoch = "checkpoints/{}/epoch".format(utility.model_name_mkr(model_params))
    os.makedirs(checkpoint_path_epoch,exist_ok=True)
    with mirroredstrategy.scope():
        ckpt_epoch = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_manager_epoch = tf.train.CheckpointManager(ckpt_epoch, checkpoint_path_epoch, 
                max_to_keep=train_params['checkpoints_to_keep_epoch'], keep_checkpoint_every_n_hours=None)    
     
        # (For Batches)
    checkpoint_path_batch = "checkpoints/{}/batch".format(utility.model_name_mkr(model_params))
    os.makedirs(checkpoint_path_batch,exist_ok=True)
        #Create the checkpoint path and the checpoint manager. This will be used to save checkpoints every n epochs
    with mirroredstrategy.scope():
        ckpt_batch = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_manager_batch = tf.train.CheckpointManager(ckpt_batch, checkpoint_path_batch, max_to_keep=train_params['checkpoints_to_keep_batch'], keep_checkpoint_every_n_hours=None)

        # restoring checkpoint from last batch if it exists
    if ckpt_manager_batch.latest_checkpoint: #restoring last checkpoint if it exists
        ckpt_batch.restore(ckpt_manager_batch.latest_checkpoint)
        print ('Latest checkpoint restored from {}'.format(ckpt_manager_batch.latest_checkpoint  ) )

    else:
        print (' Initializing from scratch')
    
    # endregion

    # region --- Setting up training parameters - to be moved to hparams file
    if model_params['model_type_settings']['location'] == "region_grid":
        train_set_size_batches= int(train_params['train_set_size_batches'] * np.prod(model_params['region_grid_params']['slides_v_h']) )
        val_set_size_batches = int(train_params['val_set_size_batches'] * np.prod(model_params['region_grid_params']['slides_v_h']))
    else:
        train_set_size_batches= train_params['train_set_size_batches'] 
        val_set_size_batches = train_params['val_set_size_batches'] 
    
    train_batch_reporting_freq = max( int(train_set_size_batches*train_params['dataset_trainval_batch_reporting_freq'] ), 1 )
    val_batch_reporting_freq = max( int(val_set_size_batches*2*train_params['dataset_trainval_batch_reporting_freq'] ), 1)
    #endregion

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
    #endregion

    # region --- Tensorboard Setup
    os.makedirs("log_tensboard/{}".format(utility.model_name_mkr(model_params)), exist_ok=True ) 
    writer = tf.summary.create_file_writer( "log_tensboard/{}".format(utility.model_name_mkr(model_params) ) )
    # endregion

    # region ---  Distributed Params
    BUFFER_SIZE = len(train_images)
    BATCH_SIZE_PER_REPLICA = train_params['batch_size'] #TODO: multiple train_set_size_batches in hparameters by strategy.num_replciase
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    #endregion

    # region ---- Making Datasets
    #TODO: Don't share datasets across workers. Instead Give each worker a specific number of datasets.
    #           - Note -sdc should be a multiple of the number of GPU's available
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.OFF

    li_start_days_train = np.arange( train_params['train_start_date'], 
                                train_params['train_start_date'] + np.timedelta64(train_params['lookback_target'],'D'),
                                np.timedelta64(train_params['lookback_target']//train_params['strided_dataset_count'],'D'), dtype='datetime64[D]')[:train_params['strided_dataset_count']]
    
    li_start_days_val = np.arange( train_params['val_start_date'], 
                                train_params['val_start_date'] + np.timedelta64(train_params['lookback_target'],'D'),
                                np.timedelta64(train_params['lookback_target']//train_params['strided_dataset_count'],'D'), dtype='datetime64[D]')[:train_params['strided_dataset_count']]

    li_ds_trains = [ data_generators.load_data_ati( train_params, model_params, day_to_start_at=sd, data_dir=train_params['data_dir']).unbatch().batch(GLOBAL_BATCH_SIZE) for sd in li_start_days_train ]
    li_ds_vals = [ data_generators.load_data_ati( train_params, model_params, day_to_start_at=sd, data_dir=train_params['data_dir']).unbatch().batch(GLOBAL_BATCH_SIZE) for sd in li_start_days_val ]
    
    li_ds_trains = [ _ds.take( math.ceil( train_set_size_batches/train_params['strided_dataset_count'] ) )  if idx==0 
                        else _ds.take( train_set_size_batches//train_params['strided_dataset_count'] )  #This ensures that the for loops switch between validation and train sets at the right counts
                        for idx,_ds in  enumerate(li_ds_trains) ] #Only the first ds takes the full amount
    
    li_ds_vals = [ _ds.take( math.ceil( val_set_size_batches/train_params['strided_dataset_count'] ) )  if idx==0 
                else _ds.take( val_set_size_batches//train_params['strided_dataset_count'] )  #This ensures that the for loops switch between validation and train sets at the right counts
                for idx,_ds in  enumerate(li_ds_vals) ] #Only the first ds takes the full amount

    ds_train = li_ds_trains[0]
    for idx in range(1,len(li_ds_trains[1:]) ):
        ds_train = ds_train.concatenate( li_ds_trains[idx] )

    ds_val = li_ds_vals[0]
    for idx in range(1,len(li_ds_vals[1:]) ):
        ds_val = ds_val.concatenate( li_ds_vals[idx] )

    ds_train = strategy.experimental_distribute_dataset(ds_train.repeat( train_params['epochs']-starting_epoch) ).with_options(options)
    ds_val = strategy.experimental_distribute_dataset( ds_val.repeat( train_params['epochs']-starting_epoch)  ).with_options(options)
    with mirroredstrategy.scope():
        iter_train = enumerate(ds_train)
        iter_val = enumerate(ds_val)
    #endregion

    for epoch in range(starting_epoch, int(train_params['epochs']) ):
        
        #region metrics, loss, dataset, and standardization
        train_metric_mse_mean_groupbatch.reset_states()
        train_loss_var_free_nrg_mean_groupbatch.reset_states()
        train_metric_mse_mean_epoch.reset_states()
        train_loss_var_free_nrg_mean_epoch.reset_states()
        val_metric_mse_mean.reset_states()
                        
        df_training_info = df_training_info.append( { 'Epoch':epoch, 'Last_Trained_Batch':0 }, ignore_index=True )
        
        start_epoch = time.time()
        start_epoch_val = None
        inp_time = None
        start_batch_time = time.time()
        #endregion 

        batch=0
        print("\n\nStarting EPOCH {} Batch {}/{}".format(epoch, batches_to_skip+1, train_set_size_batches))


        #region Train 
        for batch in range(batches_to_skip,train_set_size_batches):
            idx, (feature, target, mask) = next(iter_train)
            
            per_replica_train_mse_loss, per_replica_train_metric_loss = strategy.expiremental_run_v2( train_step, args=(feature, target, mask) )

            batch_train_mse_loss += mirroredstrategy.reduce( tf.distribute.ReduceOp.Sum, per_replica_train_mse_loss, axis=None ) / GLOBAL_BATCH_SIZE
            batch_train_metric_loss += mirroredstrategy.reduce( tf.distribute.ReduceOp.Sum, per_replica_train_metric_loss, axis=None ) / GLOBAL_BATCH_SIZE
            
            per_replica_loss = strategy.experimental_run_v2( self.train_step, args=(one_batch,) )

            batch_loss += strategy.reduce( tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None) //

            #region Tensorboard Update
            step = batch + (epoch)*train_set_size_batches
            with writer.as_default():
                if( model_params['model_type_settings']['stochastic']==True ):
                    tf.summary.scalar('train_loss_var_free_nrg', var_free_nrg_loss , step =  step )
                    tf.summary.scalar('kl_loss', kl_loss, step=step )
                    tf.summary.scalar('neg_log_likelihood', -log_likelihood, step=step )
                    tf.summary.scalar('train_metric_mse', total_train_mse_loss , step = step )

                    if model_params['model_type_settings']['discrete_continuous'] == True:
                        tf.summary.scalar('train_loss_mse_condrain', loss_mse_condrain, step=step )
                
                elif( model_params['model_type_settings']['stochastic']==False ):
                    tf.summary.scalar('train_mse_loss', total_train_mse_loss , step = step )
                    tf.summary.scalar('train_metric_loss', total_train_metric_loss , step = step )

                for grad, _tensor in zip( gradients, model.trainable_variables):
                    if grad is not None:
                        tf.summary.histogram( "Grad:{}".format( _tensor.name ) , grad, step = step  )
                        tf.summary.histogram( "Weights:{}".format(_tensor.name), _tensor , step = step ) 
            #endregion
        

        #endregion








#@tf.function
def train_step(dist_inputs,train_params, model_params):
    
    def step_fn(inputs):
        with tf.GradientTape(persistent=False) as tape:
        
            if model_params['model_type_settings']['location'] == 'region_grid':
                if( tf.reduce_any( mask[:, :, 6:10, 6:10] ) == False ):
                    continue #TODO: This will have to be changed
            else:
                target, mask = target # (bs, h, w) 

            if (model_params['model_type_settings']['stochastic']==False or model_params['model_type_settings']['var_model_type'] == dropout ): #non stochastic version

                preds = model( tf.cast(feature,tf.float16), train_params['trainable'] )
                preds = tf.squeeze(preds)

                if ( model_params['model_type_settings']['location']=='region_grid' ): #focusing on centre of square only
                    preds = preds[:, :, 6:10, 6:10]
                    mask = mask[:, :, 6:10, 6:10]
                    target = target[:, :, 6:10, 6:10]

                preds_filtrd = tf.boolean_mask( preds, mask )
                target_filtrd = tf.boolean_mask( target, mask )

                preds_filtrd = utility.standardize_ati( preds_filtrd, train_params['normalization_shift']['rain'], 
                                train_params['normalization_scales']['rain'], reverse=True)


                loss_mse = tf.nn.compute_average_loss( tf.keras.losses.MSE(target_filtrd, preds_filtrd), global_batch_size=train_paranms['train_set_size_batches']*strategy.num_replicas_in_sync )
                metric_mse = loss_mse
                scaled_loss = optimizer.get_scaled_loss(loss_mse + sum(model.losses) )
                
                scaled_gradients = tape.gradient( scaled_loss, model.trainable_variables )
                gradients = optimizer.get_unscaled_gradients(scaled_gradients)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            gc.collect()
        #region Tensorboard Update
        step = batch + (epoch)*train_set_size_batches
        with writer.as_default():
            if( model_params['model_type_settings']['stochastic']==True ):
                tf.summary.scalar('train_loss_var_free_nrg', var_free_nrg_loss , step =  step )
                tf.summary.scalar('kl_loss', kl_loss, step=step )
                tf.summary.scalar('neg_log_likelihood', -log_likelihood, step=step )
                tf.summary.scalar('train_metric_mse', metric_mse , step = step )

                if model_params['model_type_settings']['discrete_continuous'] == True:
                    tf.summary.scalar('train_loss_mse_condrain', loss_mse_condrain, step=step )
            
            elif( model_params['model_type_settings']['stochastic']==False ):
                tf.summary.scalar('train_loss_mse', loss_mse , step = step )
                tf.summary.scalar('train_metric_mse', metric_mse , step = step )


            for grad, _tensor in zip( gradients, model.trainable_variables):
                if grad is not None:
                    tf.summary.histogram( "Grad:{}".format( _tensor.name ) , grad, step = step  )
                    tf.summary.histogram( "Weights:{}".format(_tensor.name), _tensor , step = step ) 
        #endregion

        #region training Reporting and Metrics updates
        if( model_params['model_type_settings']['stochastic']==True ):
            train_loss_var_free_nrg_mean_groupbatch( var_free_nrg_loss )
            train_loss_var_free_nrg_mean_epoch( var_free_nrg_loss )
            train_metric_mse_mean_groupbatch( metric_mse )
            train_metric_mse_mean_epoch( metric_mse )

        elif( model_params['model_type_settings']['stochastic']==False ):
            train_metric_mse_mean_groupbatch( metric_mse )
            train_metric_mse_mean_epoch( metric_mse )
                                    
        ckpt_manager_batch.save()
        if( (batch+1)%train_batch_reporting_freq==0 or batch+1 == train_set_size_batches):
            batches_report_time =  time.time() - start_batch_time

            est_completion_time_seconds = (batches_report_time/train_params['dataset_trainval_batch_reporting_freq']) * (train_set_size_batches - batch)/train_set_size_batches
            est_completion_time_mins = est_completion_time_seconds/60

            print("\tBatch:{}/{}\tTrain MSE Loss: {:.8f} \t Batch Time:{:.4f}\tEpoch mins left:{:.1f}".format(batch, train_set_size_batches, train_metric_mse_mean_groupbatch.result(), batches_report_time, est_completion_time_mins ) )
            train_metric_mse_mean_groupbatch.reset_states()
            start_batch_time = time.time()

            # Updating record of the last batch to be operated on in training epoch
        df_training_info.loc[ ( df_training_info['Epoch']==epoch) , ['Last_Trained_Batch'] ] = batch
        df_training_info.to_csv( path_or_buf="checkpoints/{}/checkpoint_scores.csv".format(utility.model_name_mkr(model_params)), header=True, index=False )

        if batch in np.arange(math.ceil( train_set_size_batches/train_params['strided_dataset_count'] ), train_set_size_batches,  train_set_size_batches//train_params['strided_dataset_count'] ).tolist() :
            model.reset_states()
    
    start_epoch_val = time.time()
    start_batch_time = time.time()

    print("\n\tStarting Validation")
    
    #endregion
    # endregion

    return mse_loss, train_metric


def val_loop():

    #region Valid
    model.reset_states()
    for batch in range(val_set_size_batches):

        if model_params['model_type_settings']['location'] == 'region_grid':
            idx, (feature, target, mask) = next(iter_val)
        else:
            idx, (feature, target) = next(iter_val)

        if model_params['model_name'] == "THST":
            if model_params['model_type_settings']['location'] == 'region_grid':
                if( tf.reduce_any( mask[:, :, 6:10, 6:10] )==False ):
                    continue
            else:
                target, mask = target # (bs, h, w) 

            if model_params['model_type_settings']['stochastic'] == False: #non stochastic version
                preds = model(tf.cast(feature,tf.float16), training=False )
                preds = tf.squeeze(preds)

                if (model_params['model_type_settings']['location']=='region_grid' ): #focusing on centre of square only
                    preds = preds[:, :, 6:10, 6:10]
                    mask = mask[:, :, 6:10, 6:10]
                    target = target[:, :, 6:10, 6:10]

                preds_filtrd = tf.boolean_mask( preds, mask )
                target_filtrd = tf.boolean_mask( target, mask )
                preds_filtrd = utility.standardize_ati( preds_filtrd, train_params['normalization_shift']['rain'], 
                                    train_params['normalization_scales']['rain'], reverse=True)
                
                val_metric_mse_mean( tf.reduce_mean(tf.keras.metrics.MSE( target_filtrd , preds_filtrd ) )  )
            
            elif model_params['model_type_settings']['stochastic'] ==True:
                raise NotImplementedError 
        

        if ( (batch+1) % val_batch_reporting_freq) ==0 or batch+1==val_set_size_batches :
            batches_report_time =  time.time() - start_batch_time
            est_completion_time_seconds = (batches_report_time/train_params['dataset_trainval_batch_reporting_freq']) *( 1 -  ((batch)/val_set_size_batches ) )
            est_completion_time_mins = est_completion_time_seconds/60

            print("\tCompleted Validation Batch:{}/{} \t Time:{:.4f} \tEst Time Left:{:.1f}".format( batch, val_set_size_batches ,batches_report_time,est_completion_time_mins ))
                                        
            start_batch_time = time.time()
            #iter_train = None
            if( batch +1 == val_set_size_batches  ):
                batches_to_skip = 0
    # endregion


    return mse_loss, val_metric


if __name__ == "__main__":
    s_dir = utility.get_script_directory(sys.argv[0])
    args_dict = utility.parse_arguments(s_dir)
    train_params, model_params = utility.load_params_train_model(args_dict)
    
    train_loop(train_params(), model_params )

    

