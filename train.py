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

#DType.is_compatible_with = is_compatible_with
from tensorflow.keras.mixed_precision import experimental as mixed_precision
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
    model = models.model_loader(train_params, model_params)
    if type(model_params) == list:
        model_params = model_params[0]
    
    if tfa==None:
        optimizer = tf.keras.optimizers.Adam( learning_rate=1e-4, beta_1=0.1, beta_2=0.99, epsilon=1e-5 )
    else:
        radam = tfa.optimizers.RectifiedAdam( **model_params['rec_adam_params'], total_steps=int(train_params['train_set_size_batches']*0.55) )
        optimizer = tfa.optimizers.Lookahead(radam, **model_params['lookahead_params'])
    
    optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic' )
    ##monkey patch so optimizer works with mixed precision
    
    
    
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

    # region --- Setting up training parameters - to be moved to hparams file
    train_set_size_batches= train_params['train_set_size_batches']
    val_set_size_batches = train_params['val_set_size_batches'] 
    
    train_batch_reporting_freq = max( int(train_set_size_batches*train_params['dataset_trainval_batch_reporting_freq'] ), 1 )
    val_batch_reporting_freq = max( int(val_set_size_batches*2*train_params['dataset_trainval_batch_reporting_freq'] ), 1)
    #endregion

    # region Logic for setting up resume location
    starting_epoch =  int(max( df_training_info['Epoch'], default=1 )) 
    df_batch_record = df_training_info.loc[ df_training_info['Epoch'] == starting_epoch,'Last_Trained_Batch' ]

    if( len(df_batch_record)==0 or df_batch_record.iloc[0]==-1 ):
        batches_to_skip = 0
    else:
        batches_to_skip = int(df_batch_record.iloc[0])
        if batches_to_skip == train_params['train_set_size_batches'] :
            starting_epoch = starting_epoch + 1
            batches_to_skip = 0
    
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

    elif model_params['model_name'] == "THST":
        ds_train = data_generators.load_data_ati( train_params, model_params, day_to_start_at=train_params['train_start_date'], data_dir=train_params['data_dir'])
        ds_val = data_generators.load_data_ati( train_params, model_params, day_to_start_at=train_params['val_start_date'], data_dir=train_params['data_dir'] )
    
    elif model_params['model_name'] == "SimpleLSTM":
        ds_train = data_generators.load_data_ati( train_params, model_params, day_to_start_at=train_params['train_start_date'], data_dir=train_params['data_dir'])
        ds_val = data_generators.load_data_ati( train_params, model_params, day_to_start_at=train_params['val_start_date'], data_dir=train_params['data_dir'] )

    elif model_params['model_name'] == "SimpleConvLSTM":
        raise NotImplementedError
    
    ds_train = ds_train.take(train_params['train_set_size_batches']).repeat(train_params['epochs']-starting_epoch)
    ds_val = ds_val.take(train_params['val_set_size_batches']).repeat(train_params['epochs']-starting_epoch)
    ds_train = ds_train.skip(batches_to_skip)
    #ds_val = ds_val.skip(batches_to_skip)
    iter_train = enumerate(ds_train)
    iter_val = enumerate(ds_val)
    # endregion

    # region --- Train and Val

    if model_params['model_type_settings']['var_model_type'] in ['horseshoefactorized','horseshoestructured'] :
        tf.config.experimental_run_functions_eagerly(True)

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
                        preds_scale = tf.where(tf.equal(preds_scale,0.0), .5, preds_scale)

                        if(model_params['model_type_settings']['distr_type']=="Normal" ):
                            preds_distribution = tfd.Normal( loc=preds_mean, scale= preds_scale)
                            
                            _1 = tf.where(train_params['bool_water_mask'], target, 1e-2) 
                            _2 = preds_distribution.log_prob( _1)
                            _3 = tf.boolean_mask( _2, train_params['bool_water_mask'],axis=1 )
                            log_likelihood = tf.reduce_mean( _3)
                                #This represents the expected log_likelihood corresponding to each target y_i in the mini batch

                        kl_loss_weight = utility.kl_loss_weighting_scheme(train_set_size_batches) #TODO: Implement scheme where kl loss increases during training
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
                            preds_distribution_condrain = tfd.Normal( loc=_preds_cond_rain_mean, scale= tf.where( _preds_cond_rain_scale==0, 1e-7, _preds_cond_rain_scale  ) )

                        elif(model_params['model_type_settings']['distr_type'] == "LogNormal" ):
                            epsilon = tf.random.uniform( preds_stacked.shape.as_list(),minval=1e-10,maxval=1e-7 )
                            preds_stacked_adj = tf.where( preds_stacked==0,epsilon,preds_stacked )
                            log_vals = tf.math.log( preds_stacked_adj)
                            log_distr_mean = tf.math.reduce_mean( log_vals, axis=-1)
                            log_distr_std = tf.math.reduce_std(log_vals, axis=-1)
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

                        kl_loss_weight = utility.kl_loss_weighting_scheme(train_set_size_batches) 
                        kl_loss = tf.cast(tf.math.reduce_sum( model.losses ) * kl_loss_weight * (1/model_params['model_type_settings']['stochastic_f_pass'] ), tf.float32)  #This KL-loss is already normalized against the number of samples of weights drawn #TODO: Later implement your own Adam type method to determine this

                        var_free_nrg_loss = kl_loss  - log_likelihood
                        l = var_free_nrg_loss
                        

                        loss_mse_condrain = tf.reduce_mean( tf.keras.losses.MSE( _target_cond_rain , _preds_cond_rain_mean) )
                    #endregion

                    target_filtrd = tf.reshape( tf.boolean_mask(  target , train_params['bool_water_mask'], axis=1 ), [train_params['batch_size'], -1] )
                    preds_mean_filtrd = tf.reshape( tf.boolean_mask( preds_mean, train_params['bool_water_mask'],axis=1 ), [train_params['batch_size'], -1] )

                    metric_mse = tf.reduce_mean( tf.keras.losses.MSE( target_filtrd , preds_mean_filtrd)  )

                    scaled_loss = optimizer.get_scaled_loss(l)
                    scaled_gradients = tape.gradient( scaled_loss, model.trainable_variables )
                    gradients = optimizer.get_unscaled_gradients(scaled_gradients) 
                    gc.collect()
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    
                elif( model_params['model_name'] == "THST"):
                    if (model_params['model_type_settings']['stochastic']==False): #non stochastic version
                        target, mask = target

                        preds = model( tf.cast(feature,tf.float16), tape=tape )
                        preds = tf.squeeze(preds)

                        preds_filtrd = tf.boolean_mask( preds, mask )
                        target_filtrd = tf.boolean_mask( target, mask )

                        loss_mse = tf.keras.losses.MSE(target_filtrd, preds_filtrd) 
                        metric_mse = loss_mse
                        scaled_loss_mse = optimizer.get_scaled_loss(loss_mse)

                    elif (model_params['model_type_settings']['stochastic']==True) :
                        raise NotImplementedError
                
                    scaled_gradients = tape.gradient( scaled_loss_mse, model.trainable_variables )
                    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                elif( model_params['model_name'] == "SimpleLSTM"):
                    if( model_params['model_type_settings']['stochastic']==False):
                        target, mask = target # (bs, seq_len)

                        preds = model( tf.cast(feature,tf.float16) ) #( bs, tar_seq_len)
                        preds = tf.squeeze(preds)

                        preds_filtrd = tf.boolean_mask( preds, mask )
                        target_filtrd = tf.boolean_mask( target, mask )

                        loss_mse = tf.keras.losses.MSE(target_filtrd, preds_filtrd) 
                        metric_mse = loss_mse
                        scaled_loss_mse = optimizer.get_scaled_loss(loss_mse)

                    elif(model_params['model_type_settings']['stochatic']==True):
                        raise NotImplementedError
                    
                    scaled_gradients = tape.gradient( scaled_loss_mse, model.trainable_variables )
                    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                elif (model_params['model_name'] == "SimpleConvLSTM"):
                    raise NotImplementedError

                gc.collect()
                            
            #region Tensorboard Update
            step = batch + (epoch-1)*train_set_size_batches
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

                print("\tBatch:{}/{}\tTrain MSE Loss: {:.5f} \t Batch Time:{:.4f}\tEpoch mins left:{:.1f}".format(batch, train_set_size_batches, train_metric_mse_mean_groupbatch.result(), batches_report_time, est_completion_time_mins ) )
                train_metric_mse_mean_groupbatch.reset_states()
                start_batch_time = time.time()

                # Updating record of the last batch to be operated on in training epoch
            df_training_info.loc[ ( df_training_info['Epoch']==epoch) , ['Last_Trained_Batch'] ] = batch
            df_training_info.to_csv( path_or_buf="checkpoints/{}/checkpoint_scores.csv".format(utility.model_name_mkr(model_params)), header=True, index=False )
        
        print("\nStarting Validation")
        start_epoch_val = time.time()
        start_batch_time = time.time()
        if( model_params['model_type_settings']['stochastic']==True ):
            print('EPOCH {}:\tVar_Free_Nrg: {:.5f} \tMSE: {:.5f}\tTime: {:.2f}'.format(epoch, train_loss_var_free_nrg_mean_epoch.result() ,train_metric_mse_mean_epoch.result(), (time.time()-start_epoch ) ) )
        else:
            print('EPOCH {}:\tMSE: {:.3f}\tTime: {:.2f}'.format(epoch ,train_metric_mse_mean_epoch.result(), (time.time()-start_epoch ) ) )
            # endregion
        model.reset_states()
        #endregion

        #region Valid
        for batch in range(val_set_size_batches):
            idx, (feature, target) = next(iter_val)

            if model_params['model_name'] == "DeepSD":
                if model_params['model_type_settings']['stochastic'] ==True: #non stochastic version

                    preds = model( feature, training=False )
                    preds = utility.water_mask( tf.squeeze(preds), train_params['bool_water_mask'])
                    
                    target_filtrd = tf.reshape( tf.boolean_mask(  target , train_params['bool_water_mask'], axis=1 ), [train_params['batch_size'], -1] )
                    preds_filtrd = tf.reshape( tf.boolean_mask( preds, train_params['bool_water_mask'],axis=1 ), [train_params['batch_size'], -1] )
                    val_metric_mse_mean( tf.reduce_mean( tf.keras.metrics.MSE( target_filtrd , preds_filtrd ) )  ) #TODO: Ensure that both preds and target are reshaped prior 
            
            elif model_params['model_name'] == "THST":
                if model_params['model_type_settings']['stochastic'] ==False: #non stochastic version
                    target, mask = target
                    preds = model(tf.cast(feature,tf.float16) )
                    preds = tf.squeeze(preds)

                    preds_filtrd = tf.boolean_mask( preds, mask )
                    target_filtrd = tf.boolean_mask( target, mask )

                    val_metric_mse_mean( tf.reduce_mean(tf.keras.metrics.MSE( target_filtrd , preds_filtrd ) )  )
            
            elif model_params['model_name'] == "SimpleLSTM":
                if model_params['model_type_settings']['stochastic'] == False:
                    target, mask = target
                    preds = model(tf.cast(feature,tf.float16) )
                    preds = tf.squeeze(preds)

                    preds_filtrd = tf.boolean_mask( preds, mask )
                    target_filtrd = tf.boolean_mask( target, mask )

                    val_metric_mse_mean( tf.reduce_mean(tf.keras.metrics.MSE( target_filtrd , preds_filtrd ) )  )

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

        print("Epoch:{}\t Train MSE:{:.5f}\tValidation Loss: MSE:{:.5f}\tTime:{:.5f}".format(epoch, train_metric_mse_mean_epoch.result(), val_metric_mse_mean.result(), time.time()-start_epoch_val  ) )
        with writer.as_default():
            tf.summary.scalar('Validation Loss MSE', val_metric_mse_mean.result() , step =  epoch )
        df_training_info = utility.update_checkpoints_epoch(df_training_info, epoch, train_metric_mse_mean_epoch, val_metric_mse_mean, ckpt_manager_epoch, train_params, model_params )
        # endregion
            
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

    

