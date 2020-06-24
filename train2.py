from netCDF4 import Dataset, num2date 
import data_generators
import argparse
import ast
import gc
import logging
import math
import os
import re
import sys
import time

#os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import psutil
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training.tracking import data_structures
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl
try:
    import tensorflow_addons as tfa
except Exception as e:
    tfa = None

import custom_losses as cl

import hparameters
import models
import utility

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.keras.backend.set_floatx('float16')
tf.keras.backend.set_epsilon(1e-3)

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
    Monkey patch: incompatibility issues between tfa.optimizers and mixed precision training
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
tf.DType.is_compatible_with = is_compatible_with

class TrainTrueNet():
    """Handles the Training of the TRUNET model
    """    
    def __init__(self, t_params, m_params): 
        """Train the TRU_NET Model
            Initialize_scheme_**: Handles the data specific initializations initializing 
            train_model : Handles the iterative training
        """
        self.t_params = t_params
        self.m_params = m_params
        
    def initialize_scheme_era5Eobs(self):
        """Initialization scheme for the ERA5 and E-OBS dataset
        Args:
        """        
        # region ---- Variables to initialize
        era5_eobs = data_generators.Era5_Eobs( self.t_params, self.m_params )

        self.t_params['train_batches'] = int(self.t_params['train_batches'] * era5_eobs.loc_count)
        self.t_params['val_batches'] = int(self.t_params['val_batches'] * era5_eobs.loc_count)

        self.train_batch_report_freq = max( int(self.t_params['train_batches']*self.t_params['reporting_freq']), 1)
        self.val_batch_report_freq = max( int(self.t_params['val_batches']*self.t_params['reporting_freq']), 1)
        #endregion

        # region ---- Restoring/Creating New Training Records and Restoring training progress
        try:
            self.df_training_info = pd.read_csv( "checkpoints/{}/checkpoint_scores.csv".format(utility.model_name_mkr(m_params,t_params=self.t_params)), header=0, index_col=False) 
            self.start_epoch =  int(max(self.df_training_info['Epoch'], default=0)) 
            last_batch = int( self.df_training_info.loc[self.df_training_info['Epoch']==self.start_epoch,'Last_Trained_Batch'].iloc[0] )
            if(last_batch in [-1, self.t_params['train_batches']] ):
                self.start_epoch = self.start_epoch + 1
                self.batches_to_skip = 0
            else:
                self.batches_to_skip = last_batch
            print("Recovered training records")

        except FileNotFoundError as e:
            self.df_training_info = pd.DataFrame(columns=['Epoch','Train_loss','Val_loss','Checkpoint_Path', 'Last_Trained_Batch'] ) #key: epoch number #Value: the corresponding loss #TODO: Implement early stopping
            self.batches_to_skip = 0
            self.start_epoch = 0
            print("Did not recover training records. Starting from scratch")
        # endregion

        # region ---- Defining Model / Optimizer / Losses / Metrics / Records / Checkpoints / Tensorboard 
        #model
        self.model = models.model_loader( self.t_params, m_params )
            
        #Optimizer
        optimizer = tfa.optimizers.RectifiedAdam( **m_params['rec_adam_params'], total_steps=self.t_params['train_batches']*40 ) 
        
        #optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=1, slow_step_size=0.99 )
        self.optimizer = mixed_precision.LossScaleOptimizer( optimizer, loss_scale=tf.mixed_precision.experimental.DynamicLossScale() )
        
        #Losses and Metrics
            # These objects will aggregate losses and metrics across batches
        self.loss_agg_batch = tf.keras.metrics.Mean(name='loss_agg_batch')
        self.loss_agg_epoch = tf.keras.metrics.Mean(name="loss_agg_epoch")
        self.mse_agg_epoch = tf.keras.metrics.Mean(name='self.mse_agg_epoch')
        
        self.loss_agg_val = tf.keras.metrics.Mean(name='loss_agg_val')
        self.mse_agg_val = tf.keras.metrics.Mean(name='mse_agg_val')    
            
        #checkpoints  (For Epochs)
        checkpoint_path_epoch = "checkpoints/{}/epoch".format(utility.model_name_mkr(m_params,t_params=self.t_params ))
        os.makedirs(checkpoint_path_epoch,exist_ok=True)
        ckpt_epoch = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.ckpt_mngr_epoch = tf.train.CheckpointManager(ckpt_epoch, checkpoint_path_epoch, max_to_keep=self.t_params['checkpoints_to_keep'], keep_checkpoint_every_n_hours=None)    
        
        #checkpoints (For Batches)
        checkpoint_path_batch = "checkpoints/{}/batch".format(utility.model_name_mkr(m_params,t_params=self.t_params))
        os.makedirs(checkpoint_path_batch,exist_ok=True)
        ckpt_batch = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)#, optimizer=optimizer)
        self.ckpt_mngr_batch = tf.train.CheckpointManager(ckpt_batch, checkpoint_path_batch, max_to_keep=self.t_params['checkpoints_to_keep'], keep_checkpoint_every_n_hours=None)

        if self.ckpt_mngr_batch.latest_checkpoint: #restoring last checkpoint if it exists
            # compat: Initializing model and optimizer before restoring from checkpoint
            inp_shape = [t_params['batch_size'], t_params['lookback_feature']] + m_params['region_grid_params']['outer_box_dims'] + [len(t_params['vars_for_feature'])]
            inp_shape1 = [t_params['batch_size'], t_params['lookback_target']] + m_params['region_grid_params']['outer_box_dims'] 

            _ = self.model( tf.zeros( inp_shape ,dtype=tf.float16), self.t_params['trainable'] ) #( bs, tar_seq_len, h, w)
            gradients = [ tf.zeros_like(t_var, dtype=tf.float32 ) for t_var in self.model.trainable_variables  ]
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            ckpt_batch.restore(self.ckpt_mngr_batch.latest_checkpoint).assert_consumed()              
        else:
            print (' Initializing model from scratch')
        
        #Tensorboard
        os.makedirs("log_tensboard/{}".format(utility.model_name_mkr(m_params, t_params=self.t_params)), exist_ok=True ) 
        self.writer = tf.summary.create_file_writer( "log_tensboard/{}".format(utility.model_name_mkr(m_params,t_params=self.t_params) ) )
        # endregion
        
        # region ---- Making Datasets
        #era5_eobs = data_generators.Era5_Eobs( self.t_params, self.m_params )
        ds_train, _  = era5_eobs.load_data_era5eobs( self.t_params['train_batches'], self.t_params['start_date'] )
        ds_val, _ = era5_eobs.load_data_era5eobs( self.t_params['val_batches'], self.t_params['val_start_date'] )
        
        # setting naming system for cache 
        cache_suffix = utility.cache_suffix_mkr( m_params, self.t_params )
        ds_train = ds_train.cache('Data/data_cache/train'+cache_suffix ) 
        ds_val = ds_val.cache('Data/data_cache/val'+cache_suffix )
        
        # preparing iterators for train and validation
        # data loading schemes based on ram limitations
        if psutil.virtual_memory()[0] / 1e9 <= 38.0 : 
            #Data Loading Scheme 1 - Version that works on low memeory devices e.g. under 30 GB RAM
            ds_train_val = ds_train.concatenate(ds_val).repeat(self.t_params['epochs']-self.start_epoch)
            self.ds_train_val = ds_train_val.skip(self.batches_to_skip)
            self.iter_val_train = enumerate(self.ds_train_val)
            self.iter_train = self.iter_val_train
            self.iter_val = self.iter_val_train
        else:
            #Data Loading Scheme 2 - Version that ensures validation and train set are well defined 
            self.ds_train = ds_train.repeat(self.t_params['epochs']-self.start_epoch)
            self.ds_val = ds_val.repeat(self.t_params['epochs']-self.start_epoch)
            self.ds_train = self.ds_train.skip(self.batches_to_skip)
            self.iter_train = enumerate(self.ds_train)
            self.iter_val = enumerate(self.ds_val)

        #endregion


    def train_model(self):

        bounds = cl.central_region_bounds(self.m_params['region_grid_params']) #bounds for central region which we evaluate on 
        
        for epoch in range(self.start_epoch, int(self.t_params['epochs']) ):
            
            #region resetting metrics, losses and records
            self.loss_agg_batch.reset_states()
            self.loss_agg_epoch.reset_states()
            self.mse_agg_epoch.reset_states()
            
            self.loss_agg_val.reset_states()
            self.mse_agg_val.reset_states()
            self.df_training_info = self.df_training_info.append( { 'Epoch':epoch, 'Last_Trained_Batch':0 }, ignore_index=True )
            
            start_epoch = time.time()
            start_epoch_val = None
            inp_time = None
            start_batch_group_time = time.time()
            
            batch=0
            
            if( epoch==self.start_epoch  ):
                self.batches_to_skip = 0

            print("\n\nStarting EPOCH {} Batch {}/{}".format(epoch, self.batches_to_skip+1, self.t_params['train_batches']))
            #endregion 
            
            #region Training
            for batch in range(self.batches_to_skip+1, self.t_params['train_batches']+1):
                
                step = batch + (epoch)*self.t_params['train_batches']
                # get next set of training datums
                idx, (feature, target, mask) = next(self.iter_train)

                with tf.GradientTape(persistent=False) as tape:
                    
                    #if region in datum is completely masked then skip to next training datum
                    if( tf.reduce_any( cl.extract_central_region(mask, bounds) )==False ):
                        continue

                    if( self.m_params['model_type_settings']['stochastic']==False):

                        if self.m_params['model_type_settings']['discrete_continuous'] == False:
                            
                            #making predictions
                            preds = self.model( tf.cast(feature, tf.float16), self.t_params['trainable'] ) #( bs, tar_seq_len, h, w)
                            preds = tf.squeeze( preds,axis=[-1] )

                            #Extracting central region of predictions
                            # preds = preds[:, :, 6:10, 6:10]
                            # mask = mask[:, :, 6:10, 6:10]
                            # target = target[:, :, 6:10, 6:10]

                            preds   = cl.extract_central_region(preds, bounds)
                            mask    = cl.extract_central_region(mask, bounds)
                            target  = cl.extract_central_region(target, bounds)

                            #Applying mask
                            preds_masked = tf.boolean_mask( preds, mask )
                            target_masked = tf.boolean_mask( target, mask ) 

                            # reversing standardization
                            preds_masked = utility.standardize_ati( preds_masked, self.t_params['normalization_shift']['rain'], self.t_params['normalization_scales']['rain'], reverse=True)

                            # getting losses for records and/or optimizer
                            metric_mse = cl.mse(target_masked, preds_masked) 
                            loss_to_optimize = metric_mse
                                                
                        elif self.m_params['model_type_settings']['discrete_continuous'] == True:
                            
                            # Producing predictions - rain value and prob of rain
                            preds   = self.model( tf.cast(feature,tf.float16), self.t_params['trainable'] ) # ( bs, seq_len, h, w, 1)
                            preds   = tf.squeeze(preds, axis=[-1])
                            preds, probs = tf.unstack(preds, axis=0) 

                            # Focusing on central region of predictions
                            # preds   = preds[:,    :, 6:10, 6:10]
                            # probs   = probs[:,    :, 6:10, 6:10]
                            # mask    = mask[:,     :, 6:10, 6:10]
                            # target  = target[:,  :, 6:10, 6:10]

                            preds   = cl.extract_central_region(preds, bounds)
                            probs   = cl.extract_central_region(probs, bounds)
                            mask    = cl.extract_central_region(mask, bounds)
                            target  = cl.extract_central_region(target, bounds)
                            
                            # Masking out undesired areas
                            preds_masked    = tf.boolean_mask(preds, mask )
                            probs_masked    = tf.boolean_mask(probs, mask ) 
                            target_masked   = tf.boolean_mask(target, mask )
                            
                            # Reverising standardization of predictions 
                            preds_masked    = utility.standardize_ati( preds_masked, self.t_params['normalization_shift']['rain'], 
                                                                    self.t_params['normalization_scales']['rain'], reverse=True) 
                                                                    
                            # Getting true and predicted labels for whether or not it rained [ 1 if if did rain, 0 if it did not rain]
                            labels_true = tf.where( target_masked > 0.5, 1.0, 0.0 )
                            labels_pred = probs_masked 

                            all_count = tf.size( labels_true, out_type=tf.int64)

                            # Seperating predictions for the days it did rain and the days it did not rain
                            bool_rain = (labels_true==1.0) 

                            preds_cond_rain         = tf.boolean_mask( preds_masked, bool_rain )                        
                            probs_cond_rain         = tf.boolean_mask( probs_masked, bool_rain )                        
                            target_cond_rain        = tf.boolean_mask( target_masked, bool_rain )                   
                            
                            # region Calculating Losses
                            loss_to_optimize = 0

                            #NOTE: Error on vandal equation 14 last line, he forgets to put the logs and forgets the negative symbol so correct in yours
                            loss_to_optimize += tf.reduce_mean( 
                                            tf.keras.backend.binary_crossentropy( labels_true, labels_pred, from_logits=False) ) 

                            if self.m_params['model_type_settings']['distr_type'] == 'Normal': 
                                # DC Normal
                                loss_to_optimize += cl.mse( target_cond_rain, preds_cond_rain, all_count )
                            
                            elif self.m_params['model_type_settings']['distr_type'] == 'LogNormal':    
                                #DC LogNormal                                                             
                                loss_to_optimize += cl.log_mse( target_cond_rain, preds_cond_rain, all_count)                                                         
                            
                            # Calculated mse for reporting
                            metric_mse  = cl.mse( target_masked, cl.cond_rain(preds_masked, probs_masked) )   
                                # To calc metric_mse under DC settings we assume that pred_rain =0 if pred_prob<0.5 
                            # endregion

                    elif(self.m_params['model_type_settings']['stochastic']==True):
                        raise NotImplementedError

                    # gradient Update step - mixed precision training
                    scaled_loss = self.optimizer.get_scaled_loss(loss_to_optimize+tf.math.add_n(self.model.losses) )
                    scaled_gradients = tape.gradient( scaled_loss, self.model.trainable_variables )
                    gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
                    gradients, _ = tf.clip_by_global_norm( gradients, 4 )
                    
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    gc.collect()
                
                #region training Reporting and Metrics updates

                # Tensorboard          
                li_losses = [loss_to_optimize, metric_mse]
                li_names = ['train_loss','train_mse']
                utility.tensorboard_record( self.writer.as_default(), li_losses, li_names, step, gradients, self.model.trainable_variables )

                # Metrics (batchwise, epoch)            
                self.loss_agg_batch( loss_to_optimize )
                self.loss_agg_epoch( loss_to_optimize )
                self.mse_agg_epoch( metric_mse )
                                        
                self.ckpt_mngr_batch.save()

                if( batch % self.train_batch_report_freq==0 or batch == self.t_params['train_batches']):
                    batch_group_time =  time.time() - start_batch_group_time
                    est_completion_time_seconds = (batch_group_time/self.t_params['reporting_freq']) * (1 - batch/self.t_params['train_batches'])
                    est_completion_time_mins = est_completion_time_seconds/60

                    print("\t\tBatch:{}/{}\tTrain Loss: {:.8f} \t Batch Time:{:.4f}\tEpoch mins left:{:.1f}".format(batch, self.t_params['train_batches'], self.loss_agg_batch.result(), batch_group_time, est_completion_time_mins ) )
                    
                    # resetting time and losses
                    self.loss_agg_batch.reset_states()
                    start_batch_group_time = time.time()

                    # Updating record of the last batch to be operated on in training epoch
                    self.df_training_info.loc[ ( self.df_training_info['Epoch']==epoch) , ['Last_Trained_Batch'] ] = batch
                    self.df_training_info.to_csv( path_or_buf="checkpoints/{}/checkpoint_scores.csv".format(utility.model_name_mkr(self.m_params,t_params=self.t_params)), header=True, index=False )
                    #endregion
            #endregion

            print("\tStarting Validation")
            start_epoch_val = time.time()
            start_batch_group_time = time.time()
            self.model.reset_states()
            
            #region Validation Loops
            for batch in range(1, self.t_params['val_batches']+1):

                idx, (feature, target, mask) = next(self.iter_val)
                
                #Skipping any completely masked data     
                if tf.reduce_any( mask[:, :, 6:10, 6:10] )==False  :
                    continue

                if self.m_params['model_type_settings']['stochastic'] == False:

                    if self.m_params['model_type_settings']['discrete_continuous'] == False:
                        #Retreive predictions
                        preds = self.model(tf.cast(feature,tf.float16), training=False )
                        preds = tf.squeeze(preds)

                        #selecting central region for evaluation                
                        preds = preds[:, :, 6:10, 6:10]
                        mask = mask[:, :, 6:10, 6:10]
                        target = target[:, :, 6:10, 6:10]
                        
                        #applying masks
                        preds_masked = tf.boolean_mask( preds, mask )
                        target_masked = tf.boolean_mask( target, mask )
                        preds_masked = utility.standardize_ati( preds_masked, self.t_params['normalization_shift']['rain'], 
                                                                self.t_params['normalization_scales']['rain'], reverse=True)
                        #updating losses
                        loss = cl.mse( target_masked , preds_masked ) 
                        self.loss_agg_val( loss )
                        self.mse_agg_val( loss )

                    elif self.m_params['model_type_settings']['discrete_continuous'] == True:
                        # make predictions
                        preds = self.model(tf.cast(feature,tf.float16), training=False )
                        preds = tf.squeeze(preds)
                        preds, probs = tf.unstack(preds, axis=0)

                        # selecting central region for evaluation
                        preds   = preds[:, :, 6:10, 6:10]
                        probs   = probs[:, :,  6:10, 6:10]
                        mask    =  mask[:, :, 6:10, 6:10]
                        target  = target[:, :, 6:10, 6:10]

                        # applying mask
                        preds_masked    = tf.boolean_mask( preds, mask )
                        probs_masked    = tf.boolean_mask( probs, mask)
                        target_masked   = tf.boolean_mask( target, mask )
                        preds_masked    = utility.standardize_ati( preds_masked, self.t_params['normalization_shift']['rain'], 
                                                                self.t_params['normalization_scales']['rain'], reverse=True)

                        # getting classification labels for whether or not it rained
                        labels_true = tf.cast( tf.greater( target_masked, 0.5 ), tf.float32 )
                        labels_pred = probs_masked 

                        all_count = tf.size( labels_true, out_type=tf.int64 )

                        # gather predictions which are conditional on true rain occuring
                        bool_rain = tf.where(tf.equal(labels_true,1), True, False )

                        preds_cond_rain     = tf.boolean_mask( preds_masked, bool_rain)
                        probs_cond_rain     = tf.boolean_mask( probs_masked, bool_rain)
                        target_cond_rain    = tf.boolean_mask( target_masked, bool_rain )
                                            
                        # calculating cross entropy loss                         
                        loss = tf.reduce_mean(  tf.keras.backend.binary_crossentropy( labels_true, labels_pred, from_logits=False) )

                        # calculating discrete continuous loss
                        if self.m_params['model_type_settings']['distr_type'] == 'Normal':
                            #Normal
                            loss    += cl.mse( preds_cond_rain, target_cond_rain, all_count )

                        elif self.m_params['model_type_settings']['distr_type'] == 'LogNormal':  
                            #LogNormal                 
                            loss    += cl.log_mse( preds_cond_rain, target_cond_rain, all_count)

                        # calculating seperate mse loss for reporting
                        mse = cl.mse( target_cond_rain, cl.cond_rain( preds_cond_rain, probs_cond_rain), all_count )
                    
                        self.loss_agg_val(loss)
                        self.mse_agg_val(mse)

                elif(self.m_params['model_type_settings']['stochastic']==True):
                    raise NotImplementedError                
                
                # Reporting for validation
                if batch % self.val_batch_report_freq == 0 or batch==self.t_params['val_batches'] :
                    batch_group_time            =  time.time() - start_batch_group_time
                    est_completion_time_seconds = (batch_group_time/self.t_params['reporting_freq']) * (1 -  batch/self.t_params['val_batches'])
                    est_completion_time_mins    = est_completion_time_seconds/60

                    print("\t\tCompleted Validation Batch:{}/{} \t Time:{:.4f} \tEst Time Left:{:.1f}".format( batch, self.t_params['val_batches'], batch_group_time,est_completion_time_mins ))
                                                
                    start_batch_group_time = time.time()
            #endregion        
            self.model.reset_states()

            # region - End of Epoch Reporting and Early iteration Callback
            if( self.m_params['model_type_settings']['stochastic'] == False ):
                print("\tEpoch:{}\t Train Loss:{:.8f}\tVal Loss:{:.5f}\t Train MSE:{:.5f}\t Val MSE:{:.5f}\t Time:{:.5f}".format(epoch, self.loss_agg_epoch.result(), self.loss_agg_val.result(), self.mse_agg_epoch.result(), self.mse_agg_val.result() ,time.time()-start_epoch_val  ) )

            if( self.m_params['model_type_settings']['stochastic'] == True ):
                raise NotImplementedError
                    
            utility.tensorboard_record( self.writer.as_default(), [self.loss_agg_val.result(), self.mse_agg_val.result()], ['Validation Loss', 'Validation MSE' ], epoch  )                    
            self.df_training_info = utility.update_checkpoints_epoch(self.df_training_info, epoch, self.loss_agg_epoch, self.loss_agg_val, self.ckpt_mngr_epoch, self.t_params, self.m_params, self.mse_agg_epoch, self.mse_agg_val )
            
            # Early Stop Callback 
            if epoch > ( max( self.df_training_info.loc[:, 'Epoch'], default=0 ) + self.t_params['early_stopping_period']) :
                print("Model Stopping Early at EPOCH {}".format(epoch))
                print(self.df_training_info)
                break
            # endregion
            



        print("Model Training Finished")

if __name__ == "__main__":
    s_dir = utility.get_script_directory(sys.argv[0])
    args_dict = utility.parse_arguments(s_dir)
    t_params, m_params = utility.load_params(args_dict)
    
    train_tru_net = TrainTrueNet(t_params, m_params)
    train_tru_net.initialize_scheme_era5Eobs()
    train_tru_net.train_model()
    