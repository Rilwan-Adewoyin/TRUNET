from netCDF4 import Dataset, num2date
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import argparse
import ast
import gc
import logging
import math

import sys
import time

import numpy as np
import pandas as pd
import psutil

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

try:
    import tensorflow_addons as tfa
except Exception as e:
    tfa = None

import data_generators
import custom_losses as cl
import hparameters
import models
import utility


#logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.keras.backend.set_floatx('float16')
tf.keras.backend.set_epsilon(1e-3)

try:
    gpu_devices = tf.config.list_physical_devices('GPU')
except Exception as e:
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')

# print("GPU Available: {}\n GPU Devices:{} ".format(tf.test.is_gpu_available(), gpu_devices) )
# for idx, gpu_name in enumerate(gpu_devices):
#     tf.config.experimental.set_memory_growth(gpu_name, True)

# try:
#     tf.config.set_logical_device_configuration(
#         gpu_devices[0], 
#         [tf.config.LogicalDeviceConfiguration(memory_limit=4000),
#             tf.config.LogicalDeviceConfiguration(mem6ry_limit=3800)] )
#     gpu_devices = tf.config.list_logical_devices('GPU')

# except:
#     pass

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

class TrainTruNet():
    """Handles the Training of the TRUNET model

        Example of how to use:
        traintrunet = TrainTruNet( t_params, m_params)
        traintrunet.initialize_scheme_era5Eobs()    #Initializes datasets for ERA5 and Eobs
        traintrunet.train_model()                   #Trains and saves model

    """    
    def __init__(self, t_params, m_params): 
        """Train the TRU_NET Model
        """
        self.t_params = t_params
        self.m_params = m_params
        
    def initialize_scheme_era5Eobs(self):
        """Initialization scheme for the ERA5 and E-OBS datasets.
            This method creates the datasets
        """        
        # region ---- Parameters  related to training length and training reporting frequency 
        era5_eobs = data_generators.Era5_Eobs( self.t_params, self.m_params)

        # hparameters files calculates train_batches assuing we are only evaluating one location, 
            # therefore we must adjust got multiple locations (loc_count)
        self.t_params['train_batches'] = int(self.t_params['train_batches'] * era5_eobs.loc_count)
        self.t_params['val_batches'] = int(self.t_params['val_batches'] * era5_eobs.loc_count)

        # The fequency at which we report during training and validation i.e every 10% of minibatches report training loss and training mse
        self.train_batch_report_freq = max( int(self.t_params['train_batches']*self.t_params['reporting_freq']), 3)
        self.val_batch_report_freq = max( int(self.t_params['val_batches']*self.t_params['reporting_freq']), 3)
        #endregion

        # region ---- Restoring/Creating New Training Records and Restoring training progress
            #This training records keeps track of the losses on each epoch
        try:
            self.df_training_info = pd.read_csv( "checkpoints/{}/checkpoint_scores.csv".format(utility.model_name_mkr(m_params,t_params=self.t_params)), header=0, index_col=False) 
            self.df_training_info = self.df_training_info[['Epoch','Train_loss','Train_Mse','Val_loss','Val_mse','Checkpoint_Path','Last_Trained_Batch']]
            self.start_epoch =  int(max([self.df_training_info['Epoch'][0]], default=0))
            last_batch = int( self.df_training_info.loc[self.df_training_info['Epoch']==self.start_epoch,'Last_Trained_Batch'].iloc[0] )
            if(last_batch in [-1, self.t_params['train_batches']] ):
                self.start_epoch = self.start_epoch + 1
                #self.batches_to_skip = 0
            else:
                #self.batches_to_skip = last_batch
                pass
            self.batches_to_skip = 0
            print("Recovered training records")

        except FileNotFoundError as e:
            #If no file found, then make new training records file
            self.df_training_info = pd.DataFrame(columns=['Epoch','Train_loss','Train_Mse','Val_loss','Val_mse','Checkpoint_Path','Last_Trained_Batch'] ) 
            self.batches_to_skip = 0
            self.start_epoch = 0
            print("Did not recover training records. Starting from scratch")
        # endregion

        # region ---- Defining Model / Optimizer / Losses / Metrics / Records / Checkpoints / Tensorboard 
                    
        self.strategy = tf.distribute.MirroredStrategy() #OneDeviceStrategy(device="/GPU:0") # 
        
        assert self.t_params['batch_size'] % self.strategy.num_replicas_in_sync  == 0
        print("Nuber of Devices used in MirroredStrategy: {}".format(self.strategy.num_replicas_in_sync))
        with self.strategy.scope():   
            #Model
            self.strategy_gpu_count = self.strategy.num_replicas_in_sync    
            self.t_params['gpu_count'] = self.strategy.num_replicas_in_sync    
            self.model = models.model_loader( self.t_params, self.m_params )
            #Optimizer
            optimizer = tfa.optimizers.RectifiedAdam( **self.m_params['rec_adam_params'], total_steps=self.t_params['train_batches']*20 ) #, weight_decay=1.25e-5 )         
            self.optimizer = mixed_precision.LossScaleOptimizer( optimizer, loss_scale=tf.mixed_precision.experimental.DynamicLossScale() ) 
            
        
        
        with self.strategy.scope():
            # These objects will aggregate losses and metrics across batches and epochs
            self.loss_agg_batch = tf.keras.metrics.Mean(name='loss_agg_batch' )
            self.loss_agg_epoch = tf.keras.metrics.Mean(name="loss_agg_epoch")

            self.mse_agg_epoch = tf.keras.metrics.Mean(name='mse_agg_epoch')
            
            self.loss_agg_val = tf.keras.metrics.Mean(name='loss_agg_val')
            self.mse_agg_val = tf.keras.metrics.Mean(name='mse_agg_val')

        #checkpoints  (For Epochs)
            #The CheckpointManagers can be called to serializae the weights within TRUNET
        checkpoint_path_epoch = "checkpoints/{}/epoch".format(utility.model_name_mkr(m_params,t_params=self.t_params ))
        os.makedirs(checkpoint_path_epoch,exist_ok=True)
        with self.strategy.scope():
            ckpt_epoch = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
            self.ckpt_mngr_epoch = tf.train.CheckpointManager(ckpt_epoch, checkpoint_path_epoch, max_to_keep=self.t_params['checkpoints_to_keep'], keep_checkpoint_every_n_hours=None)    
        

            #restoring last checkpoint if it exists
        if self.ckpt_mngr_epoch.latest_checkpoint: 
            # compat: Initializing model and optimizer before restoring from checkpoint
            inp_shape = [t_params['batch_size'], t_params['lookback_feature']] + m_params['region_grid_params']['outer_box_dims'] + [len(t_params['vars_for_feature'])]
            inp_shape1 = [t_params['batch_size'], t_params['lookback_target']] + m_params['region_grid_params']['outer_box_dims'] 
            bool_cmpltd = self.distributed_train_step( 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 )

            ckpt_epoch.restore(self.ckpt_mngr_epoch.latest_checkpoint).assert_consumed()              
        else:
            print (' Initializing model from scratch')
        
        #Tensorboard
        os.makedirs("log_tensboard/{}".format(utility.model_name_mkr(m_params, t_params=self.t_params)), exist_ok=True ) 
        self.writer = tf.summary.create_file_writer( "log_tensboard/{}".format(utility.model_name_mkr(m_params,t_params=self.t_params) ) )
        # endregion
        
        
        bc_ds_in_train = int( self.t_params['train_batches']/era5_eobs.loc_count  ) #batch_count
        bc_ds_in_val = int( self.t_params['val_batches']/era5_eobs.loc_count )

        self.reset_idxs_training = np.cumsum( [bc_ds_in_train]*era5_eobs.loc_count )
        self.reset_idxs_validation = np.cumsum( [bc_ds_in_val]*era5_eobs.loc_count )

        # region ---- Making Datasets
        
        #caching dataset to file post pre-processing steps have been completed 
        cache_suffix = utility.cache_suffix_mkr( m_params, self.t_params )
        os.makedirs( './Data/data_cache/', exist_ok=True  )

        _ds_train_val, _  = era5_eobs.load_data_era5eobs( self.t_params['train_batches'] + self.t_params['val_batches'] , self.t_params['start_date'] )

        ds_train = _ds_train_val.take(self.t_params['train_batches'] )
        ds_val = _ds_train_val.skip(self.t_params['train_batches'] ).take(self.t_params['val_batches'])

        ds_train = ds_train.cache('Data/data_cache/train'+cache_suffix ) 
        ds_val = ds_val.cache('Data/data_cache/val'+cache_suffix )
        ds_train = ds_train.unbatch().shuffle( self.t_params['batch_size']*int(self.t_params['train_batches']/5), reshuffle_each_iteration=True).batch(self.t_params['batch_size']) #.repeat(self.t_params['epochs']-self.start_epoch)

        ds_train_val = ds_train.concatenate(ds_val)
        ds_train_val = ds_train_val.repeat(self.t_params['epochs']-self.start_epoch)
        self.ds_train_val = self.strategy.experimental_distribute_dataset(dataset=ds_train_val)
        self.iter_train_val = enumerate(self.ds_train_val)
        
    def train_model(self):
        """During training we produce a prediction for a (n by n) square patch. 
            But we caculate losses on a central (h, w) region within the (n by n) patch
            This central region is defined by "bounds" below
        """        

        bounds = cl.central_region_bounds(self.m_params['region_grid_params']) #list [ lower_h_bound[0], upper_h_bound[0], lower_w_bound[1], upper_w_bound[1] ]
        
        #Training for n epochs
        for epoch in range(self.start_epoch, int(self.t_params['epochs']) ):
            
            #region resetting metrics, losses, records, timers
            self.loss_agg_batch.reset_states()
            self.loss_agg_epoch.reset_states()
            self.mse_agg_epoch.reset_states()
            
            self.loss_agg_val.reset_states()
            self.mse_agg_val.reset_states()
            self.df_training_info = self.df_training_info.append( { 'Epoch':epoch, 'Last_Trained_Batch':0 }, ignore_index=True )
            
            r_batch_size = self.t_params['batch_size'] // self.strategy_gpu_count
            last_update_epoch = int( max( self.df_training_info['Epoch'][:], default=0 ) )
            epoch_non_update = last_update_epoch - epoch  
            if epoch_non_update == 0:
                pass
            elif epoch_non_update >= -2:
                r_batch_size = max( [self.t_params['batch_size']//(2*self.strategy_gpu_count), 2] )
            elif epoch_non_update >= -5:
                r_batch_size = max( [self.t_params['batch_size']//(4*self.strategy_gpu_count) ,2] )
            elif epoch_non_update >= -15:
                r_batch_size = max( [self.t_params['batch_size']//(8*self.strategy_gpu_count) ,2] ) 
            
            start_epoch_train = time.time()
            start_batch_group_time = time.time()
            batch=0           
            
            # if( epoch!=self.start_epoch  ):
            #     self.batches_to_skip = 0
            #print("\n\nStarting EPOCH {} Batch {}/{}".format(epoch, self.batches_to_skip+1, self.t_params['train_batches']))
            print("\n\nStarting EPOCH {}".format(epoch ))
            #endregion 
            
            # --- Training Loops
            for batch in range(self.batches_to_skip+1, self.t_params['train_batches']+1):
                               
                # get next set of training datums
                idx, (feature, target, mask) = next(self.iter_train_val)
                

                gradients = self.distributed_train_step( feature, target, mask, bounds, 0.0, r_batch_size )
                #print(gradients)
                
                # reporting
                if( batch % self.train_batch_report_freq==0 or batch == self.t_params['train_batches']):
                    batch_group_time =  time.time() - start_batch_group_time
                    est_completion_time_seconds = (batch_group_time/self.t_params['reporting_freq']) * (1 - batch/self.t_params['train_batches'])
                    est_completion_time_mins = est_completion_time_seconds/60

                    print("\t\tBatch:{}/{}\tTrain Loss: {:.8f} \t Batch Time:{:.4f}\tEpoch mins left:{:.1f}".format(batch, self.t_params['train_batches'], self.loss_agg_batch.result(), batch_group_time, est_completion_time_mins ) )
                    
                    # resetting time and losses
                    start_batch_group_time = time.time()

                    # Updating record of the last batch to be operated on in training epoch
                    self.df_training_info.loc[ ( self.df_training_info['Epoch']==epoch) , ['Last_Trained_Batch'] ] = batch
                    self.df_training_info.to_csv( path_or_buf="checkpoints/{}/checkpoint_scores.csv".format(utility.model_name_mkr(self.m_params,t_params=self.t_params)), header=True, index=False )

                li_losses = [self.loss_agg_batch.result()]
                li_names = ['train_loss_batch']
                step = batch + (epoch)*self.t_params['train_batches']
                #utility.tensorboard_record( self.writer.as_default(), li_losses, li_names, step, gradients, self.model.trainable_variables )
                #utility.tensorboard_record( self.writer.as_default(), li_losses, li_names, step, None, None )
                self.loss_agg_batch.reset_states()

                if batch in self.reset_idxs_training:
                    self.model.reset_states()
                    
            # --- Tensorboard record          
            li_losses = [self.loss_agg_epoch.result(), self.mse_agg_epoch.result()]
            li_names = ['train_loss_epoch','train_mse_epoch']
            #utility.tensorboard_record( self.writer.as_default(), li_losses, li_names, epoch)
            
            
            print("\tStarting Validation")
            start_batch_group_time = time.time()

            # --- Validation Loops
            for batch in range(1, self.t_params['val_batches']+1):
                
                # next datum
                idx, (feature, target, mask) = next(self.iter_train_val)
                
                bool_cmpltd = self.distributed_val_step(feature, target, mask, bounds)

                # Reporting for validation
                if batch % self.val_batch_report_freq == 0 or batch==self.t_params['val_batches'] :
                    batch_group_time            =  time.time() - start_batch_group_time
                    est_completion_time_seconds = (batch_group_time/self.t_params['reporting_freq']) * (1 -  batch/self.t_params['val_batches'])
                    est_completion_time_mins    = est_completion_time_seconds/60

                    print("\t\tCompleted Validation Batch:{}/{} \t Time:{:.4f} \tEst Time Left:{:.1f}".format( batch, self.t_params['val_batches'], batch_group_time, est_completion_time_mins))
                                                
                    start_batch_group_time = time.time()
                
                if batch in self.reset_idxs_validation:
                    self.model.reset_states()

            # region - End of Epoch Reporting and Early iteration Callback
            print("\tEpoch:{}\t Train Loss:{:.8f}\t Train MSE:{:.5f}\t Val Loss:{:.5f}\t Val MSE:{:.5f}\t Time:{:.5f}".format(epoch, self.loss_agg_epoch.result(), self.mse_agg_epoch.result(), 
                        self.loss_agg_val.result(), self.mse_agg_val.result(), time.time()-start_epoch_train  ) )
                    
            #utility.tensorboard_record( self.writer.as_default(), [self.loss_agg_val.result(), self.mse_agg_val.result()], ['Validation Loss', 'Validation MSE' ], epoch  )                    
            self.df_training_info = utility.update_checkpoints_epoch(self.df_training_info, epoch, self.loss_agg_epoch, self.loss_agg_val, self.ckpt_mngr_epoch, self.t_params, 
                    self.m_params, self.mse_agg_epoch, self.mse_agg_val )
            
            # Early Stop Callback 
            if epoch > ( max( self.df_training_info.loc[:, 'Epoch'], default=0 ) + self.t_params['early_stopping_period']) :
                print("Model Stopping Early at EPOCH {}".format(epoch))
                print(self.df_training_info)
                break
            # endregion
        
        print("Model Training Finished")

    def train_step(self, feature, target, mask, bounds, _init, r_batch_size):
        
        if _init==1.0:
            inp_shape = [self.t_params['batch_size'], self.t_params['lookback_feature']] + self.m_params['region_grid_params']['outer_box_dims'] + [len(self.t_params['vars_for_feature'])]
            inp_2 = [self.t_params['batch_size'], self.t_params['lookback_target']] + self.m_params['region_grid_params']['outer_box_dims'] 

            _ = self.model( tf.zeros( inp_shape ,dtype=tf.float16), self.t_params['trainable'] ) #( bs, tar_seq_len, h, w)
            gradients = [ tf.zeros_like(t_var, dtype=tf.float32 ) for t_var in self.model.trainable_variables  ]
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return [0]

        with tf.GradientTape(persistent=False) as tape:
                                   
            # non conditional continuous training
            if self.m_params['model_type_settings']['discrete_continuous'] == False:
                
                #making predictions
                preds = self.model( feature, self.t_params['trainable'] ) #( bs, tar_seq_len, h, w)
                preds = tf.squeeze( preds,axis=[-1] )

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

            # conditional continuous training        
            elif self.m_params['model_type_settings']['discrete_continuous'] == True:
                
                # Producing predictions - conditional rain value and prob of rain
                preds   = self.model( feature, self.t_params['trainable'] ) # ( bs, seq_len, h, w, 1)
                preds   = tf.squeeze(preds, axis=[-1])
                preds, probs = tf.unstack(preds, axis=0) 

                # extracting the central region of interest
                preds   = cl.extract_central_region(preds, bounds)
                probs   = cl.extract_central_region(probs, bounds)
                mask    = cl.extract_central_region(mask, bounds)
                target  = cl.extract_central_region(target, bounds)

                # For Large Batches we need to dynamically operate on smaller batches to improve loss, retains speed of larger epochs tho
                per_gpu_bs = self.t_params['batch_size']//self.strategy_gpu_count
                
                if r_batch_size != per_gpu_bs:
                    indices = tf.random.uniform( [ r_batch_size ], minval=0 , maxval=per_gpu_bs, dtype=tf.int32 )
                    preds = tf.gather( preds, indices, axis=0)
                    probs = tf.gather( probs, indices, axis=0)
                    target= tf.gather( target,indices, axis=0)
                    mask    = tf.gather(mask, indices, axis=0)

                # applying mask to predicted values
                preds_masked    = tf.boolean_mask(preds, mask )
                probs_masked    = tf.boolean_mask(probs, mask ) 
                target_masked   = tf.boolean_mask(target, mask )
                
                # Reverising standardization of predictions 
                preds_masked    = utility.standardize_ati( preds_masked, self.t_params['normalization_shift']['rain'], 
                                                        self.t_params['normalization_scales']['rain'], reverse=True) 
                                                        
                # Getting true labels and predicted labels for whether or not it rained [ 1 if if did rain, 0 if it did not rain]
                labels_true = tf.where( target_masked > 0.5, 1.0, 0.0 )
                labels_pred = probs_masked 

                all_count = tf.size( labels_true, out_type=tf.int64 )

                # Seperating predictions for the days it did rain and the days it did not rain
                bool_rain = (labels_true==1.0) 

                preds_cond_rain         = tf.boolean_mask( preds_masked, bool_rain)
                #probs_cond_rain         = tf.boolean_mask( probs_masked, bool_rain)                        
                target_cond_rain        = tf.boolean_mask( target_masked, bool_rain)                   
                
                # region Calculating Losses
                loss_to_optimize = 0

                #NOTE: Error on vandal equation 14 last line, he forgets to put the logs and forgets the negative symbol so correct in yours

                if self.m_params['model_type_settings']['distr_type'] == 'Normal': 
                    # CC Normal
                    loss_to_optimize += 1.1*cl.mse( target_cond_rain, preds_cond_rain, all_count )
                
                elif self.m_params['model_type_settings']['distr_type'] == 'LogNormal':    
                    # CC LogNormal                                                             
                    loss_to_optimize += cl.log_mse( target_cond_rain, preds_cond_rain, all_count)                                                         
                
                if True:
                    loss_to_optimize += 0.9*tf.reduce_mean( tf.keras.backend.binary_crossentropy(labels_true, labels_pred, from_logits=False) ) 
                    loss_for_record = loss_to_optimize
                
                else:
                    loss_for_record  = loss_to_optimize + tf.reduce_mean( 
                                    tf.keras.backend.binary_crossentropy(labels_true, labels_pred, from_logits=False) ) 

                metric_mse  = cl.mse( target_masked, cl.cond_rain(preds_masked, probs_masked) )   
                    # To calculate metric_mse for CC model we assume that pred_rain=0 if pred_prob<0.5 
                # endregion

            loss_to_optimize_agg = tf.grad_pass_through( lambda x:  x/self.strategy_gpu_count )(loss_to_optimize + tf.reduce_sum( [tf.cast(x, dtype=tf.float32) for x in self.model.losses]))
            #loss_to_optimize_agg = tf.grad_pass_through( lambda x:  x/self.strategy_gpu_count )(loss_to_optimize + tf.cast( tf.reduce_sum(self.model.losses), dtype=tf.float16) ) 
            scaled_loss = self.optimizer.get_scaled_loss( loss_to_optimize_agg )
            scaled_gradients = tape.gradient( scaled_loss, self.model.trainable_variables )
            unscaled_gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
            gradients, _ = tf.clip_by_global_norm( unscaled_gradients, clip_norm=self.m_params['clip_norm'] ) #gradient clipping
            self.optimizer.apply_gradients( zip(gradients, self.model.trainable_variables))
        
        # Metrics (batchwise, epoch)  
        # we pass in non global_batch_averaged results to keras.Metrics.Mean 
        # as advised in tensorflow multi-gpu script  
        self.loss_agg_batch( loss_for_record )
        self.loss_agg_epoch( loss_for_record )
        self.mse_agg_epoch( metric_mse )        
        return gradients
                
    def val_step(self, feature, target, mask, bounds):
                    
        # Non CC distribution
        if self.m_params['model_type_settings']['discrete_continuous'] == False:
            
            # Get predictions
            preds = self.model(feature, False )
            preds = tf.squeeze(preds)

            # Extracting central region for evaluation
            preds   = cl.extract_central_region(preds, bounds)
            mask    = cl.extract_central_region(mask, bounds)
            target  = cl.extract_central_region(target, bounds)
            
            # Applying masks to predictions
            preds_masked = tf.boolean_mask( preds, mask )
            target_masked = tf.boolean_mask( target, mask )
            preds_masked = utility.standardize_ati( preds_masked, self.t_params['normalization_shift']['rain'], 
                                                    self.t_params['normalization_scales']['rain'], reverse=True)
            # Updating losses
            mse = cl.mse( target_masked , preds_masked ) 
            loss = mse                

        # CC distribution
        elif self.m_params['model_type_settings']['discrete_continuous'] == True:
            
            # Get predictions
            preds = self.model(feature, training=False )
            preds = tf.squeeze(preds,axis=[-1])
            preds, probs = tf.unstack(preds, axis=0)

            # Extracting central region for evaluation
            preds   = cl.extract_central_region(preds, bounds)
            probs   = cl.extract_central_region(probs, bounds)
            mask    = cl.extract_central_region(mask,  bounds)
            target  = cl.extract_central_region(target,bounds)

            # Applying masks to predictions
            preds_masked    = tf.boolean_mask( preds, mask )
            probs_masked    = tf.boolean_mask( probs, mask)
            target_masked   = tf.boolean_mask( target, mask )
            preds_masked    = utility.standardize_ati( preds_masked, self.t_params['normalization_shift']['rain'], 
                                                    self.t_params['normalization_scales']['rain'], reverse=True)

            # Getting classification labels for whether or not it rained
            #labels_true = tf.cast( tf.greater( target_masked, 0.5 ), tf.float32 )
            labels_true = tf.where( target_masked > 0.5, 1.0, 0.0 )
            labels_pred = probs_masked 

            all_count = tf.size( labels_true, out_type=tf.int64 )

            # Gathering predictions which are conditional on rain actually occuring
            bool_rain = (labels_true==1.0) 

            preds_cond_rain     = tf.boolean_mask( preds_masked, bool_rain)
            target_cond_rain    = tf.boolean_mask( target_masked, bool_rain )
                                
            # Calculating cross entropy loss                         
            loss = tf.reduce_mean(  tf.keras.backend.binary_crossentropy( labels_true, labels_pred, from_logits=False) )

            # Calculating conditional continuous loss
            if self.m_params['model_type_settings']['distr_type'] == 'Normal':
                #Conditional Normal distribution
                loss    += cl.mse( preds_cond_rain, target_cond_rain, all_count )

            elif self.m_params['model_type_settings']['distr_type'] == 'LogNormal':  
                #COnditional LogNormal distribution                 
                loss    += cl.log_mse( preds_cond_rain, target_cond_rain, all_count)

            # calculating seperate mse for reporting
                # This mse metric assumes that if probability of rain is predicted below 0.5, the rain value is 0
            mse = cl.mse( target_masked, cl.cond_rain( preds_masked, probs_masked) )
        
        self.loss_agg_val(loss)
        self.mse_agg_val(mse)

        return True
    
    @tf.function
    def distributed_train_step(self, feature, target, mask, bounds, _init, r_batch_size):
        #bool_completed = self.strategy.run( self.train_step, args=(feature, target, mask, bounds, _init))
        gradients = self.strategy.experimental_run_v2( self.train_step, args=(feature, target, mask, bounds, _init, r_batch_size))
        return gradients
    
    @tf.function
    def distributed_val_step(self, feature, target, mask, bounds):
        #bool_completed = self.strategy.run( self.val_step, args=(feature, target, mask, bounds))
        bool_completed = self.strategy.experimental_run_v2( self.val_step, args=(feature, target, mask, bounds))
        return bool_completed


if __name__ == "__main__":
    s_dir = utility.get_script_directory(sys.argv[0])
    args_dict = utility.parse_arguments(s_dir)

    # get training and model params
    t_params, m_params = utility.load_params(args_dict)
    
    # Initialize and  train model
    train_tru_net = TrainTruNet(t_params, m_params)
    train_tru_net.initialize_scheme_era5Eobs()
    train_tru_net.train_model()
    
