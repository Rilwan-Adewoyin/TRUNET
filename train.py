# BNN - Uncertainty SCRNN - ATI Project - PhD Computer Science
#region imports
import os
import sys

import utility

import tensorflow as tf
import tensorflow_probability as tfp
try:
    import tensorflow_addons as tfa
except Exception as e:
    tfa = None
from tensorflow_probability import layers as tfpl
from tensorflow_probability import distributions as tfd
from tensorboard.plugins.hparams import api as hp
import pandas as pd

import math
import numpy as np

import argparse 
from tqdm import tqdm
import traceback
import time

import models
import hparameters
import data_generators
# endregion

# region train 
def train_loop(train_params, model_params): 
    print("GPU Available: ", tf.test.is_gpu_available() )
    
    # region ----- Defining Model /Optimizer / Losses / Metrics / Records
    model = models.model_loader(train_params, model_params)
    if type(model_params) == list:
        model_params = model_params[0]

    if tfa==None:
        optimizer = tf.keras.optimizers.Adam( learning_rate=1e-4, beta_1=0.1, beta_2=0.99, epsilon=1e-5 )
    else:
        radam = tfa.optimizers.RectifiedAdam( learning_rate=2e-4, total_steps=500, warmup_proportion=0.5, min_lr=1e-4, beta_1=0.9, beta_2=0.99, epsilon=1e-5 )
        optimizer = tfa.optimizers.Lookahead(radam, sync_period = 7, slow_step_size=0.5)

    train_metric_mse_mean_groupbatch = tf.keras.metrics.Mean(name='train_loss_mse_obj')
    train_metric_mse_mean_epoch = tf.keras.metrics.Mean(name="train_loss_mse_obj_epoch")
    train_loss_var_free_nrg_mean_groupbatch = tf.keras.metrics.Mean(name='train_loss_var_free_nrg_obj ')
    train_loss_var_free_nrg_mean_epoch = tf.keras.metrics.Mean(name="train_loss_var_free_nrg_obj_epoch")
    val_metric_mse_mean = tf.keras.metrics.Mean(name='val_metric_mse_obj')

    try:
        df_training_info = pd.read_csv( "checkpoints/{}/checkpoint_scores_model_{}.csv".format(model_params['model_name'],model_params['model_version']), header=0, index_col =False   )
        print("Recovered checkpoint scores model csv")

    except Exception as e:
        df_training_info = pd.DataFrame(columns=['Epoch','Train_loss_MSE','Val_loss_MSE','Checkpoint_Path', 'Last_Trained_Batch'] ) #key: epoch number #Value: the corresponding loss #TODO: Implement early stopping
        print("Did not recover checkpoint scores model csv")
  
    # endregion

    # region ----- Setting up Checkpoints 
        #  (For Epochs)
    checkpoint_path_epoch = "checkpoints/{}/epoch/{}".format(model_params['model_name'],model_params['model_version'])
    if not os.path.exists(checkpoint_path_epoch):
        os.makedirs(checkpoint_path_epoch)
        
    ckpt_epoch = tf.train.Checkpoint(att_con=model, optimizer=optimizer)
    ckpt_manager_epoch = tf.train.CheckpointManager(ckpt_epoch, checkpoint_path_epoch, max_to_keep=train_params['epochs'], keep_checkpoint_every_n_hours=None)    
     
        # (For Batches)
    checkpoint_path_batch = "checkpoints/{}/batch/{}".format(model_params['model_name'],model_params['model_version'])
    if not os.path.exists(checkpoint_path_batch):
        os.makedirs(checkpoint_path_batch)
        #Create the checkpoint path and the checpoint manager. This will be used to save checkpoints every n epochs
    ckpt_batch = tf.train.Checkpoint(att_con=model, optimizer=optimizer)
    ckpt_manager_batch = tf.train.CheckpointManager(ckpt_batch, checkpoint_path_batch, max_to_keep=train_params['checkpoints_to_keep'], keep_checkpoint_every_n_hours=None)

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
    
    train_batch_reporting_freq = int(train_set_size_batches*train_params['dataset_trainval_batch_reporting_freq'] )
    val_batch_reporting_freq = int(val_set_size_batches*2*train_params['dataset_trainval_batch_reporting_freq'] )
    #endregion

    # region Logic for setting up resume location
    starting_epoch =  int(max( df_training_info['Epoch'], default=1 )) 
    df_batch_record = df_training_info.loc[ df_training_info['Epoch'] == starting_epoch,'Last_Trained_Batch' ]

    if( len(df_batch_record)==0 or df_batch_record.iloc[0]==-1 ):
        batches_to_skip = 0
    else:
        batches_to_skip = int(df_batch_record.iloc[0])   
    
    #batches_to_skip_on_error = 2
    # endregion

    # region --- Tensorboard
    os.makedirs("log_tensboard/{}/{}".format(model_params['model_name'],model_params['model_version']), exist_ok=True )
    writer = tf.summary.create_file_writer( "log_tensboard/{}/{}/tblog".format(model_params['model_name'], model_params['model_version']) )
    # endregion

    # region --- Train and Val
    for epoch in range(starting_epoch, int(train_params['epochs']+1) ):
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

        #region Setting up Datasets
        if( model_params['model_name'] == "DeepSD" ):
            ds_train = data_generators.load_data_vandal( batches_to_skip*train_params['batch_size'], train_params  )
            ds_val = data_generators.load_data_vandal( train_set_size_batches*train_params['batch_size'], train_params )

        elif( model_params['model_name'] == "THST" ):

            ds_train = data_generators.load_data_ati(train_params, model_params, day_to_start_at=train_params['train_start_date'] )
            ds_val = data_generators.load_data_ati(train_params, model_params, day_to_start_at=train_params['val_start_date'] )

        #endregion

        batch=0
        epoch_finished = False  
        while(epoch_finished==False):

            print("\n\nStarting EPOCH {} Batch {}/{}".format(epoch, batches_to_skip+1, train_set_size_batches))
            with writer.as_default():
                iter_train = iter(ds_train)
                iter_val = iter(ds_val)

                for batch in range(1+batches_to_skip,2+train_set_size_batches+val_set_size_batches):
                    # region Train Loop
                    if(batch<=train_set_size_batches):
                        feature, target = next(iter_train)  #ATI- ( (feature,mask), (target) ) V-shape( (bs, 39, 88, 17 ) (bs,156,352) )
                        
                        with tf.GradientTape(persistent=True) as tape:
                            if model_params['model_name'] == "DeepSD":
                                preds = model( feature, tape=tape ) #shape batch_size, output_h, output_w, 1 #TODO Debug, remove tape variable from model later
                                #noise_std = tfd.HalfNormal(scale=5)     #TODO(akanni-ade): remove (mask) eror for predictions that are water i.e. null, through water_mask
                                preds = utility.water_mask( tf.squeeze(preds), train_params['bool_water_mask'])
                                preds = tf.reshape( preds, [train_params['batch_size'], -1] )       #TODO:(akanni-ade) This should decrease exponentially during training #RESEARCH: NOVEL Addition #TODO:(akanni-ade) create tensorflow function to add this
                                target = tf.reshape( target, [train_params['batch_size'], -1] )     #NOTE: In the original Model Selection paper they use Guassian Likelihoods for loss with a precision (noise_std) that is Gamma(6,6)
                                
                                preds_distribution_norm = tfd.Normal( loc=preds, scale= 0.1)#noise_std.sample() )  #The sample here should be dependent on previous model loss, relative to maybe KL term
                                                                
                                log_likelihood = tf.reduce_mean( preds_distribution_norm.log_prob( target ),axis=-1 ) #This represents the expected log_likelihood corresponding to each target y_i in the mini batch

                                kl_loss_weight = utility.kl_loss_weighting_scheme(train_set_size_batches) 
                                kl_loss = tf.math.reduce_sum( model.losses ) * kl_loss_weight * (1/train_params['train_monte_carlo_samples'])  #This KL-loss is already normalized against the number of samples of weights drawn #TODO: Later implement your own Adam type method to determine this
                                
                                var_free_nrg_loss = kl_loss  - tf.reduce_sum(log_likelihood)/train_params['batch_size'] 

                                metric_mse = tf.reduce_sum( tf.keras.losses.MSE( target , preds ) )

                                gradients = tape.gradient( var_free_nrg_loss, model.trainable_variables )

                            
                            elif model_params['model_name'] == "THST" and model_params['stochastic'] ==False: #non stochastic version
                                feature, mask = feature

                                preds = model(feature, tape=tape )

                                preds_filtrd = tf.boolean_mask( preds, tf.logical_not(mask) )
                                target_filtrd = tf.boolean_mask( preds, tf.logical_not(mask) )

                                loss_mse = tf.keras.losses.MSE(target_filtrd, preds_filtrd)
                                metric_mse = tf.keras.losses.MSE(target_filtrd, preds_filtrd)

                                gradients = tape.gradient( loss_mse, model.trainable_variables )


                        
                        gradients_clipped_global_norm, _ = tf.clip_by_global_norm(gradients, model_params['gradients_clip_norm'] )
                        optimizer.apply_gradients( zip( gradients_clipped_global_norm, model.trainable_variables ) )
                        
                        #region Tensorboard Update
                        step = batch + (epoch-1)*train_set_size_batches
                        if( model_params['stochastic']==True ):
                            tf.summary.scalar('train_loss_var_free_nrg', var_free_nrg_loss , step =  step )
                            tf.summary.scalar('kl_loss', kl_loss, step=step )
                            tf.summary.scalar('neg_log_likelihood', - tf.reduce_sum(log_likelihood)/train_params['batch_size'], step=step )
                            tf.summary.scalar('train_metric_mse', metric_mse , step = step )
                        
                        else:
                            tf.summary.scalar('train_loss_mse', loss_mse , step = step )
                            tf.summary.scalar('train_metric_mse', metric_mse , step = step )
          

                        for grad, grad_clipped, _tensor in zip( gradients, gradients_clipped_global_norm ,model.trainable_variables):
                            if grad is not None:
                                tf.summary.histogram( "Grad:{}".format( _tensor.name ) , grad, step = step  )
                                tf.summary.histogram( "Grads_Norm:{}".format( _tensor.name ) , grad_clipped, step = step )
                                tf.summary.histogram( "Weights:{}".format(_tensor.name), _tensor , step = step )
                                
                        #tf.summary.trace_export( "Graph", step=step, profiler_outdir="log_tensboard/{}/{}/tblog".format(model_params['model_name'],model_params['model_version']) )
                        #endregion

                        #region training Reporting and Metrics updates
                        if( model_params['stochastic']==True ):
                            train_loss_var_free_nrg_mean_groupbatch( var_free_nrg_loss )
                            train_loss_var_free_nrg_mean_epoch( var_free_nrg_loss )
                            train_metric_mse_mean_groupbatch( metric_mse )
                            train_metric_mse_mean_epoch( metric_mse )
                        else:
                            train_metric_mse_mean_groupbatch( metric_mse )
                                                    
                        ckpt_manager_batch.save()
                        if( (batch%train_batch_reporting_freq)==0):
                            batches_report_time =  time.time() - start_batch_time

                            est_completion_time_seconds = (batches_report_time/train_params['dataset_trainval_batch_reporting_freq']) * (train_set_size_batches - batch)/train_set_size_batches
                            est_completion_time_mins = est_completion_time_seconds/60

                            print("\tBatch:{}/{}\tTrain MSE Loss: {:.4f} \t Time:{:.4f}\tEpoch mins left:{:.1f}".format(batch, train_set_size_batches, train_metric_mse_mean_groupbatch.result(), batches_report_time, est_completion_time_mins ) )
                            

                            # Updating record of the last batch to be operated on in training epoch
                        df_training_info.loc[ ( df_training_info['Epoch']==epoch) , ['Last_Trained_Batch'] ] = batch
                        df_training_info.to_csv( path_or_buf="checkpoints/{}/checkpoint_scores_model_{}.csv".format(model_params['model_name'],model_params['model_version']), header=True, index=False )
                        # endregion
                        
                        continue  
                    # endregion

                    # region Transition 
                    if(batch==train_set_size_batches+1):
                        if( model_params['stochastic']==True ):
                            print('EPOCH {}:\tVAR_FREE_NRG: {:.3f} \tMSE: {:.3f}\tTime: {:.2f}'.format(epoch, train_loss_var_free_nrg_mean_epoch.result() ,train_metric_mse_mean_epoch.result(), (time.time()-start_epoch ) ) )
                        else:
                            print('EPOCH {}:\tMSE: {:.3f}\tTime: {:.2f}'.format(epoch ,train_metric_mse_mean_epoch.result(), (time.time()-start_epoch ) ) )

                        print("\nStarting Validation")
                        start_epoch_val = time.time()
                        start_batch_time = time.time()
                    # endregion

                    #region Validation Loop
                    if(train_set_size_batches+1<= batch <= train_set_size_batches + val_set_size_batches  ):
                        feature, target = next(iter_val)

                        if model_params['model_name'] == "DeepSD":
                            preds = model( feature )
                            preds = utility.water_mask( tf.squeeze(preds), train_params['bool_water_mask'])
                            val_metric_mse_mean( tf.keras.metrics.MSE( tf.squeeze(preds) , target )  )
                        
                        elif model_params['model_name'] == "THST" and model_params['stochastic'] ==False: #non stochastic version
                            feature, mask = feature
                            preds = model(feature )

                            preds_filtrd = tf.boolean_mask( preds, tf.logical_not(mask) )
                            target_filtrd = tf.boolean_mask( preds, tf.logical_not(mask) )

                            val_metric_mse_mean( tf.keras.metrics.MSE( target_filtrd , preds_filtrd )  )



                        if  ( (batch-train_set_size_batches) % val_batch_reporting_freq) ==0 or batch==(train_set_size_batches+ val_set_size_batches) :
                            batches_report_time =  time.time() - start_batch_time
                            est_completion_time_seconds = (batches_report_time/train_params['dataset_trainval_batch_reporting_freq']) *( 1 -  ((batch-train_set_size_batches)/val_set_size_batches ) )
                            est_completion_time_mins = est_completion_time_seconds/60

                            print("\tCompleted Validation Batch:{}/{} \t Time:{:.4f} \tEst Time Left:{:.1f}".format( batch-train_set_size_batches, val_set_size_batches ,batches_report_time,est_completion_time_mins ))
                                                        
                            start_batch_time = time.time()
                        continue
                    # endregion

                    print("Epoch:{}\t Train MSE:{:.3f}\tValidation Loss: MSE:{:.4f}\tTime:{:.4f}".format(epoch, train_metric_mse_mean_epoch.result(), val_metric_mse_mean.result(), time.time()-start_epoch_val  ) )
                    epoch_finished = True
                    batches_to_skip = 0  
                    
        df_training_info = utility.update_checkpoints_epoch(df_training_info, epoch, train_metric_mse_mean_epoch, val_metric_mse_mean, ckpt_manager_epoch, train_params, model_params )
                            
        #region Early iteration Stop Check
        if epoch > ( max( df_training_info.loc[:, 'Epoch'], default=0 ) + train_params['early_stopping_period']) :
            print("Model Early Stopping at EPOCH {}".format(epoch))
            break
        #endregion
            
    # endregion

    print("Model Training Finished")

# endregion 



if __name__ == "__main__":
    s_dir = utility.get_script_directory(sys.argv[0])

    args_dict = utility.parse_arguments(s_dir)

    if(args_dict['model_name'] == "DeepSD"):
        train_params = hparameters.train_hparameters( **args_dict )

        
        #stacked DeepSd methodology
        li_input_output_dims = [ {"input_dims": [39, 88 ], "output_dims": [98, 220 ] , 'var_model_type':'guassian_factorized' } ,
                    {"input_dims": [98, 220 ] , "output_dims": [ 156, 352 ] , 'conv1_inp_channels':1, 'var_model_type':'guassian_factorized' }  ]

        model_params = [ hparameters.model_deepsd_hparameters(**_dict) for _dict in li_input_output_dims  ]
        model_params = [ mp() for mp in model_params]
    
    elif(args_dict['model_name'] == "THST"):

        train_params = hparameters.train_hparameters_ati( **args_dict )
        model_params = hparameters.model_THST_hparameters()()
        

    train_loop(train_params(), model_params )

    

