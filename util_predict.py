import tensorflow as tf
import pandas as pd
import models
import os
import pickle
import glob
import utility
import numpy as np
import time

def load_model(test_params, model_params):
    model = None
    model_name = model_params['model_name']

        # Option 2 - From checkpoint and model.py
    if model_params['model_type_settings']['var_model_type'] == "mc_dropout" and int(model_params['model_type_settings']['model_version'])<100 :
        mode = "mc_dropout_test"
    else:
        mode = "Generic"

    if(test_params['model_recover_method'] == 'checkpoint_batch'):
        
        if(model_name=="DeepSD"):
            model = models.SuperResolutionModel( test_params, model_params) 
            if type(model_params)== list:
                model_params = model_params[0]

                #Just initliazing model so checkpoint method can work
            init_inp = tf.ones( [ test_params['batch_size'], model_params['input_dims'][0],
                        model_params['input_dims'][1], model_params['conv1_inp_channels'] ] , dtype=tf.float16 )
            model(init_inp, training=False )

        elif(model_name=="THST"):
            model = models.THST(test_params, model_params)
            
            if model_params['model_type_settings']['location'] == "wholeregion":
                init_inp = tf.zeros(
                    [test_params['batch_size'], model_params['data_pipeline_params']['lookback_feature'] , 100,  140, 6 ], dtype=tf.float16 )           
            else:
                init_inp = tf.zeros(
                    [test_params['batch_size'], model_params['data_pipeline_params']['lookback_feature'] ,16 , 16,6 ], dtype=tf.float16 )
            
            model(init_inp, training=False )
        
        elif(model_name in ["SimpleLSTM","SimpleGRU"] ):
            model = models.SimpleGRU(test_params, model_params)
            init_inp = tf.zeros( [test_params['batch_size'], model_params['data_pipeline_params']['lookback_feature'], 6 ], dtype=tf.float16 )
            model( init_inp, training=False )
        
        elif(model_name=="SimpleDense"):
            model = models.SimpleDense(test_params, model_params)
            init_inp = tf.zeros( [test_params['batch_size'], model_params['data_pipeline_params']['lookback_feature'], 6 ], dtype=tf.float16 )
            model( init_inp, training=False )
        
        elif(model_name=="SimpleConvGRU"):
            model = models.SimpleConvGRU(test_params,model_params)
            if model_params['model_type_settings']['location'] == "wholeregion":
                init_inp = tf.zeros(
                    [test_params['batch_size'], model_params['data_pipeline_params']['lookback_feature'] , 100,  140, 6 ], dtype=tf.float16 )
            else:
                init_inp = tf.zeros(
                    [test_params['batch_size'], model_params['data_pipeline_params']['lookback_feature'] ,16 , 16, 6 ], dtype=tf.float16 )
            model(init_inp, training=False )

        checkpoint_path = test_params['script_dir']+"/checkpoints/{}/batch".format(utility.model_name_mkr(model_params,mode=mode,train_params=test_params))

        ckpt = tf.train.Checkpoint(att_con=model)
        checkpoint_code = "B"+ str(tf.train.latest_checkpoint(checkpoint_path)[-5:])
        status = ckpt.restore( tf.train.latest_checkpoint(checkpoint_path) ).expect_partial()

        print("Are weights empty after restoring from checkpoint?", model.weights == [])
    
    elif(test_params['model_recover_method'] == 'checkpoint_epoch'):
        

        if(model_name=="DeepSD"):
            #Just initliazing model so checkpoint method can work
            model = models.SuperResolutionModel( test_params, model_params)
            if type(model_params) == list:
                model_params = model_params[0]

            init_inp = tf.ones( [ test_params['batch_size'], model_params['input_dims'][0],
                        model_params['input_dims'][1], model_params['conv1_inp_channels'] ] , dtype=tf.float16 )
            model(init_inp, training=False )

        elif(model_name=="THST"):
            model = models.THST(test_params, model_params)
            if model_params['model_type_settings']['location'] == "wholeregion":
                init_inp = tf.zeros(
                    [test_params['batch_size'], model_params['data_pipeline_params']['lookback_feature'] , 100,  140, 6 ], dtype=tf.float16 )
            else: #elif model_params['model_type_settings']['location'] == "region_grid":
                    init_inp = tf.zeros(
                        [test_params['batch_size'], model_params['data_pipeline_params']['lookback_feature'] ,16 , 16,6 ], dtype=tf.float16 )
            model(init_inp, training=False )

        elif(model_name=="SimpleGRU"):
            model = models.SimpleGRU(test_params, model_params)
            init_inp = tf.zeros( [test_params['batch_size'], model_params['data_pipeline_params']['lookback_feature'], 6 ], dtype=tf.float16 )
            model( init_inp, training=False )
        
        elif(model_name=="SimpleDense"):
            model = models.SimpleDense(test_params, model_params)
            init_inp = tf.zeros( [test_params['batch_size'], model_params['data_pipeline_params']['lookback_feature'], 6 ], dtype=tf.float16 )
            model( init_inp, training=False )
        
        elif(model_name=="SimpleConvGRU"):
            model = models.SimpleConvGRU(test_params,model_params)

            if model_params['model_type_settings']['location'] == "wholeregion":
                init_inp = tf.zeros(
                    [test_params['batch_size'], model_params['data_pipeline_params']['lookback_feature'] , 100,  140, 6 ], dtype=tf.float16 )
            else:
                    init_inp = tf.zeros(
                        [test_params['batch_size'], model_params['data_pipeline_params']['lookback_feature'] ,16 , 16,6 ], dtype=tf.float16 )


            model(init_inp, training=False )
        
        ckpt = tf.train.Checkpoint(model=model)

        #We will use Optimal Checkpoint information from checkpoint_scores_model.csv
        df_checkpoint_scores = pd.read_csv( test_params['script_dir']+'/checkpoints/{}/checkpoint_scores.csv'.format(utility.model_name_mkr(model_params,mode=mode, load_save="load", train_params=test_params )), header=0  ) #Give the load_save param a better name
        best_checkpoint_path = df_checkpoint_scores['Checkpoint_Path'][0]
        checkpoint_code = "E"+str(df_checkpoint_scores['Epoch'][0])
        status = ckpt.restore( best_checkpoint_path ).expect_partial()

        print("Are weights empty after restoring from checkpoint?", model.weights == [] )

    return model, checkpoint_code

def save_preds( test_params, model_params, li_preds, li_timestamps, li_truevalues, precip_threshold=None ):
    """
    
    """
    if type(model_params) == list:
        model_params = model_params[0]

    if test_params.get('ctsm',None) ==None:
        _path_pred = test_params['output_dir'] + "/{}/Predictions".format(utility.model_name_mkr(model_params, load_save="save", train_params=test_params))
    
    elif test_params['ctsm'] == "Rolling_2_Year_test" :
        _path_pred = test_params['output_dir'] + "/{}/Predictions/R2yt".format(utility.model_name_mkr(model_params, load_save="save", train_params=test_params) )
    
    elif test_params['ctsm'] == "4ds_10years":
        _path_pred = test_params['output_dir'] + "/{}/Predictions/4ds_{}".format(utility.model_name_mkr(model_params, load_save="save", train_params=test_params), test_params['fyi_test'] )
    
    if model_params['model_type_settings']['discrete_continuous'] == False:
        fn = str(li_timestamps[0][0]) + "___" + str(li_timestamps[-1][-1]) + ".dat"
    else:
        fn = str(li_timestamps[0][0]) + "___" + str(li_timestamps[-1][-1]) + "pt{:.3f}.dat".format(precip_threshold)

    if(not os.path.exists(_path_pred) ):
        os.makedirs(_path_pred)
    
    li_preds = [ tf.where(tnsr<0, 0.0, tnsr) for tnsr in li_preds ]
    li_preds = [ tnsr.numpy() for tnsr in li_preds   ] #list of 1D - (tss, preds_dim ) 2D-(samples, tss, h, w )


    if( model_params['model_name'] in ["SimpleLSTM", "SimpleGRU" ,"SimpleDense"]):
        li_truevalues = [ tens.numpy().reshape([-1]) for tens in li_truevalues]     #list of 1D - (tss, preds_dim ) 

    elif( model_params['model_name'] in ["SimpleConvLSTM", "THST", "SimpleConvGRU"] ): 
        if "location_test" in model_params['model_type_settings'].keys():
            li_truevalues = [ tens.numpy().reshape([-1]) for tens in li_truevalues]     #list of 1D - (preds ) 
        else:
            li_truevalues = [ tens.numpy() for tens in li_truevalues] #2D - (tss, h, w)

    elif( model_params['model_name'] in ["DeepSD"] ): 
        li_truevalues = [ tens.numpy() for tens in li_truevalues]
    
    li_timestamps = [ np.array(_li).reshape([-1]) for _li in li_timestamps ]
    data_tuple = (li_timestamps, li_preds, li_truevalues)

    pickle.dump( data_tuple, open( _path_pred + "/" +fn ,"wb"), protocol=4 )
    
    t1 = time.strftime('%Y-%m-%d', time.localtime(li_timestamps[0][0]))
    t2 = time.strftime('%Y-%m-%d', time.localtime(li_timestamps[-1][-1]))
    print("Saved predictions\t", t1, "--", t2)
    return True

def load_predictions_gen(_path_pred):
    li_pred_fns = list( glob.glob(_path_pred+"/*") )
    li_pred_fns = [pred_fns for pred_fns in li_pred_fns if pred_fns[-4:]!="json" ]
    for pred_fn in li_pred_fns:
        pred = pickle.load(open(pred_fn,"rb"))
        yield pred # list of lists; each sublist [ts, [stochastic preds], true] #shape ( x, [(width, hieght),.. ], (width, hieght) )
