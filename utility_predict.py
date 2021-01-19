import tensorflow as tf
import pandas as pd
import models
import os
import pickle
import glob
import utility
import numpy as np
import time

def load_model(t_params, m_params):
    """Loads a model

    Args:
        t_params ([type]): [description]
        m_params ([type]): [description]

    Returns:
        [type]: [description]
    """    
    model = None
    model_name = m_params['model_name']

    if(model_name=="TRUNET"):
        model = models.TRUNET(t_params, m_params)
        inp_shape = [t_params['batch_size'], t_params['lookback_feature']] + m_params['region_grid_params']['outer_box_dims'] + [len(t_params['vars_for_feature'])]
        init_inp = tf.zeros(inp_shape, dtype=tf.float16 )
        model(init_inp, training=False )
    
    elif(model_name=="HCGRU"):
        model = models.HCGRU(t_params,m_params)
        inp_shape = [t_params['batch_size'], t_params['lookback_feature']] + m_params['region_grid_params']['outer_box_dims'] + [len(t_params['vars_for_feature'])]
        init_inp = tf.zeros(inp_shape, dtype=tf.float16 )
        model(init_inp, training=False )

    elif(model_name=="UNET"):
        model = models.UNET(t_params,m_params)
        inp_shape = [t_params['batch_size'] ] + m_params['region_grid_params']['outer_box_dims'] + [int(t_params['lookback_feature']*len(t_params['vars_for_feature']))]
        init_inp = tf.zeros(inp_shape, dtype=tf.float16 )
        model(init_inp, training=False )

    ckpt = tf.train.Checkpoint(model=model)

    # Choosing checkpoint with lowest validation loss
    df_checkpoint_scores = pd.read_csv( t_params['script_dir']+'/checkpoints/{}/checkpoint_scores.csv'.format(utility.model_name_mkr(m_params, train_test="train", t_params=t_params, htuning=m_params.get('htuning',False)  )), header=0 )

    best_checkpoint_path = df_checkpoint_scores['Checkpoint_Path'][0]
    checkpoint_code = "E"+str(df_checkpoint_scores['Epoch'][0])
    
    status = ckpt.restore( best_checkpoint_path ).expect_partial()

    print("Are weights empty after restoring from checkpoint?", model.weights == [] )

    return model, checkpoint_code

def save_preds( t_params, m_params, li_preds, li_timestamps, li_truevalues, custom_test_loc=None, count=0 ):
    """Save predictions to file

        Args:
            t_params (dict): dictionary for train/test params
            m_params (dict): dictionary for m params
            li_preds (list): list of predictions
            li_timestamps (list): corresponding list of timestamps
            li_truevalues (list): corresponding list of true values
            custom_test_loc ([type], optional): [description]. Defaults to None.

        Returns:
            bool
    """ 

    li_preds = [ np.where(tnsr<0.0, 0.0, tnsr) for tnsr in li_preds ]        
    #li_preds = [ tnsr.numpy() for tnsr in li_preds   ] #list of preds: (tss, samples ) or (tss, h, w, samples )

    if custom_test_loc in ["All"] or t_params['t_settings'].get('region_pred', False)==True:
        li_truevalues = [ np.squeeze(tens,axis=1) for tens in li_truevalues]                   #2D - (tss, h, w)
    else:
        li_truevalues = [ np.reshape(tens, [-1]) for tens in li_truevalues]     #list of 1D - (preds ) 
                                
    li_timestamps = [ np.array(_li).reshape([-1]) for _li in li_timestamps ]
    data_tuple = (li_timestamps, li_preds, li_truevalues)
    t1 = time.strftime('%Y-%m-%d', time.localtime(li_timestamps[0][0]))
    t2 = time.strftime('%Y-%m-%d', time.localtime(li_timestamps[-1][-1]))

    # Savging to pickle
    if type( t_params['ctsm_test'] ) == str:
        _path_pred = t_params['output_dir'] + "/{}/Predictions".format(utility.model_name_mkr(m_params, train_test="test", t_params=t_params, custom_test_loc=custom_test_loc, htuning=m_params.get('htuning',False) ))
    
    if t_params['t_settings'].get('region_pred', False) == True:
        fn = f"_regional"
    else:
        fn = f"local"
    
    if count >0:
        fn += f"_chunk{count}"

    fn += ".dat"

    if(not os.path.exists(_path_pred) ):
        os.makedirs(_path_pred)
    
    pickle.dump( data_tuple, open( _path_pred + "/" +fn ,"wb"), protocol=4 )



    print("Saved predictions\t", t1, "--", t2)
    return True

def load_predictions_gen(_path_pred):
    li_pred_fns = list( glob.glob(_path_pred+"/*") )
    li_pred_fns = [pred_fns for pred_fns in li_pred_fns if pred_fns[-4:]!="json" ]
    for pred_fn in li_pred_fns:
        pred = pickle.load(open(pred_fn,"rb"))
        yield pred # list of lists; each sublist [ts, [stochastic preds], true] #shape ( x, [(width, hieght),.. ], (width, hieght) )
