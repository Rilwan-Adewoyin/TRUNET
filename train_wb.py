import xarray as xr

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
import tensorflow as tf
import tenorflow.keras as keras
import tensorflow_addons as tfa

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utility
import datagen
import models
from score import *

tf.keras.backend.set_floatx('float16')
tf.keras.backend.set_epsilon(1e-3)
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

def main(t_params, h_params):
    
    #Load Dataset
    ds_train, ds_val, ds_test = datagen.make_datasets( t_params )
    ds_train = ds_train.cache(t_params['data_dir']+"/ds_cache/ds_train")
    ds_val = ds_val.cache(t_params['data_dir']+"/ds_cache/ds_val")
     
    #Load Model In
    #HERE: Making model
    model = models.model_loader( None, m_params )
    #Consider uing custom loss based on similarity of patterns in convolutional layer
    #Use a custom loss which only calculates the loss on the third and fifth prediction

    #TODO: build 3 custom losses
        #loss 1: Simply based on MSE @Day 3 and 5
        #loss 2: Based on Perceptual Loss @Day 3 and 5
        #loss 3: Perceptual Loss w/ certain days masked similar to BERT Language model
    model.compile( optimizer=tfa.optimizers.RectifiedAdam( t_params['rec_adam_params']),
                    loss=custom_loss_3_5, metrics=' ' )

    print(model.summary())

    #Train model
    model.compile( tfa.optimizers.RectifiedAdam( **m_params['rec_adam_params'] ) )

    model.fit( ds_train, epochs=1, verbose=1, #t_params['epochs']
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    min_delta=0,
                    patience=t_params['patience'],
                    verbose=1,
                    mode='auto')],
                validation_data = ds_val,
                ) #use_multiprocessing=True
    
    print(f"Saving model weights: saved_models/mv_{m_params['model_version']}")
    model.save_weights(f"saved_models/mv_{m_params['model_version']}")

    #TODO: edit this prediction script to make it work, your model is not iterative for 5 days
    pred = create_predictions(model, ds_test)
    print(f"Saving predictions: Output/Predictions/mv_{m_params['model_version']}")
    pred.to_netcdf("Output/Predictions/mv_{}")

    z500_test = load_test_data(f'{t_params["data_dir"]}geopotential_500_5.265deg.nc', 'z')
    t850_test = load_test_data(f'{t_params["data_dir"]}temperature_850_5.265deg.nc', 't')
    
    #Need to edit this so only loss for the values at 3 days and 5 days are calculated
    #TODO:(akanni-ade) fix this method below
    z_weighted_rmse = compute_weighted_rmse(pred.z, z500_test).load()
    t_weighted_rmse = compute_weighted_rmse(pred.t, t850_test).load()

    scores_dict = pd.DataFrame( {
        "z_rmse_3_day":[z_weighted_rmse_3day],
        "z_rmse_5_day":[z_weighted_rmse_5day],
        "t_rmse_3_day":[t_weighted_rmse_3day],
        "t_rmse_5_day":[t_weighted_rmse_5day]
    } )

def create_predictions(model, dg):
    """Create non-iterative predictions"""
    preds = model.predict_generator(dg)
    # Unnormalize
    preds = preds * dg.std.values + dg.mean.values
    das = []
    lev_idx = 0
    for var, levels in dg.var_dict.items():
        if levels is None:
            das.append(xr.DataArray(
                preds[:, :, :, lev_idx],
                dims=['time', 'lat', 'lon'],
                coords={'time': dg.valid_time, 'lat': dg.ds.lat, 'lon': dg.ds.lon},
                name=var
            ))
            lev_idx += 1
        else:
            nlevs = len(levels)
            das.append(xr.DataArray(
                preds[:, :, :, lev_idx:lev_idx+nlevs],
                dims=['time', 'lat', 'lon', 'level'],
                coords={'time': dg.valid_time, 'lat': dg.ds.lat, 'lon': dg.ds.lon, 'level': levels},
                name=var
            ))
            lev_idx += nlevs
    return xr.merge(das)

class custom_loss_3_5( tf.keras.losses.Loss ):

    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        return K.mean(math_ops.square(y_pred[:,11:20:8,:,:,:] - y_true[:,11:20:8,:,:,:]), axis=-1)

if __name__ == '__main__':

    args_dict = utility.parse_arguments()

    m_params, t_params  = utility.load_params(args_dict)

    """
    -mp 
        "model_name"
        "dropout"
        "inp_dropout"
        "rec_dropout"
        


    """

    """
    -tsp
        "data_dir"
        "batch_size"
        "outp_dir"

    """