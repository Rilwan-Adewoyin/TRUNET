import numpy as np
import math
import tensorflow as tf

#precip data import - use in data pipeline
def read_prism_precip(bil_path, hdr_path=None, hdr_known=True, tensorf = True):
    """
        Read an array from ESRI BIL raster file using Info from the hdr file too
        https://pymorton.wordpress.com/2016/02/26/plotting-prism-bil-arrays-without-using-gdal/
    """
    if( tensorf ):
        bil_path = bil_path.numpy()

    if(hdr_known):
        NROWS = 621
        NCOLS = 1405
        NODATA_VAL = float(-9999)
    else:
        hdr_dict = read_hdr(hdr_path)
        NROWS = int(hdr_dict['NROWS'])
        NCOLS = int(hdr_dict['NCOLS'])
        NODATA_VAL = hdr_dict['NODATA']
    # For now, only use NROWS, NCOLS, and NODATA
    # Eventually use NBANDS, BYTEORDER, LAYOUT, PIXELTYPE, NBITS
 
    prism_array = np.fromfile(bil_path, dtype='<f4')
    prism_array = prism_array.astype( np.float32 )
    prism_array = prism_array.reshape( NROWS , NCOLS )
    prism_array[ prism_array == float(NODATA_VAL) ] = np.nan
    return prism_array

#prism data import
def read_prism_elevation(bil_path, hdr_path=None, hdr_known=True):
    if(hdr_known):
        NROWS = 6000
        NCOLS = 4800
        NODATA_VAL = float(-9999)
    else:
        hdr_dict = read_hdr(hdr_path)
        NROWS = int(hdr_dict['NROWS'])
        NCOLS = int(hdr_dict['NCOLS'])
        NODATA_VAL = int(hdr_dict['NODATA'])
    # For now, only use NROWS, NCOLS, and NODATA
    # Eventually use NBANDS, BYTEORDER, LAYOUT, PIXELTYPE, NBITS
 
    _array = np.fromfile(open(bil_path,"rb"), dtype='>i2')
    _array = _array.astype(dtype=np.float32)
    _array = _array.reshape( NROWS , NCOLS )
    _array[ _array == NODATA_VAL ] = np.nan
    return _array

def read_hdr(hdr_path):
    """Read an ESRI BIL HDR file"""
    with open(hdr_path, 'r') as input_f:
        header_list = input_f.readlines()
    return dict(item.strip().split() for item in header_list)


# Miscallaneous
def replace_inf_nan(_tensor):
    nan_bool_ind_tf = tf.math.is_nan( tf.dtypes.cast(_tensor,dtype=tf.float32 ) )
    inf_bool_ind_tf = tf.math.is_inf( tf.dtypes.cast( _tensor, dtype=tf.float32 ) )

    bool_ind_tf = tf.math.logical_or( nan_bool_ind_tf, inf_bool_ind_tf )
    
    _tensor = tf.where( bool_ind_tf, tf.constant(0.0,dtype=tf.float32), _tensor )
    return tf.dtypes.cast(_tensor, dtype=tf.float32)


# Methods Related to Training
def update_checkpoints_epoch(df_training_info, epoch, train_metric_mse_mean_epoch, val_metric_mse_mean, ckpt_manager_epoch, train_params, model_params  ):

    df_training_info = df_training_info[ df_training_info['Epoch'] != epoch ] #rmv current batch records for compatability with code below
    if( ( val_metric_mse_mean.result().numpy() <= max( df_training_info.loc[ : ,'Val_loss_MSE' ], default= val_metric_mse_mean.result().numpy()+1 ) ) ):
        print('Saving Checkpoint for epoch {}'.format(epoch)) 
        ckpt_save_path = ckpt_manager_epoch.save()

        
        # Possibly removing old non top5 records from end of epoch
        if( len(df_training_info.index) >= train_params['checkpoints_to_keep'] ):
            df_training_info = df_training_info.sort_values(by=['Val_loss_MSE'], ascending=False)
            df_training_info = df_training_info.iloc[:-1]
            df_training_info.reset_index(drop=True)

        
        df_training_info = df_training_info.append( other={ 'Epoch':epoch,'Train_loss_MSE':train_metric_mse_mean_epoch.result().numpy(), 'Val_loss_MSE':val_metric_mse_mean.result().numpy(),
                                                            'Checkpoint_Path': ckpt_save_path, 'Last_Trained_Batch':-1 }, ignore_index=True ) #A Train batch of -1 represents final batch of training step was completed

        print("\nTop {} Performance Scores".format(train_params['checkpoints_to_keep']))
        print(df_training_info[['Epoch','Val_loss_MSE']] )
        df_training_info = df_training_info.sort_values(by=['Val_loss_MSE'], ascending=True)
        df_training_info.to_csv( path_or_buf="checkpoints/{}/checkpoint_scores_model_{}.csv".format(model_params['model_name'],train_params['model_version']), header=True, index=False ) #saving df of scores                      
    return df_training_info


def kl_loss_weighting_scheme( max_batch ):
    return 1/max_batch