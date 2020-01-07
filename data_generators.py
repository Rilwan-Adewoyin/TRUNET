import itertools
import tensorflow as tf
import numpy as np
import pickle
import glob
import utility


def load_data( elements_to_skip, hparams, _num_parallel_calls =tf.data.experimental.AUTOTUNE, data_dir="./Data"):

    # region prepare elevation Preprocess
    _path = data_dir+"/Preprocessed/elevation.pkl" #TODO:(akanni-ade) change to value passed in via a h-parameters dictionary
    with open(_path,"rb") as f: #TODO:(akanni-ade) change to value passed in via a h-parameters dictionary
        arr_elev = pickle.load(f)
        
    arr_elev = arr_elev[::4, ::4]  #shape( 156,352 ) #16kmby16km
        #creating layered representation of 16kmby16km arr_elev such that it is same shape as 64kmby64km precip
            ##take cell i,j in 2d array. each cell in the square matrix around cell i,j is stacked underneath i,j. 
            ## The square has dimensions (rel to i,j): 2 to the right, 2 down, 1 left, 1 right
            ## This achieves a dimension reduction of 4

    AVG_ELEV = np.nanmean(arr_elev) #TODO: (akanni-ade) Find actual max elev
    arr_elev = arr_elev / AVG_ELEV 
        #now making all elevation values above 1, then converting all nan values to 0
    arr_elev = arr_elev + np.min(arr_elev) + 1
    arr_elev = np.nan_to_num( arr_elev, nan=0.0, posinf=0.0, neginf=0.0 )
    # endregion

    def stacked_reshape( arr, first_centre, downscale_x, downscale_y, batch_size = hparams['batch_size'] ):
        """
            This produces a list of tiled arrays. This ensures higher resolution data _arr has the same shape as lower resolution data, ignoring a depth dimension
            i.e.

            The array is stacked to create a new dimension (axis=-1). The stack happens on the following cells:
                first centre (i,j)
                (i+n*downscale_x, j+n*downscale_y ) for integer n

            :param nparray arr: 2D array  #shape( 156,352 ) #16kmby16km
            :param tuple first centre: tuple (i,j) indexing where the upperleft most position to stack on
            

            returns arr_stacked
        """
        # region standardisation
        
        MAX_ELEV = 2500 #TODO: (akanni-ade) Find true value, incorporate into Hparams
        arr = arr / MAX_ELEV 
        # end region

        new_depth = downscale_x * downscale_y
        dim_x, dim_y = arr.shape
        li_arr = []

        idxs_x = list( range(downscale_x) )
        idxs_y = list( range(downscale_y) )
        starting_idxs = list( itertools.product( idxs_x, idxs_y ) )

        for x,y in starting_idxs:
            #This helps define the different boundaries for the tiled elevation
            end_x = dim_x - ( downscale_x-first_centre[0] - x) 
            end_y = dim_y - ( downscale_y-first_centre[1] - y)
            arr_cropped = arr[ x:end_x, y:end_y ]

            li_arr.append(arr_cropped)

        li_tnsr = [ tf.expand_dims(_arr[::downscale_x, ::downscale_y],0) for _arr in li_arr ]
        li_tnsr_elev =  [ tf.tile(_tnsr,[batch_size,1,1]) for _tnsr in li_tnsr ]
        #tnsr_elev_tiled = tf.stack(li_tnsr_elev, axis=-1)
        
        #arr_stacked = np.stack( li_arr, axis=-1 )
        #arr_stacked = arr_stacked[::downscale_x, ::downscale_y] 

        return li_tnsr_elev

    li_elev_tiled = stacked_reshape( arr_elev, (1,1), 4, 4  ) # list[ (1, 39, 88), (1, 39, 88), ... ] #16kmby16km 

    # endregion

    # region features, targets
    _dir_precip = data_dir+"/PRISM/daily_precip"
    file_paths_bil = list( glob.glob(_dir_precip+"/*/*.bil" ) )
    file_paths_bil.sort(reverse=False)

    if elements_to_skip !=None:
        file_paths_bil =  file_paths_bil[elements_to_skip: ]

    ds_fns_precip = tf.data.Dataset.from_tensor_slices(file_paths_bil)

    ds_precip_imgs = ds_fns_precip.map( lambda fn: tf.py_function(utility.read_prism_precip,[fn], [tf.float32] ) )#, num_parallel_calls=_num_parallel_calls ) #shape(bs, 621, 1405) #4km by 4km

    ds_precip_imgs = ds_precip_imgs.batch(hparams['batch_size'] ,drop_remainder=True)

    def features_labels_mker( arr_images, li_elev_tiled=li_elev_tiled ):
        """Produces the precipitation features and the target for training
        shape(bs, rows, cols)
        """
        #standardisation and preprocess of nans
        MAX_RAIN = 200 #TODO:(akanni-ade) Find actual max rain
        arr_images = arr_images / MAX_RAIN
        arr_images = arr_images + 1 #TODO:(akanni-ade) remember to undo these preprocessing steps when looking to predict the future
        arr_images = utility.replace_inf_nan(arr_images)

        #features
        precip_feat = reduce_res( arr_images, 16, 16 ) #shape(bs, 621/16, 1405/16) (bs, 39, 88)  64km by 64km


        feat = tf.stack( [precip_feat,*li_elev_tiled], axis=-1 ) #shape(bs, 39, 88, 17)  64km by 64km

        #targets        
        precip_tar = reduce_res( arr_images, 4, 4)   #shape( bs, 621/4, 1405/4 ) (bs,156,352) 16km by 16km

        #TODO(akanni-ade): consider applying cropping to remove large parts that are just water 
            # #cropping
            # precip_tar = precip_tar[:, : , : ]
            # feat = feat[:, :, :, :]

        return feat, precip_tar

    def reduce_res(arr_imgs, x_axis, y_axis):
        arr_imgs_red = arr_imgs[:,::x_axis, ::y_axis]
        return arr_imgs_red
    
    ds_precip_feat_tar = ds_precip_imgs.map( features_labels_mker)#, num_parallel_calls=_num_parallel_calls ) #shape( (bs, 39, 88, 17 ) (bs,156,352) )
    ds_precip_feat_tar = ds_precip_feat_tar
        # endregion

    ds_precip_feat_tar = ds_precip_feat_tar.prefetch(buffer_size=_num_parallel_calls)

    return ds_precip_feat_tar #shape( (bs, 39, 88, 17 ) (bs,156,352) )
