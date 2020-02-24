import itertools
import numpy as np
from netCDF4 import Dataset, num2date
import tensorflow as tf
import pickle
import glob
import utility
import json



# region Vandal related

def load_data_vandal( elements_to_skip, hparams, m_params,_num_parallel_calls =-1, data_dir="./Data", drop_remainder=True):

    # region prepare elevation Preprocess
    _path = data_dir+"/Preprocessed/elevation.pkl" #TODO:(akanni-ade) change to value passed in via a h-parameters dictionary
    with open(_path,"rb") as f: #TODO:(akanni-ade) change to value passed in via a h-parameters dictionary
        arr_elev = pickle.load(f)
        
    arr_elev = arr_elev[::4, ::4]  #shape( 156,352 ) #16kmby16km
        #creating layered representation of 16kmby16km arr_elev such that it is same shape as 64kmby64km precip
            ##take cell i,j in 2d array. each cell in the square matrix around cell i,j is stacked underneath i,j. 
            ## The square has dimensions (rel to i,j): 2 to the right, 2 down, 1 left, 1 right
            ## This achieves a dimension reduction of 4

    AVG_ELEV = np.nanmean(arr_elev) 
    VAR_ELEV = np.nanvar(arr_elev)

    arr_elev = ( arr_elev - AVG_ELEV ) / VAR_ELEV
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
    file_paths_bil1 = list( glob.glob(_dir_precip+"/*/PRISM_tdmean_stable*.bil" ) )
    file_paths_bil2 = list( glob.glob(_dir_precip+"/*/PRISM_ppt_stable*.bil" ) )
    file_paths_bil1.sort(reverse=False)
    file_paths_bil2.sort(reverse=False)
    file_paths_bil = file_paths_bil1 + file_paths_bil2

    if elements_to_skip !=None:
        file_paths_bil =  file_paths_bil[elements_to_skip: ]

    ds_fns_precip = tf.data.Dataset.from_tensor_slices(file_paths_bil)

    ds_precip_imgs = ds_fns_precip.map( lambda fn: tf.py_function(utility.read_prism_precip,[fn], [tf.float32] ) )#, num_parallel_calls=_num_parallel_calls ) #shape(bs, 621, 1405) #4km by 4km

    ds_precip_imgs = ds_precip_imgs.batch(hparams['batch_size'] ,drop_remainder=drop_remainder)

    def features_labels_mker( arr_images, li_elev_tiled=li_elev_tiled ):
        """Produces the precipitation features and the target for training
        shape(bs, rows, cols)
        """
        #standardisation and preprocess of nans
        MAX_RAIN = 200 #TODO:(akanni-ade) Find actual max rain
        arr_images = utility.standardize(arr_images, distr_type=m_params['model_type_settings']['distr_type'])
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

# endregion

# region ATI

"""
    Example of how to use
    import Generators

    rr_ens file 
    _filename = "Data/Rain_Data/rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_djf_uk.nc"
    rain_gen = Generator_rain(_filename, all_at_once=True)
    data = next(iter(grib_gen))

    Grib Files
    _filename = 'Data/Rain_Data/ana_coarse.grib'
    grib_gen = Generators.Generator_grib(fn=_filename, all_at_once=True)
    data = next(iter(grib_gen))

    Grib Files Location:
    _filename = 'Data/Rain_Data/ana_coarse.grib'
    grib_gen = Generators.Generator_grib(fn=_filename, all_at_once=True)
    arr_long, arr_lat = grib_gen.locaiton()
    #now say you are investingating the datum x = data[15,125]
    #   to get the longitude and latitude you must do
    long, lat = arr_long(15,125), arr_lat(15,125)

"""

class Generator():
    
    def __init__(self, fn = "", all_at_once=False, channel=None):
        self.generator = None
        self.all_at_once = all_at_once
        self.fn = fn
        self.channel = channel
        self.city_location = {
            "London": [51.5074, -0.1278],
            #+0.15,-0.1 to avoid masking a coastal city
            "Cardiff": [51.4816 + 0.15, -3.1791 -0.05], 
            "Edinburgh": [55.9533, -3.1883],
            "Belfast": [54.5973, -5.9301],
            "Dublin": [53.3498, -6.2603]}
        self.latitude_array = np.linspace(58.95, 49.05, 100)
        self.longitude_array = np.linspace(-10.95, 2.95, 140)
        
    
    def yield_all(self):
        pass

    def yield_iter(self):
        pass

    def __call__(self):
        if(self.all_at_once):
            return self.yield_all()
        else:
            return self.yield_iter()
    
    def find_idx_of_city(self, city="London"):
        coordinates = self.city_location[city]
        return self.find_nearest_latitude_longitude( coordinates)

    def find_nearest_latitude_longitude(self, coordinates):

        latitude_index =    np.abs(self.latitude_array - coordinates[0] ).argmin()
        longitude_index =   np.abs(self.longitude_array - coordinates[1]).argmin()

        return(latitude_index, longitude_index)
    
class Generator_rain(Generator):
    def __init__(self, **generator_params):
        super(Generator_rain, self).__init__(**generator_params)
        self.data_len = 14822

    def yield_all(self,):
        with Dataset(self.fn, "r", format="NETCDF4",keepweakref=True) as f:
            _data = f.variables['rr'][:]
            yield np.ma.getdata(_data), np.ma.getmask(_data)   
            
    def yield_iter(self):
        fn = self.fn

        f = Dataset(fn, "r", format="NETCDF4", keepweakref=True)
        for chunk in f.variables['rr']:
            data = np.ma.getdata(chunk)
            mask = np.ma.getmask(chunk)
            yield data , mask


    def __call__(self):
        return self.yield_iter()
    
class Generator_mf(Generator):
    """
        Creates a generator for the model_fields_data
    
        :param all_at_once: whether to return all data, or return data in batches

        :param channel: the desired channel of information in the grib file
            Default None, then concatenate all channels together and return
            If an integer return this band
    """

    def __init__(self, **generator_params):

        super(Generator_mf, self).__init__(**generator_params)

        self.vars_for_feature = ['unknown_local_param_137_128', 'unknown_local_param_133_128', 'air_temperature', 'geopotential', 'x_wind', 'y_wind' ]
        self.channel_count = len(self.vars_for_feature)
        self.data_len = 16072
        
    def yield_all(self):
        raise NotImplementedError
    
    def yield_iter(self):

        
        f = Dataset(self.fn, "r", format="NETCDF4")
        for tuple_mfs in zip( *[f.variables[var_name] for var_name in self.vars_for_feature] ):
            list_datamask = [(np.ma.getdata(_mar), np.ma.getmask(_mar) ) for _mar in tuple_mfs]
            _data, _masks = list(zip(*list_datamask))
            _masks = [ np.full(_data[0].shape, np.logical_not(_mask_val) , dtype=bool) for _mask_val in _masks ] #projecting masks from the (6,) shape to a square array shape

            stacked_data = np.stack(_data, axis=-1)
            stacked_masks = np.stack(_masks, axis=-1)
            
            yield stacked_data[ -2:1:-1 , 2:-2, :], stacked_masks[ -2:1:-1 , 2:-2, :] #(h,w,6) #(h,w,6)  #this aligns it to rain 
    
    def __call__(self):
        return self.yield_iter()

def load_data_ati(t_params, m_params, target_datums_to_skip=None, day_to_start_at=None, _num_parallel_calls=-1,
                    data_dir="./Data/Rain_Data_Nov19", mode="Normal" ):
    """
        This is specific for the following two files -> ana_input_1.nc and rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_uk.nc
        :param day_to_start_at: should be of type np.datetime64

    """
    # region Data Syncing/Skipping
    def get_start_idx( start_date, t_params ):
        """
            :param start_date: a numpy datetime object
        """

        feature_start_date = t_params['feature_start_date']
        target_start_date = t_params['target_start_date']

        feat_days_diff = ( np.timedelta64(start_date - feature_start_date,'6h') ).astype(int)
        tar_days_diff = ( np.timedelta64(start_date - target_start_date, 'D') ).astype(int)

        feat_start_idx = feat_days_diff #since the feature comes in four hour chunks
        tar_start_idx = tar_days_diff 

        return feat_start_idx, tar_start_idx

    if(target_datums_to_skip!=None):
        raise NotImplementedError
    
    elif(day_to_start_at!=None):
        start_idx_feat, start_idx_tar = get_start_idx(day_to_start_at, t_params)

    else:
        raise ValueError("Either one of target_datums_to_skip or day_to_start_from must not be None")
    # endregion


    # region prepare target_rain_data
    fn_rain = data_dir+"/rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_uk.nc"
    #fn_rain = "/home/u1862646/ATI/BNN/Data/Rain_Data_Nov19/rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_uk.nc" 

    rain_data = Generator_rain(fn=fn_rain, all_at_once=False)
    
    ds_tar = tf.data.Dataset.from_generator( rain_data, output_types=( tf.float32, tf.bool) ) #(values, mask) 
    ds_tar = ds_tar.skip(start_idx_tar) #skipping to correct point

    def rain_mask(arr_rain, arr_mask, fill_value):
        """ 
            :param tnsr arr_rain:
            :param tnsr arr_mask: true if value is not valid, false if value is valid
            :param tnsr fill_value: 

        """
        arr_rain = tf.where( arr_mask, fill_value ,arr_rain )

        return arr_rain, arr_mask
    
    ds_tar = ds_tar.map( lambda _vals, _mask: rain_mask( _vals, _mask, t_params['mask_fill_value']['rain'] ) ) # (values, mask)

    def normalize_rain_values(arr_rain, arr_mask, scale):

        arr_rain = arr_rain/scale
        utility.standardize_ati(arr_rain, scale, reverse=False  )

        return arr_rain, arr_mask
    
    ds_tar = ds_tar.map( lambda _vals, _mask: normalize_rain_values( _vals, _mask, t_params['normalization_scales']['rain'] ) ) # (values, mask)
    
    ds_tar = ds_tar.window(size =t_params['lookback_target'], stride=1, shift=t_params['window_shift'] , drop_remainder=True )
    
    ds_tar = ds_tar.flat_map( lambda *window: tf.data.Dataset.zip( tuple([ w.batch(t_params['lookback_target']) for w in window ] ) ) ) #shape (lookback,h, w)
    # endregion

    # region prepare feature model fields

    fn_mf = data_dir + "/ana_input_1.nc"
    mf_data = Generator_mf(fn=fn_mf, all_at_once=False)
    
    ds_feat = tf.data.Dataset.from_generator( mf_data , output_types=( tf.float32, tf.bool)) #(values, mask) 
    ds_feat = ds_feat.skip(start_idx_feat)
    def mf_normalization_mask(arr_data, scales, arr_mask, fill_value):
        """

            :param tnsr arr_data: #shape (lb, h, w, n)
            :param tnsr scales: #shape ( n )

        """
        #TODO: Find the appropriate values for scales, by finding the reasonable max/min for each weather datum 
        arr_data = tf.divide( arr_data, scales)

        arr_data = tf.where( arr_mask, arr_data, fill_value )
        return arr_data # (h,w,c)

    ds_feat = ds_feat.map( lambda arr_data, arr_mask: mf_normalization_mask( arr_data, t_params['normalization_scales']['model_fields'],
                arr_mask ,t_params['mask_fill_value']['model_field'] ) ,num_parallel_calls= _num_parallel_calls) #_num_parallel_calls  )

    ds_feat = ds_feat.window(size = t_params['lookback_feature'], stride=1, shift=t_params['lookback_feature'], drop_remainder=True )

    ds_feat = ds_feat.flat_map( lambda window: tf.data.Dataset.zip( window.batch(t_params['lookback_feature']) ) )  #shape (lookback,h, w, 6)
    # endregion
    
    ds = tf.data.Dataset.zip( (ds_feat, ds_tar) ) #( model_fields, (rain, rain_mask) ) 
    ds = ds.batch(t_params['batch_size'], drop_remainder=True)

    # region mode of data
    model_settings = m_params['model_type_settings']
    if(model_settings['location']=="wholegrid"):
        pass

    elif(model_settings['location'] in rain_data.city_location.keys() ):
        idxs = rain_data.find_idx_of_city( model_settings['location'] )
        ds = ds.map( lambda mf, rain_rmask : load_data_ati_select_location(mf, rain_rmask[0], rain_rmask[1], idxs ), num_parallel_calls=_num_parallel_calls )

    elif(model_settings['location']=="regiongrid"):
        ds = ds.map( lambda mf, rain_rmask : load_data_ati_folded_regions(mf, rain_rmask[0], rain_rmask[1] ) )

    else:
        raise ValueError

    #ds = ds.prefetch(1)
    return ds

def load_data_ati_select_location(mf, rain, rain_mask, idxs ):
    """
        feature shape (bs, lookback, h, w, c)
        target shape (bs, lookback, h, w, c)
    """
    
    mf = mf[:, :, idxs[0], idxs[1],:]
    rain = rain[:, :, idxs[0], idxs[1]]
    rain_mask = rain_mask[ :, :, idxs[0], idxs[1] ]

    return mf, (rain, rain_mask)  #(bs, lookback, c)

def load_data_ati_folded_regions(feat, rain, rain_mask):
    raise NotImplementedError
