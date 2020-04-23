import itertools
from functools import partial
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
            "Cardiff": [51.4816 + 0.15, -3.1791 -0.05], #1st Rainiest
            "Glasgow": [55.8642,  -4.2518], #3rd rainiest
            "Lancaster":[54.466, -2.8007], #2nd hieghest
            "Bradford": [53.7960, -1.7594], #3rd highest
            "Manchester":[53.4808, -2.2426], #15th rainiest
            "Birmingham":[52.4862, -1.8904], #25th
            "Liverpool":[53.4084 , -2.9916 +0.1 ], #18th rainiest
            "Leeds":[ 53.8008, -1.5491 ], #8th
            "Edinburgh": [55.9533, -3.1883],
            "Belfast": [54.5973, -5.9301], #25
            "Dublin": [53.3498, -6.2603],
            "LakeDistrict":[54.4500,-3.100],
            
            }
        self.latitude_array = np.linspace(58.95,49.05, 100)
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

    def find_idx_of_city_region(self, city, region_grid_params):
        """
            returns the enclosing indexes for a city, this represents the region for the city
            note: this region will be a size matching the full size used during training, and not just the size used when evaluating the loss
        """

        city_idxs = self.find_idx_of_city(city) #[h,w]

        #region Handling cases of outer_box_dims being odd or even
        if( region_grid_params['outer_box_dims'][0]%2 == 0 ):
            h_up_span = region_grid_params['outer_box_dims'][0]//2 
            h_down_span = region_grid_params['outer_box_dims'][0]//2 

        else:
            h_up_span = region_grid_params['outer_box_dims'][0]//2 -1
            h_down_span = region_grid_params['outer_box_dims'][0]//2

        if( region_grid_params['outer_box_dims'][0]%2 == 0 ):
            w_left_span = region_grid_params['outer_box_dims'][0]//2 
            w_right_span = region_grid_params['outer_box_dims'][0]//2 

        else:
            w_left_span = region_grid_params['outer_box_dims'][0]//2 -1
            w_right_span = region_grid_params['outer_box_dims'][0]//2
        #endregion

        #region Handling cases of bonding regions being beyond box size
        #TODO: Finish implementing handling of cases which are on the edge
        # h_up_overstep = - min( city_idxs[0]-h_up_span , 0  ) 
        # h_down_overstep = max( city_idxs[0]+h_down_span -input_image_shape[0], 0 )

        # w_left_overstep = - min( city_idxs[1]-w_left_span, 0 )
        # w_right_overstep = max( city_idxs[1]+h_down_span-input_image_shape[1], 0 )

        #endregion

        upper_h = city_idxs[0] - h_up_span
        lower_h = city_idxs[0] + h_down_span

        left_w = city_idxs[1] - w_left_span
        right_w = city_idxs[1] + w_right_span
        
        return ( [upper_h, lower_h], [left_w, right_w] )


    def find_nearest_latitude_longitude(self, coordinates):

        latitude_index =    np.abs(self.latitude_array - coordinates[0] ).argmin()
        longitude_index =   np.abs(self.longitude_array - coordinates[1]).argmin()

        return (latitude_index, longitude_index)
    
    def find_idx_of_city_in_folded_regions( self, city, region_grid_params ):
        """
            Given the full map has been folded into the 16 by 16 regions. This code finds the index of the region which contains the city in the dimension of the folded regions
        """
        slides = region_grid_params['slides_v_h']
        city_idx = self.find_idx_of_city(city) #[a,b]

        _ = np.array(region_grid_params['outer_box_dims'])-np.array(region_grid_params['inner_box_dims'])//2 

        idx_horiz = ( city_idx[1] - _[1]  ) // region_grid_params['horizontal_shift']
        idx_vert = (city_idx[0] - _[0]) // region_grid_params['vertical_shift']
        idx_region_in_whole_flat = idx_horiz + region_grid_params['slides_v_h'][1]*idx_vert
        
        periodicy = int( np.prod(slides) )

        idx_horiz_city_in_region = ( city_idx[1] - _[1]  ) % region_grid_params['horizontal_shift']
        idx_vert_city_in_region = (city_idx[0] - _[0]) // region_grid_params['vertical_shift']
        idx_city_in_region = [idx_vert_city_in_region,idx_horiz_city_in_region  ]
        
        return idx_region_in_whole_flat, periodicy, idx_city_in_region

class Generator_rain(Generator):
    def __init__(self, **generator_params ):
        super(Generator_rain, self).__init__(**generator_params)
        self.data_len = 14822

    def yield_all(self,):
        with Dataset(self.fn, "r", format="NETCDF4",keepweakref=True) as f:
            _data = f.variables['rr'][:]
            yield np.ma.getdata(_data), np.ma.getmask(_data)   
            
    def yield_iter(self):
        f = Dataset(self.fn, "r", format="NETCDF4", keepweakref=True)
        #with Dataset(self.fn, "r", format="NETCDF4",keepweakref=True) as f:
        for chunk in f.variables['rr'][:]:
            data = np.ma.getdata(chunk)
            mask = np.logical_not( np.ma.getmask(chunk) )
            yield data[ ::-1 , :] , mask[::-1, :]

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

    def __init__(self, downscaled_input, **generator_params):

        super(Generator_mf, self).__init__(**generator_params)

        self.vars_for_feature = ['unknown_local_param_137_128', 'unknown_local_param_133_128', 'air_temperature', 'geopotential', 'x_wind', 'y_wind' ]
        self.channel_count = len(self.vars_for_feature)
        self.data_len = 16072
        self.di = downscaled_input
        
    def yield_all(self):
        raise NotImplementedError
    
    def yield_iter(self):

        f = Dataset(self.fn, "r", format="NETCDF4")
        #with Dataset(self.fn, "r", format="NETCDF4",keepweakref=True) as f:
        for tuple_mfs in zip( *[f.variables[var_name][:] for var_name in self.vars_for_feature] ):
            list_datamask = [(np.ma.getdata(_mar), np.ma.getmask(_mar) ) for _mar in tuple_mfs]
            _data, _masks = list(zip(*list_datamask))
            _masks = [ np.full(_data[0].shape, np.logical_not(_mask_val) , dtype=bool) for _mask_val in _masks ] #projecting masks from the (6,) shape to a square array shape

            stacked_data = np.stack(_data, axis=-1)
            stacked_masks = np.stack(_masks, axis=-1)
            
            if self.di == False:
                yield stacked_data[ 1:-2, 2:-2, :], stacked_masks[ 1:-2 , 2:-2, :] #(h,w,6) #(h,w,6)  #this aligns it to rain 
            elif self.di == True:
                yield stacked_data[ :-2, 2:-1, :], stacked_masks[ :-2 , 2:-1, :] #(h,w,6) #(h,w,6)  #this aligns it to rain 
            #yield stacked_data[ 1:-2, 2:-1, :], stacked_masks[ 1:-2 , 2:-1, :] #(h,w,6) #(h,w,6)  #this aligns it to rain 
    
    def __call__(self):
        return self.yield_iter()

def load_data_ati(t_params, m_params, target_datums_to_skip=None, day_to_start_at=None, _num_parallel_calls=-1,
                    data_dir="./Data/Rain_Data_Nov19" ):
    """
        This is specific for the following two files -> ana_input.nc and rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_uk.nc
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
    rain_data = Generator_rain(fn=fn_rain, all_at_once=False)
    
    ds_tar = tf.data.Dataset.from_generator( rain_data, output_types=( tf.float32, tf.bool) ) #(values, mask) 
    ds_tar = ds_tar.skip(start_idx_tar) #skipping to correct point
        
    ds_tar = ds_tar.window(size=t_params['lookback_target'], stride=1, shift=t_params['window_shift'] , drop_remainder=True )
    
    ds_tar = ds_tar.flat_map( lambda *window: tf.data.Dataset.zip( tuple([ w.batch(t_params['lookback_target']) for w in window ] ) ) ) #shape (lookback,h, w)

    def rain_mask(arr_rain, arr_mask, fill_value):
        """ 
            :param tnsr arr_rain:
            :param tnsr arr_mask: true if value is not valid, false if value is valid
            :param tnsr fill_value: 

        """
        arr_rain = tf.where( arr_mask, arr_rain, fill_value )

        return arr_rain, arr_mask
    
    ds_tar = ds_tar.map( lambda _vals, _mask: rain_mask( _vals, _mask, t_params['mask_fill_value']['rain'] ), num_parallel_calls=_num_parallel_calls ) # (values, mask)
    # endregion

    # region prepare feature model fields
    fn_mf = data_dir + "/ana_input.nc"
    mf_data = Generator_mf(fn=fn_mf, all_at_once=False, downscaled_input=t_params['downscaled_input'])
    
    ds_feat = tf.data.Dataset.from_generator( mf_data , output_types=( tf.float32, tf.bool)) #(values, mask) 
    ds_feat = ds_feat.skip(start_idx_feat)

    ds_feat = ds_feat.window(size = t_params['lookback_feature'], stride=1, shift=t_params['lookback_feature'], drop_remainder=True )

    ds_feat = ds_feat.flat_map( lambda *window: tf.data.Dataset.zip( tuple([w.batch(t_params['lookback_feature']) for w in window ] ) ) )  #shape (lookback,h, w, 6)
    
    def mf_normalization_mask(arr_data, scales, shift, arr_mask, fill_value):
        """

            :param tnsr arr_data: #shape (lb, h, w, n)
            :param tnsr scales: #shape ( n )

        """
        #TODO: Find the appropriate values for scales, by finding the reasonable max/min for each weather datum 
        arr_data = tf.subtract( arr_data, shift)  #shift
        arr_data = tf.divide( arr_data, scales) #divide

        arr_data = tf.where( arr_mask, arr_data, fill_value )
        return arr_data # (h,w,c)

    ds_feat = ds_feat.map( lambda arr_data, arr_mask: mf_normalization_mask( arr_data, t_params['normalization_scales']['model_fields'],
                t_params['normalization_shift']['model_fields'],arr_mask ,t_params['mask_fill_value']['model_field'] ) ,
                num_parallel_calls= _num_parallel_calls) #_num_parallel_calls  )
    # endregion

    ds = tf.data.Dataset.zip( (ds_feat, ds_tar) ) #( model_fields, (rain, rain_mask) ) 
    
    # region mode of data
    model_settings = m_params['model_type_settings']

    if(model_settings['location']=="wholeregion"):
        ds = ds.batch(t_params['batch_size'], drop_remainder=True)
        ds = ds.map( lambda mf, rain_rmask: tuple( [mf, rain_rmask[0], rain_rmask[1] ] ) , num_parallel_calls=_num_parallel_calls )  #unbatching final two elements

        if 'location_test' in model_settings.keys():
        
            if m_params['model_name'] in ["SimpleLSTM","SimpleDense","SimpleGRU"]:
                raise NotImplementedError

            elif m_params['model_name'] in ["SimpleConvGRU",'THST']:
                idxs_of_city = rain_data.find_idx_of_city( model_settings['location_test'] )

                

            return ds, idxs_of_city

        return ds        
    
    elif( type(model_settings['location'][:]) == list ):

        if m_params['model_name'] in ["SimpleLSTM","SimpleDense","SimpleGRU"]:
            
            if 'location_test' in model_settings.keys():
                ds = ds.batch(t_params['batch_size'], drop_remainder=True)
                idxs = rain_data.find_idx_of_city( model_settings['location_test'] )
                ds = ds.map( lambda mf, rain_rmask : load_data_ati_select_location(mf, rain_rmask[0], rain_rmask[1], idxs ), num_parallel_calls=_num_parallel_calls )
                ds = ds.unbatch().batch( t_params['batch_size'],drop_remainder=True )

            elif 'location_test' not in model_settings.keys():
                ds = ds.batch(t_params['batch_size'], drop_remainder=True)
                li_idxs = [ rain_data.find_idx_of_city( _city ) for _city in model_settings['location'] ]

                li_ds = [ ds.map( lambda mf, rain_rmask : load_data_ati_select_location(mf, rain_rmask[0], rain_rmask[1], idx ), num_parallel_calls=_num_parallel_calls ) for idx in li_idxs ]
                                
                for idx in range(len(li_ds ) ):
                    li_ds[idx] = li_ds[idx].unbatch().batch( t_params['batch_size'],drop_remainder=True )
                    if idx==0:
                        ds = li_ds[0]
                    else:
                        ds = ds.concatenate( li_ds[idx] )
            
        elif m_params['model_name'] in ["SimpleConvGRU","SimpleConvLSTM",'THST']:

            if 'location_test' in model_settings.keys():
                idx_city_in_region = [8,8] #for this setting the city will always be the middle icon

                ds = ds.map( lambda mf, rain_mask: tuple( [mf, rain_mask[0], rain_mask[1]] )  )                         #mf=(80, 100,140,6)                        
                h_idxs, w_idxs = rain_data.find_idx_of_city_region( model_settings['location_test'], m_params['region_grid_params'] )
                ds = ds.map( lambda mf, rain, rmask : load_data_ati_select_region_from_nonstack( mf, rain, rmask, h_idxs, w_idxs) , num_parallel_calls=_num_parallel_calls )
                
                ds = ds.unbatch().batch( t_params['batch_size'],drop_remainder=True )
                ds = ds.prefetch(_num_parallel_calls)
                return ds, idx_city_in_region
            
            elif 'location_test' not in model_settings.keys():

                ds = ds.map( lambda mf, rain_mask: tuple( [mf, rain_mask[0], rain_mask[1]] )  )   
                li_hw_idxs = [ rain_data.find_idx_of_city_region( _location, m_params['region_grid_params'] ) for _location in model_settings['location'] ] #[ (h_idx,w_idx) ]
                li_ds = [ ds.map( lambda mf, rain, rmask : load_data_ati_select_region_from_nonstack( mf, rain, rmask, _idx[0], _idx[1]) , num_parallel_calls=_num_parallel_calls ) for _idx in li_hw_idxs ]
                
                for idx in range(len(li_ds ) ):
                    li_ds[idx] = li_ds[idx].unbatch().batch( t_params['batch_size'],drop_remainder=True )
                    if idx==0:
                        ds = li_ds[0]
                    else:
                        ds = ds.concatenate( li_ds[idx] )
                
    elif( model_settings['location'] in rain_data.city_location.keys()  ):
        
        if m_params['model_name'] in ["SimpleLSTM","SimpleDense","SimpleGRU"]:
            ds = ds.batch(t_params['batch_size'], drop_remainder=True)
            idxs = rain_data.find_idx_of_city( model_settings['location'] )
            ds = ds.map( lambda mf, rain_rmask : load_data_ati_select_location(mf, rain_rmask[0], rain_rmask[1], idxs ), num_parallel_calls=_num_parallel_calls )
        
        elif m_params['model_name'] in ["SimpleConvGRU",'THST']:
            ds = ds.map( lambda mf, rain_mask: tuple( [mf, rain_mask[0], rain_mask[1]] )  )                         #mf=(80, 100,140,6)
            
            # region_folding_partial = partial(region_folding, **m_params['region_grid_params'], mode=4,tnsrs=None )  
            # ds = ds.map( lambda mf,rain,rmask : tf.py_function( region_folding_partial, [mf, rain, rmask], [tf.float16, tf.float32, tf.bool] ) ) #mf = mode3(704, lookback, 16, 16, 6 )  mode1(80, 85//4, 125//4, 16, 16, 6)
            h_idxs, w_idxs = rain_data.find_idx_of_city_region( model_settings['location'], m_params['region_grid_params'] )

            
            ds = ds.map( lambda mf, rain, rmask : load_data_ati_select_region_from_nonstack( mf, rain, rmask, h_idxs, w_idxs) , num_parallel_calls=_num_parallel_calls )

            if 'location_test' in model_settings.keys():
                idx_city_in_region = [8,8] #for this setting the city will always be the middle icon
                #ds = ds.map( lambda mf, rain, rmask: load_data_ati_select_region_from_stack(mf, rain, rmask, idx_region_flat, periodicy ), num_parallel_calls = _num_parallel_calls  )
                ds = ds.unbatch().batch( t_params['batch_size'],drop_remainder=True )
                ds = ds.prefetch(_num_parallel_calls)
                return ds, idx_city_in_region
            
        ds = ds.unbatch().batch( t_params['batch_size'],drop_remainder=True )

    elif(model_settings['location']=="region_grid"):
        
        ds = ds.map( lambda mf, rain_mask: tuple( [mf, rain_mask[0], rain_mask[1]] )  )                         #mf=(80, 100,140,6)
        
        region_folding_partial = partial(region_folding, **m_params['region_grid_params'], mode=3,tnsrs=None )  

        ds = ds.map( lambda mf,rain,rmask : tf.py_function( region_folding_partial, [mf, rain, rmask], [tf.float16, tf.float32, tf.bool] ) ) #mf = mode3(704, lookback, 16, 16, 6 )  mode1(80, 85//4, 125//4, 16, 16, 6)
        
        #NOTE: the below method of finding the right image in the stacked list can be replaced by the method for location_test above
        if 'location_test' in model_settings.keys():
            idx_region_flat, periodicy, idx_city_in_region = rain_data.find_idx_of_city_in_folded_regions( model_settings['location_test'],m_params['region_grid_params'] )
            ds = ds.map( lambda mf, rain, rmask: load_data_ati_select_region_from_stack(mf, rain, rmask, idx_region_flat, periodicy ), num_parallel_calls = _num_parallel_calls  )
            ds = ds.unbatch().batch( t_params['batch_size'],drop_remainder=True )
            ds = ds.prefetch(_num_parallel_calls)
            return ds, idx_city_in_region


        ds = ds.unbatch().batch( t_params['batch_size'],drop_remainder=True )

    else:
        raise ValueError
    #endregion
    
    #if len( tf.config.list_physical_devices('GPU') ) >0:
    ds = ds.prefetch(_num_parallel_calls)
    return ds

def load_data_ati_select_location(mf, rain, rain_mask, idxs ):
    """
        feature shape (bs, lookback, h, w, c)
        target shape (bs, lookback, h, w, c)
    """
    
    mf = mf[:, :, idxs[0], idxs[1],:] #(bs, seq_len1, c)
    rain = rain[:, :, idxs[0], idxs[1]] #(bs, seq_len1, h, w)
    rain_mask = rain_mask[ :, :, idxs[0], idxs[1] ]

    return mf, (rain, rain_mask)  #(bs, lookback, c)

#Make all the below into one class
def region_folding( mf, rain, rain_mask, tnsrs=None ,mode=1, 
                    outer_box_dims=[16,16], inner_box_dims=[4,4],
                    vertical_shift=4, horizontal_shift=4,
                    input_image_shape=[100,140], slides_v_h=[25,35] ):
    #TODO: convert to class and add shape checks
        #shape-check outer_box_dims must be even and a multiple of inner_box_dims
    #NOTE: this only works given square inner_box and outer_box
    as_strided = np.lib.stride_tricks.as_strided
    """ 
        Use w/ partial functions to switch between method 1 and 2
        Mode 1: Unstitch
            :param list mf, rain, rain_mask tnsrs: #[(bs, h, w), (bs, h, w), (bs, h, w) ] or #[(bs, h, w, c), (bs, h, w, c), (bs, h, w, c ) ] note bs may be lookback
                This will be a list of tensors/arrays to unstitch into multiple tensors
            returns a list of the batch size tensors all unstitched  #[(bs,h_slices,v_slices ,h1, w1), (bs, h_slices,v_slices,h1, w), (bs,h_slices,v_slices, h, w) ]

        Mode 2: Stitch Up
            :param list tnsrs: A list of tnsrs to stitch or unstitch. 
            
        
            return: list of

        #It is assumed we are operating on batches
    """
    if mode==1: #Unstitch
        if tf.is_tensor(mf):
            mf = mf.numpy().astype(dtype=np.float16)
        if tf.is_tensor(rain):
            rain = rain.numpy()#.astype(dtype=np.float16)
        if tf.is_tensor(rain_mask):
            rain_mask = rain_mask.numpy()
        

        d0 = list(mf.shape[:1])
        d1_2 = ( ( np.array(input_image_shape)-np.array(outer_box_dims)+np.array([1,1]) ) // np.array(inner_box_dims) ).tolist()
        d3_4 = outer_box_dims
        d_5 = list(mf.shape[-1:])
        shape_mf = d0 + d1_2 + d3_4 + d_5 #[10, 85//4, 31, 16, 16, c]
        
        s0 = mf.strides[-1]
        s1 = mf.strides[-2]
        s2 = mf.strides[-3]
        s3 = s1*horizontal_shift
        s4 = s2*vertical_shift
        s5 = s2*mf.shape[2]
        strides_mf = [s5, s4, s3, s2, s1, s0]
        mf_unstitched = as_strided( mf, shape_mf, strides_mf, writeable=False).copy(order='K')  #TODO: may have to add copy, but as of now it may lead to memory issues
        
        d0   = list(rain.shape[:1])
        d1_2 = ( ( np.array(input_image_shape)-np.array(outer_box_dims)+np.array([1,1]) ) // np.array(inner_box_dims) ).tolist()
        d3_4 = outer_box_dims
        shape_r = d0 + d1_2 + d3_4         

        s0 = rain.strides[-1]
        s1 = rain.strides[-2]
        s2 = s0*horizontal_shift
        s3 = s1*vertical_shift
        s4 = s1*rain.shape[1]
        strides_r = [ s4, s3, s2, s1, s0 ]
        
        rain_unstitched = as_strided( rain, shape_r, strides_r,writeable=False ).copy(order='K')
        rain_mask_unstitched = as_strided( rain_mask, shape_r, strides_r,writeable=False ).copy(order='K')
        
        return np.ones_like(mf_unstitched,dtype=np.float16 ), np.ones_like(rain_unstitched,dtype=np.float32 ), np.ones_like(rain_mask_unstitched,dtype=bool)
        #return mf_unstitched, rain_unstitched, rain_mask_unstitched 
    elif mode==2: 
        #NOTE: mode 2 does not accept arrs w/ channel dim
        
        if tf.is_tensor(tnsrs):
            #_arrays = [tnsr.numpy() for tnsr in tnsrs]
            _array = tnsrs.numpy()

        #otherwise its already in list form
        #_array = np.stack( _arrays, axis=0 ) #(region_count, bs, h, w)

        #cropping
        central_indx_crop_start = outer_box_dims[0]//2 - inner_box_dims[0]//2
        cics = central_indx_crop_start
        cice = cics + inner_box_dims[0]
        _array = _array[:, :, :, cics:cice, cics:cice]  #(bs,vertical_slices,horizontal_slices,h1, w1)

        #transposing
        _array = _array.transpose(0,1,3,2,4) #(bs,vertical_slices,horizontal_slices,h1, w1)
        #reshape
        bs = _array.shape[0]
        h = np.prod(_array.shape[1:3])
        w = np.prod(_array.shape[3:5])
        stitched_array = _array.reshape([bs,h,w])

        return stitched_array
    elif mode==3:#Unstitch
        if tf.is_tensor(mf):
            mf = mf.numpy().astype(dtype=np.float16)
        if tf.is_tensor(rain):
            rain = rain.numpy().astype(dtype=np.float16)
        if tf.is_tensor(rain_mask):
            rain_mask = rain_mask.numpy()
        _li_data1 = []
        _li_data2 = []
        _li_data3 = []
        for idx_h in range(0,input_image_shape[0]-outer_box_dims[0]+1, vertical_shift):
            for idx_w  in range(0, input_image_shape[1]-outer_box_dims[1]+1, horizontal_shift):
                _li_data1.append( mf[:, idx_h:idx_h+outer_box_dims[0], idx_w:idx_w+outer_box_dims[1], : ] )
                _li_data2.append( rain[:, idx_h:idx_h+outer_box_dims[0], idx_w:idx_w+outer_box_dims[1] ] )
                _li_data3.append( rain_mask[ :, idx_h:idx_h+outer_box_dims[0], idx_w:idx_w+outer_box_dims[1]] )
        mf = np.stack(_li_data1, axis=0) #( h_strides*w_strides, seq_len ,h1, w1, c)
        rain = np.stack(_li_data2, axis=0) #( h_strides*w_strides, seq_len ,h1, w1)
        rain_mask = np.stack(_li_data3,axis=0)
        return mf, rain, rain_mask
    else:
        raise ValueError

def load_data_ati_select_region_from_stack( mf, rain, rain_mask, idx_region_flat, periodicy):
    """
        Note: This only works following mode==3 of region folding
        
        periodicy represents how often the stacked regions represent that same regions in the first dimensions
    """
    mf = mf[ idx_region_flat::periodicy ,: , :, :, :] #( h_strides1*w_strides1, seq_len ,h1, w1, c )
    rain = rain[idx_region_flat::periodicy, :, :, :] #( h_strides1*w_strides1, seq_len ,h1, w1 )
    rain_mask = rain_mask[idx_region_flat::periodicy, :, :, :]

    return mf, rain, rain_mask

def load_data_ati_select_region_from_nonstack(mf, rain, rain_mask, h_idxs, w_idxs ):
    """
        idx_h,idx_w: refer to the top left right index for the square region of interest this includes the region which is removed after cropping to calculate the loss during train step
    """
    mf = mf[ :, h_idxs[0]:h_idxs[1] , w_idxs[0]:w_idxs[1] , : ]
    rain = rain[ :, h_idxs[0]:h_idxs[1] , w_idxs[0]:w_idxs[1] ]
    rain_mask = rain_mask[ :, h_idxs[0]:h_idxs[1] , w_idxs[0]:w_idxs[1] ]

    return tf.expand_dims(mf,axis=0), tf.expand_dims(rain,axis=0), tf.expand_dims(rain_mask,axis=0) #Note: expand_dim for unbatch/batch compatibility

def load_data_ati_post_region_folding(mf, rain, mask):
    """
        input shape (lookback, h_slices, v_slices, h1, w1)
        return shape (region_count,lookback,h1,w1)
    """
    mfs = mf.shape
    rs = rain.shape
    mf = tf.reshape( mf,[ mfs[-5],mfs[-4]*mfs[-3],mfs[-2],mfs[-1]]).transpose(1,0,2,3,4)
    rain = tf.reshape( rain,[rs[-5],rs[-4]*rs[-3],rs[-2],rs[-1]]).transpose(1,0,2,3)
    mask = tf.reshape( mask,[rs[-5],rs[-4]*rs[-3],rs[-2],rs[-1]]).transpose(1,0,2,3) #NOTE: not sure if this is correct

    return mf, (rain, mask)
#endregion