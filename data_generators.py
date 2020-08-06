from netCDF4 import Dataset, num2date
import glob
import itertools as it
import json
import os
import pickle

import numpy as np
import tensorflow as tf

import utility

import xarray as xr


"""
    Example of how to use
    import Generators

    rr_ens file 
    _filename = "Data/Rain_Data/rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_djf_uk.nc"
    rain_gen = Generator_rain(_filename, all_at_once=True)
    datum = next(iter(grib_gen))

"""
# region -- Era5_Eobs
class Generator():
    """
        Base class for Generator classes
        Example of how to use:
            fn = "Data/Rain_Data/rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_djf_uk.nc"
            rain_gen = Generator_rain(fn, all_at_once=True)
            datum = next(iter(grib_gen))
    """
    
    def __init__(self, fp, all_at_once=False):
        """Extendable Class handling the generation of model field and rain data
            from E-Obs and ERA5 datasets

        Args:
            fp (str): Filepath of netCDF4 file containing data.
            all_at_once (bool, optional): Whether or not to load all the data in RAM or not. Defaults to False.
            
        """        
        self.generator = None
        self.all_at_once = all_at_once
        self.fp = fp
        self.city_latlon = {
            "London": [51.5074, -0.1278],
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
            "Newry":[54.1751, -6.3402],
            "Preston":[53.7632, -2.7031 ],
            "Truro":[50.2632, -5.0510],
            "Bangor":[54.2274 - 0, -4.1293 - 0.3],
            "Plymouth":[50.3755 + 0.1, -4.1427],
            "Norwich": [52.6309, 1.2974]
            }
        
        #The longitude lattitude grid for the 0.1 degree E-obs and rainfall data
        self.latitude_array = np.linspace(58.95, 49.05, 100)
        self.longitude_array = np.linspace(-10.95, 2.95, 140)
        
        # Retrieving information on temporal length of  dataset        
        with Dataset(self.fp, "r+", format="NETCDF4") as ds:
            self.data_len = ds.dimensions['time'].size
        #self.data_len = Dataset(self.fp, "r+", format="NETCDF4").dimensions['time'].size
        
    def yield_all(self):
        pass

    def yield_iter(self):
        pass
    
    def __call__(self):
        if(self.all_at_once):
            return self.yield_all()
        else:
            return self.yield_iter()
    
    def find_idxs_of_loc(self, loc="London"):
        """Returns the grid indexes on the 2D map of the UK which correspond to the location (loc) point

        Args:
            loc (str, optional): name of the location. Defaults to "London".

        Returns:
            tuple: Contains indexes (h1,w1) for the location (loc)
        """        
        coordinates = self.city_latlon[loc]
        indexes = self.find_nearest_latitude_longitude( coordinates)  # (1,1)
        return indexes

    def find_idx_of_loc_region(self, loc, region_grid_params):
        """ Returns the the indexes defining gridded box that surrounds the location of interests

            Raises:
                ValueError: [If the location of interest is too close to the border for evaluation]

            Returns:
                tuple: Returns a tuple ( [upper_h, lower_h], [left_w, right_w] ), defining the grid box that 
                    surrounds the location (loc)
        """
        
        city_idxs = self.find_idxs_of_loc(loc) #[h,w]
        
        # Checking that central region of interest is not too close to the border
        bool_regioncheck1 = np.array(city_idxs) <  np.array(region_grid_params['outer_box_dims']) - np.array(region_grid_params['inner_box_dims'])
        bool_regioncheck2 = np.array(region_grid_params['input_image_shape']) - np.array(city_idxs) < np.array(region_grid_params['inner_box_dims'])
        if bool_regioncheck1.any() or bool_regioncheck2.any(): raise ValueError("The specified region is too close to the border")

        # Defining the span, in all directions, from the central region
        if( region_grid_params['outer_box_dims'][0]%2 == 0 ):
            h_up_span = region_grid_params['outer_box_dims'][0]//2 
            h_down_span = h_up_span
        else:
            h_up_span = region_grid_params['outer_box_dims'][0]//2
            h_down_span = region_grid_params['outer_box_dims'][0]//2 + 1

        if( region_grid_params['outer_box_dims'][1]%2 == 0 ):
            w_left_span = region_grid_params['outer_box_dims'][1]//2 
            w_right_span = w_left_span
        else:
            w_left_span = region_grid_params['outer_box_dims'][1]//2
            w_right_span = region_grid_params['outer_box_dims'][1]//2 + 1
        
        #Defining outer_boundaries
        upper_h = city_idxs[0] - h_up_span
        lower_h = city_idxs[0] + h_down_span

        left_w = city_idxs[1] - w_left_span
        right_w = city_idxs[1] + w_right_span
        
        return ( [upper_h, lower_h], [left_w, right_w] )

    def find_nearest_latitude_longitude(self, lat_lon):
        """Given specific lat_lon, this method finds the closest long/lat points on the
            0.1degree grid our input/target data is defined on

        Args:
            lat_lon (tuple): tuple containing the lat and lon values of interest

        Returns:
            tuple: tuple containing the idx_h and idx_w values that detail the posiiton on lat_lon on the 
                0.1degree grid on which the ERA5 and E-Obvs data is defined
        """        
        latitude_index =    np.abs(self.latitude_array - lat_lon[0] ).argmin()
        longitude_index =   np.abs(self.longitude_array - lat_lon[1]).argmin()

        return (latitude_index, longitude_index)
    
    def get_locs_for_whole_map(self, region_grid_params):
        """This function returns a list of boundaries which can be used to extract all patches
            from the 2D map

            Args:
                region_grid_params (dictionary): a dictioary containing information on the sizes of 
                    patches to be extract from the main image

            Returns:
                list : return a list of of tuples defining the boundaries of the region
                        of the form [ ([upper_h, lower_h]. [left_w, right_w]), ... ]
        """       
        input_image_shape = region_grid_params['input_image_shape']
        h_shift = region_grid_params['vertical_shift']
        w_shift = region_grid_params['horizontal_shift']
        h_span, w_span = region_grid_params['outer_box_dims']

        #list of values for upper_h and lower_h
        range_h = np.arange(0, input_image_shape[0]-h_span+1, step=h_shift, dtype=np.int32 ) 
        # list of pairs of values (upper_h, lower_h)
        li_range_h_pairs = [ [range_h[i], range_h[i]+h_span ] for i in range(0,len(range_h))]
        
        #list of values for left_w and right_w
        range_w = np.arange(0, input_image_shape[1]-w_span+1, step=w_shift, dtype=np.int32)
        # list of pairs of values (left_w, right_w)
        li_range_w_pairs = [ [range_w[i], range_w[i]+w_span ] for i in range(0,len(range_w))]

        li_boundaries = list( it.product( li_range_h_pairs, li_range_w_pairs ) )

        boundaries_to_remove = [ ([0,16],[0,16]), ([0,16],[4,20]), ([0,16],[8,24]), ([0,16],[12,28]), ([0,16],[16,32]), ([0,16],[20,36]), ([0,16],[24,40]), ([0,16],[28,44]),  ([0,16],[32,48]),                           ([0,16],[64,80]),([0,16],[68,84]),([0,16],[72,88]), ([0,16],[76,92]), ([0,16],[80,96]), ([0,16],[84,100]), ([0,16],[88,104]),([0,16],[92,108]),([0,16],[96,112]), ([0,16],[100,116]),([0,16],[104,120]),([0,16],[108,124]), ([0,16],[112,128]), ([0,16],[116,132]), ([0,16],[120,136]), ([0,16],[124,140]),
                                ([4,20],[0,16]), ([4,20],[4,20]), ([4,20],[8,24]), ([4,20],[12,28]), ([4,20],[16,32]), ([4,20],[20,36]), ([4,20],[24,40]), ([4,20],[28,44]),  ([4,20],[32,48]),                             ([4,20],[76,92]), ([4,20],[80,96]), ([4,20],[84,100]),([4,20],[88,104]),([4,20],[92,108]),([4,20],[96,112]),([4,20],[100,116]), ([4,20],[104,120]),([4,20],[108,124]),([4,20],[112,128]), ([4,20],[116,132]), ([4,20],[120,136]), ([4,20],[124,140]),
                                   ([8,24],[0,16]), ([8,24],[4,20]), ([8,24],[8,24]), ([8,24],[12,28]),  ([8,24],[16,32]), ([8,24],[20,36]), ([8,24],[24,40]), ([8,24],[28,44]),  ([8,24],[32,48]),                                 ([8,24],[96,112]),([8,24],[100,116]),([8,24],[104,120]),([8,24],[108,124]),([8,24],[112,128]), ([8,24],[116,132]), ([8,24],[120,136]), ([8,24],[124,140]),
                                ([12,28],[0,16]), ([12,28],[4,20]), ([12,28],[8,24]), ([12,28],[12,28]),  ([12,28],[16,32]), ([12,28],[20,36]), ([12,28],[24,40]), ([12,28],[28,44]),  ([12,28],[32,48]),                           ([12,28],[96,112]),([12,28],[100,116]),([12,28],[104,120]),([12,28],[108,124]),([12,28],[112,128]),([12,28],[116,132]), ([12,28],[120,136]), ([12,28],[124,140]),
                                   ([16,32],[0,16]), ([16,32],[4,20]), ([16,32],[8,24]), ([16,32],[12,28]),([16,32],[16,32]), ([16,32],[20,36]), ([16,32],[24,40]), ([16,32],[28,44]),  ([16,32],[32,48]),                           ([16,32],[96,112]),([16,32],[100,116]),([16,32],[104,120]),([16,32],[108,124]),([16,32],[112,128]), ([16,32],[120,136]), ([16,32],[124,140]),
                                   ([20,36],[0,16]), ([20,36],[4,20]), ([20,36],[8,24]), ([20,36],[20,36]),([20,36],[16,32]), ([20,36],[20,36]), ([20,36],[24,40]), ([20,36],[28,44]),  ([20,36],[32,48]),                           ([20,36],[96,112]),([20,36],[100,116]),([20,36],[104,120]),([20,36],[108,124]),([20,36],[112,128]),([20,36],[116,132]), ([20,36],[124,140]),
                                   ([24,40],[0,16]), ([24,40],[4,20]), ([24,40],[8,24]), ([24,40],[12,28]),([24,40],[16,32]), ([24,40],[20,36]), ([24,40],[24,40]), ([24,40],[28,44]),  ([24,40],[32,48]),                                  ([24,40],[100,116]),([24,40],[104,120]),([24,40],[108,124]),([24,40],[112,128]),([24,40],[116,132]), ([24,40],[120,136]) , ([24,40],[124,140]),
                                                                                                                                                                                                                                            ([28,44],[100,116]),([28,44],[104,120]),([28,44],[108,124]),([28,44],[112,128]),([28,44],[116,132]), ([28,44],[120,136]), ([28,44],[124,140]),                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                ([32,48],[104,120]),([32,48],[108,124]),([32,48],[112,128]),([32,48],[116,132]), ([32,48],[120,136]),([32,48],[124,140]),                                   
                                                                                                                                                                                                                                                                                    ([36,52],[108,124]),([36,52],[112,128]),([36,52],[116,132]), ([36,52],[120,136]),([36,52],[124,140]),                                   
                                                                                                                                                                                                                                                                                                    ([40,56],[112,128]),([40,56],[116,132]),([40,56],[120,136]),([40,56],[124,140]),      
                                                                                                                                                                                                                                                                                                                                           ([44,60],[120,136]),([44,60],[124,140]),                                   
                                                                                                                                                                                                                                                                                                                                           ([48,64],[120,136]),([48,64],[124,140]),                                   
                                                                                                                                                                                                                                            ([20,36],[14,32]), 
                                                                                                                                                                                                                                            ([24,40],[14,32]), 
                                   ([80,96],[0,16]), ([80,96],[4,20]), ([80,96],[8,24]), ([80,96],[12,28]), ([80,96],[14,32]), ([80,96],[18,36]), ([80,96],[22,40]),
                                   ([84,100],[0,16]), ([84,100],[4,20]), ([84,100],[8,24]), ([84,100],[12,28]),([84,100],[14,32]), ([84,100],[18,36]), ([84,100],[22,40])  ]

        li_boundaries = [ x for x in li_boundaries if x not in boundaries_to_remove]
        return li_boundaries 

class Generator_rain(Generator):
    """ A generator for E-obs 0.1 degree rain data
    
    Returns:
        A python generator for the rain data
        
    """
    def __init__(self, **generator_params ):
        super(Generator_rain, self).__init__(**generator_params)
        
    def yield_all(self):
        """ Return all data at once
        """
        with Dataset(self.fp, "r", format="NETCDF4",keepweakref=True) as ds:
            _data = ds.variables['rr'][:]
            yield np.ma.getdata(_data), np.ma.getmask(_data)   
            
    def yield_iter(self):
        """ Return data in chunks"""
        ds = Dataset(self.fp, "r", format="NETCDF4", keepweakref=True)
        for chunk in ds.variables['rr'][:]:
            data = np.ma.getdata(chunk)
            mask = np.logical_not( np.ma.getmask(chunk) )
            yield data[ ::-1 , :], mask[::-1, :]

    def __call__(self):
        return self.yield_iter()
    
class Generator_mf(Generator):
    """Creates a generator for the model_fields_dataset
    """

    def __init__(self, vars_for_feature, seq_len=100 ,**generator_params):
        """[summary]

        Args:
            generator_params : list of params to pass to base Generator class
        """        
        super(Generator_mf, self).__init__(**generator_params)

        self.vars_for_feature = vars_for_feature #['unknown_local_param_137_128', 'unknown_local_param_133_128', 'air_temperature', 'geopotential', 'x_wind', 'y_wind' ]       
        self.seq_len = seq_len*25
        self.start_idx = 0
        #self.ds = Dataset(self.fp, "r", format="NETCDF4")


    def yield_all(self):
        raise NotImplementedError
       

    def yield_iter(self):
        xr_gn = xr.open_dataset(self.fp, cache=False, decode_times=False, decode_cf=False)
        
        idx = self.start_idx
        
        while idx < self.data_len:
        #while _bool == True:
            #idxs = arr_idxs[idx:idx+self.seq_len] 
            
            #next_marray = [ xr_gn[name].isel(time=idxs).to_masked_array() for name in self.vars_for_feature ]
            adj_seq_len = min(self.seq_len, self.data_len - idx )

            _slice = slice( idx , idx  + adj_seq_len)
            next_marray = [ xr_gn[name].isel(time=_slice).to_masked_array(copy=True) for name in self.vars_for_feature ]
            
            list_datamask = [(np.ma.getdata(_mar), np.ma.getmask(_mar)) for _mar in next_marray]
            
            _data, _masks = list(zip(*list_datamask))
            _masks = [ np.logical_not(_mask_val) for _mask_val in _masks] 
            stacked_data = np.stack(_data, axis=-1)
            stacked_masks = np.stack(_masks, axis=-1)

            idx += adj_seq_len
            #if idx>self.data_len: _bool=False

            yield stacked_data[ :, 1:-2, 2:-2, :], stacked_masks[ :, 1:-2 , 2:-2, :] #(100,140,6) 

            
    def __call__(self):
        return self.yield_iter()

class Era5_Eobs():

    """Produces Tensorflow Datasets for the ERA5 and E-obs dataset
    
    """
    def __init__(self, t_params, m_params): 
        
        self.t_params = t_params
        self.m_params = m_params

        data_dir = self.t_params['data_dir']

        # Create python generator for rain data
        fp_rain = data_dir+"/" + self.t_params.get('rain_fn',"rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_uk.nc")
        self.rain_data = Generator_rain(fp=fp_rain, all_at_once=False)

        # Create python generator for model field data 
        mf_fp = data_dir + "/" + self.t_params.get('mf_fn', "ana_input_intrp_linear.nc")
        self.mf_data = Generator_mf(fp=mf_fp, vars_for_feature=self.t_params['vars_for_feature'], all_at_once=False, seq_len=self.t_params['lookback_feature'] )

        # Update information on the locations of interest to extract data from
        self.location_size_calc()

    def location_size_calc(self, custom_location=None): 
        """ Updates list of locations to evaluate on

            Args:
                custom_location (list optional): A list of locations to evaluate on
        """        
        model_settings = self.m_params['model_type_settings']
        
        if custom_location != None:
            self.li_loc = custom_location
        else:
            self.li_loc = utility.location_getter(model_settings)

        self.loc_count = len( self.li_loc ) if \
                self.li_loc != ["All"] else \
                    len( self.rain_data.get_locs_for_whole_map(self.m_params['region_grid_params']))

    def load_data_era5eobs(self, batch_count, start_date,_num_parallel_calls=-1, prefetch=-1):
        """Produces Tensorflow Datasets for the ERA5 and E-obs dataset

            Args:
                t_params (dict): dictionary for parameters related to training/testing
                m_params (dict): dictionary for parameters related to model
                batch_count int: Number of batches to extract for evaluation
                
                _num_parallel_calls (int, optional): Number of parallel calls to use in tensorflow dataset loading operations. Defaults to -1.
                data_dir (str, optional): path of Data directory. Defaults to "./Data/Rain_Data_Nov19".

            Raises:
                NotImplementedError: [description]
                ValueError: [description]
                NotImplementedError: [description]
                ValueError: [description]

            Returns:
                tf.dataset: Dataset containing ERA5 and Eobs predictions
        """    
        
        # Retreiving one index for each of the feature and target data. This index indicates the first value in the dataset to use
        start_idx_feat, start_idx_tar = self.get_start_idx(start_date)
        self.mf_data.start_idx = start_idx_feat
        # region - Preparing feature model fields        
        ds_feat = tf.data.Dataset.from_generator( self.mf_data , output_types=(tf.float16, tf.bool), output_shapes=( tf.TensorShape([None, None, None, None]),tf.TensorShape([None, None, None, None])) ) #(values, mask) 
        ds_feat = ds_feat.unbatch()
        #ds_feat = ds_feat.skip(start_idx_feat) # Skipping forward to the correct time
        
        ds_feat = ds_feat.window(size = self.t_params['lookback_feature'], stride=1, shift=self.t_params['lookback_feature'], drop_remainder=True )
        ds_feat = ds_feat.flat_map( lambda *window: tf.data.Dataset.zip( tuple([w.batch(self.t_params['lookback_feature']) for w in window ] ) ) )  # shape (lookback,h, w, 6)
        
        ds_feat = ds_feat.map( lambda arr_data, arr_mask: self.mf_normalize_mask( arr_data, arr_mask), num_parallel_calls= _num_parallel_calls) 

        #ds_feat = ds_feat.map(lambda arr_data : tf.cast(arr_data, tf.float16), num_parallel_calls = _num_parallel_calls )
        # endregion

        # region - Preparing Eobs target_rain_data   
        ds_tar = tf.data.Dataset.from_generator( self.rain_data, output_types=(tf.float32, tf.bool), output_shapes=( tf.TensorShape([None, None]), tf.TensorShape([ None, None])) ) # (values, mask) 
        ds_tar = ds_tar.skip(start_idx_tar)     #skipping to correct point

        ds_tar = ds_tar.window(size = self.t_params['lookback_target'], stride=1, shift=self.t_params['window_shift'] , drop_remainder=True )
        ds_tar = ds_tar.flat_map( lambda *window: tf.data.Dataset.zip( tuple([ w.batch(self.t_params['lookback_target']) for w in window ] ) ) ) # shape (lookback,h, w)
        
        ds_tar = ds_tar.map( lambda _vals, _mask: self.mask_rain( _vals, _mask ), num_parallel_calls=_num_parallel_calls ) # (values, mask)
        # endregion

        # Combining datasets
        ds = tf.data.Dataset.zip( (ds_feat, ds_tar) ) #( model_fields, (rain, rain_mask) ) 
        
        ds, idx_loc_in_region = self.location_extractor( ds, self.li_loc, batch_count)
        ds = ds.prefetch(prefetch)
        
        return ds, idx_loc_in_region
        #return ds.take(batch_count), idx_loc_in_region

    def get_start_idx(self, start_date):
        """ Returns two indexes
                The first index is the idx at which to start extracting data from the feature dataset
                The second index is the idx at which to start extracting data from the target dataset
            Args:
                start_date (np.datetime64): Start date for evaluation
            Returns:
                tuple (int, int): starting index for the feature, starting index for the target data
        """        

        feature_start_date = self.t_params['feature_start_date']
        target_start_date = self.t_params['target_start_date']

        feat_days_diff = np.timedelta64(start_date - feature_start_date,'6h').astype(int)
        tar_days_diff = np.timedelta64(start_date - target_start_date, 'D').astype(int)

        feat_start_idx = feat_days_diff #since the feature comes in four hour chunks
        tar_start_idx = tar_days_diff 

        return feat_start_idx, tar_start_idx
    
    def mask_rain(self, arr_rain, arr_mask):
        """Mask rain by applying fill_value to masked points

            Args:
                arr_rain (tensor): rain_values
                arr_mask (tesor): rain_mask
                
            Returns:
                tuple: Tuple containing masked rain and the mask values
        """            
    
        arr_rain = tf.where( arr_mask, arr_rain, self.t_params['mask_fill_value']['rain'] )

        return arr_rain, arr_mask

    def mf_normalize_mask(self, arr_data, arr_mask):
        """Normalize and Mask the model field data

            Args:
                arr_data (tensor): model field data as tensor
                arr_mask (tesor): rain_mask

            Returns:
                tensor: Masked and normalized model field data
        """

        arr_data = tf.subtract( arr_data, self.t_params['normalization_shift']['model_fields'])    #shift
        arr_data = tf.divide( arr_data, self.t_params['normalization_scales']['model_fields'])       #divide
        
        arr_data = tf.where( arr_mask, arr_data, self.t_params['mask_fill_value']['model_field'])
        return arr_data #(h,w,c)

    def location_extractor(self, ds, locations, batch_count):
        """Extracts the temporal slice of patches corresponding to the locations of interest 

                Args:
                    ds (tf.Data.dataset): dataset containing temporal slices of the regions surrounding the locations of interest
                    locations (list): list of locations (strings) to extract

                Returns:
                    tuple: (tf.data.Dataset, [int, int] ) tuple containing dataset and [h,w] of indexes of the central region
        """        
        

        
        ds = ds.map( lambda mf, rain_mask: tuple( [mf, rain_mask[0], rain_mask[1]] ), num_parallel_calls=-1)   
        
        # list of central h,w indexes from which to extract the region around
        if locations == ["All"]:
            li_hw_idxs = self.rain_data.get_locs_for_whole_map( self.m_params['region_grid_params'] ) #[ ([upper_h, lower_h]. [left_w, right_w]), ... ]
        else:
            li_hw_idxs = [ self.rain_data.find_idx_of_loc_region( _loc, self.m_params['region_grid_params'] ) for _loc in locations ] #[ (h_idx,w_idx), ... ]
        
        # Creating seperate datasets for each location
        li_ds = [ ds.map( lambda mf, rain, rmask : self.select_region( mf, rain, rmask, _idx[0], _idx[1]), num_parallel_calls=-1) for _idx in li_hw_idxs ]
        
        # Concatenating all datasets for each location
        batches_per_loc = int(batch_count/len(li_hw_idxs))
        for idx in range(len(li_ds)):
            li_ds[idx] = li_ds[idx].unbatch().batch( self.t_params['batch_size'], drop_remainder=True ).take(batches_per_loc)
            if idx==0:
                ds = li_ds[0]
            else:
                ds = ds.concatenate( li_ds[idx] )
        
        # pair of indexes locating the central location within the grid region extracted for any location
        idx_loc_in_region = np.floor_divide( self.m_params['region_grid_params']['outer_box_dims'], 2) #This specifies the index of the central location of interest within the (h,w) patch    
        return ds, idx_loc_in_region
        
    def select_region(self, mf, rain, rain_mask, h_idxs, w_idxs ):
        """ Extract the 1D specific point location relating to a [h_idxs, w_idxs] pair

            Args:
                mf : model field data
                rain : target rain data
                rain_mask : target rain mask
                h_idxs : int
                w_idxs : int

            Returns:
                tf.data.Dataset: 
        """

        """
            idx_h,idx_w: refer to the top left right index for the square region of interest this includes the region which is removed after cropping to calculate the loss during train step
        """
        mf = mf[ :, h_idxs[0]:h_idxs[1] , w_idxs[0]:w_idxs[1] , : ]
        rain = rain[ :, h_idxs[0]:h_idxs[1] , w_idxs[0]:w_idxs[1] ]
        rain_mask = rain_mask[ :, h_idxs[0]:h_idxs[1] , w_idxs[0]:w_idxs[1] ]
        return tf.expand_dims(mf,axis=0), tf.expand_dims(rain,axis=0), tf.expand_dims(rain_mask,axis=0) #Note: expand_dim for unbatch/batch compatibility

# endregion
