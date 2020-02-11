import numpy as np
import tensorflow as tf
from operator import itemgetter
import pandas as pd
from datetime import datetime
import pickle
from functools import reduce

class HParams():
    
    def __init__( self ,**kwargs ):  
        self._default_params()
        
        if( kwargs != None):
            self.params.update( kwargs) 

    def __call__(self):
        return self.params
    
    def _default_params(self):
        self.params = {}

class model_deepsd_hparameters(HParams):
    """
    model version 
    model version 5: Guassian BNN w/ 25 monte carlo forward passes /wo Discrete Continuous
    """
    def __init__(self, **kwargs):
        self.input_dims = kwargs['input_dims']
        self.output_dims = kwargs['output_dims']

        if "conv1_param_custom" in kwargs:
            self.conv1_param_custom = kwargs['conv1_param_custom']
        else:
            self.conv1_param_custom = None
        
        if "conv2_param_custom" in kwargs:
            self.conv2_param_custom = kwargs['conv2_param_custom']
        else:
            self.conv2_param_custom = None
        
        super( model_deepsd_hparameters, self ).__init__(**kwargs)


    def _default_params(self):

        # region params
        input_dims = self.input_dims #[39, 88]
        output_dims = self.output_dims #[156,352]

        #TODO: (change filter sizes back to the ones used in the paper)
        if type(self.conv1_param_custom) == dict: 

            CONV1_params = {    'filters':40, #512
                                'kernel_size': [5,5] , #[7,7] #TODO:use size from paper later
                                'activation':'relu',
                                'padding':'same',
                                'data_format':'channels_last',
                                'name':"Conv1" }
            CONV1_params.update(self.conv1_param_custom)
        if type(self.conv2_param_custom) == dict :
            conv2_kernel_size = np.ceil( np.ceil( np.array(output_dims)/np.array(input_dims) )*1.5 )  #This makes sure that each filter in conv2, sees at least two of the real non zero values. The zero values occur due to the upscaling
            CONV2_params = {    'filters':40, #512
                                'kernel_size':  conv2_kernel_size.astype(np.int32).tolist() , #TODO:use size from paper later
                                #each kernel covers 2 non original values from the upsampled tensor
                                'activation':'relu',
                                'padding':'same',
                                'data_format':'channels_last',
                                "name":"Conv2" }
            CONV2_params.update(self.conv2_param_custom)
                

        CONV3_params = {
                            'filters':1,
                            'kernel_size': [4,4], #[5,5] #TODO:use size from paper later
                            'activation':'relu',
                            'padding':'same',
                            'data_format':'channels_last',
                            "name":"Conv3"  }

        conv1_inp_channels = 17
        conv1_input_weights_per_filter = np.prod(CONV1_params['kernel_size']) * conv1_inp_channels
        conv1_input_weights_count =  CONV1_params['filters'] * conv1_input_weights_per_filter
        conv1_output_node_count = CONV1_params['filters']

        conv2_inp_channels = CONV1_params['filters']
        conv2_input_weights_per_filter = np.prod(CONV2_params['kernel_size']) * conv2_inp_channels
        conv2_input_weights_count = CONV2_params['filters'] * conv2_input_weights_per_filter
        conv2_output_node_count = CONV2_params['filters']

        conv3_inp_channels = CONV2_params['filters']
        conv3_input_weights_count = CONV3_params['filters'] * np.prod(CONV3_params['kernel_size'] ) * conv3_inp_channels
        conv3_output_node_count = CONV3_params['filters']
        #endregion params

        REC_ADAM_PARAMS = {
            "learning_rate":2e-4 , "warmup_proportion":0.6,
            "min_lr": 1e-4, "beta_1":0.9 , "beta_2": 1.0, "epsilon":1e-9 }
        LOOKAHEAD_PARAMS = { "sync_period":5 , "slow_step_size":0.75}

        model_type_settings = {'stochastic':False ,'stochastic_f_pass':50,
                                'distr_type':"Normal", 'discrete_continuous':True,
                                'precip_threshold':0.5 , 'var_model_type':"flipout"}

        self.params = {
            'model_name':"DeepSD",
            'model_version': 1,
            'model_type_settings':model_type_settings,

            'input_dims':input_dims,
            'output_dims':output_dims,

            'conv1_params': CONV1_params,
            'conv2_params': CONV2_params,
            'conv3_params': CONV3_params,

            'conv1_input_weights_count':conv1_input_weights_count,
            'conv1_output_node_count':conv1_output_node_count,
            'conv1_inp_channels':conv1_inp_channels,
            'conv1_input_weights_per_filter': conv1_input_weights_per_filter,

            'conv2_input_weights_count':conv2_input_weights_count,
            'conv2_output_node_count':conv2_output_node_count,
            'conv2_inp_channels': conv2_inp_channels,
            'conv2_input_weights_per_filter':conv2_input_weights_per_filter,

            'conv3_input_weights_count': conv3_input_weights_count,
            'conv3_output_node_count':conv3_output_node_count,
            'conv3_inp_channels':conv3_inp_channels,

            'gradients_clip_norm':150.0,
            
            'rec_adam_params':REC_ADAM_PARAMS,
            'lookahead_params':LOOKAHEAD_PARAMS
        }

class model_THST_hparameters(HParams):

    def __init__(self, **kwargs):
        """ 
            Hierachical 2D Convolution Model
        """

        super( model_THST_hparameters, self ).__init__(**kwargs)


    def _default_params(self):
        
        # region general params
        DROPOUT = 0.05

        #Deployment Settings        
        SEQ_LEN_FACTOR_REDUCTION = [4, 30, 3, 4 ] #This represents the rediction in seq_len when going from layer 1 to layer 2 and layer 2 to layer 3 in the encoder / decoder
        seq_len_for_highest_hierachy_level = 2

        #Low Memory Testing Settings
        SEQ_LEN_FACTOR_REDUCTION = [4, 2, 2, 2] #This represents the rediction in seq_len when going from layer 1 to layer 2 and layer 2 to layer 3 in the encoder / decoder
        seq_len_for_highest_hierachy_level = 2

        
        #NUM_OF_SPLITS = [ seq_len_for_highest_hierachy_level*SEQ_LEN_FACTOR_REDUCTION[1] , seq_len_for_highest_hierachy_level ] 
        NUM_OF_SPLITS = list(reversed((np.cumprod( list( reversed(SEQ_LEN_FACTOR_REDUCTION[1:] + [1] ) ) ) *seq_len_for_highest_hierachy_level ).tolist())) #for all rows except the first one
        
        # 5*3*4 5*3 5
        # endregion
        
        # region Model Specific Data Generator Params
        mf_time_scale = 0.25 #days
        rain_time_scale = 1 #days
        
        target_to_feature_time_ratio = SEQ_LEN_FACTOR_REDUCTION[0] # int(rain_time_scale/mf_time_scale)
        lookback_feature = reduce( (lambda x,y: x*y ), SEQ_LEN_FACTOR_REDUCTION ) * seq_len_for_highest_hierachy_level
        DATA_PIPELINE_PARAMS = {
            'lookback_feature':lookback_feature,
            'lookback_target': int(lookback_feature/target_to_feature_time_ratio),
            'target_to_feature_time_ratio' :  target_to_feature_time_ratio
        }
        # endregion

        # region --------------- ENCODER params -----------------
        encoder_layers = 5

        # region CLSTM params
        output_filters_enc = [10, 10, 10, 10] #output filters for each convLSTM2D layer in the encoder
        output_filters_enc = [2, 1, 1, 1] #NOTE: development settings
        output_filters_enc = output_filters_enc + output_filters_enc[-1:] #the last two layers in the encoder must output the same number of channels

        kernel_size_enc = [ (4,4) , (4,4) , (4,4), (4,4), (4,4)]
        kernel_size_enc = [ (3,3) , (3,3) , (3,3), (3,3), (3,3)]

        attn_layers = encoder_layers - 1
        key_depth = [0 ]*attn_layers  #This will be updated dynamically during the first iteration of the model
        attn_heads = [ 1]*attn_layers#NOTE: dev settings #must be a factor of h or w or c, so 100, 140 or 6 -> 2, 5, 7, 
        kq_downscale_stride = [1,13,13]
        kq_downscale_kernelshape = [1, 13, 13]
        vector_k_downscale_factor = 2
        vector_v_downscale_factor = 1
    
        ATTN_params_enc = [
            {'bias':None, 'total_key_depth': kd , 'total_value_depth':kd, 'output_depth': kd   ,
            'num_heads': nh , 'dropout_rate':DROPOUT, 'attention_type':"dot_product",
            "transform_value_antecedent":False ,  "transform_output":False }
            for kd, nh in zip( key_depth , attn_heads )
        ] #using key depth and Value depth smaller to reduce footprint

        ATTN_DOWNSCALING_params_enc = {
             "vector_k_downscale_factor":vector_k_downscale_factor,
             "vector_v_downscale_factor":vector_v_downscale_factor,
            'kq_downscale_stride': kq_downscale_stride, 'kq_downscale_kernelshape':kq_downscale_kernelshape
        }

        CLSTMs_params_enc = [
            {'filters':f , 'kernel_size':ks, 'padding':'same', 
                'return_sequences':True, 'dropout':DROPOUT, 'recurrent_dropout':DROPOUT,
                'stateful':True }
             for f, ks in zip( output_filters_enc, kernel_size_enc)
        ]
        # endregion


        ENCODER_PARAMS = {
            'encoder_layers': encoder_layers,
            'attn_layers': attn_layers,
            'CLSTMs_params' : CLSTMs_params_enc,
            'ATTN_params': ATTN_params_enc,
            'ATTN_DOWNSCALING_params_enc':ATTN_DOWNSCALING_params_enc,
            'seq_len_factor_reduction': SEQ_LEN_FACTOR_REDUCTION,
            'num_of_splits': NUM_OF_SPLITS,
            'dropout':DROPOUT
        }
        #endregion

        # region --------------- DECODER params -----------------
        decoder_layers = encoder_layers-2
        
        output_filters_dec = [ 2 ] + output_filters_enc[ decoder_layers-2:decoder_layers ] # This is written in the correct order
        kernel_size_dec = kernel_size_enc[ 1:1+decoder_layers  ]                             # This is written in the correct order

        CLSTMs_params_dec = [
            {'filters':f , 'kernel_size':ks, 'padding':'same', 
                'return_sequences':True, 'dropout':DROPOUT, 'gates_version':2, 'recurrent_dropout':DROPOUT,
                'stateful':True }
             for f, ks in zip( output_filters_dec, kernel_size_dec)
        ]

        DECODER_PARAMS = {
            'decoder_layers': decoder_layers,
            'CLSTMs_params' : CLSTMs_params_dec,
            'seq_len_factor_reduction': SEQ_LEN_FACTOR_REDUCTION[-decoder_layers:], #This is written in the correct order
            'num_of_splits': NUM_OF_SPLITS[-decoder_layers:],
            'dropout':DROPOUT
        }
        # endregion

        # region --------------- OUTPUT_LAYER_PARAMS -----------------
        output_filters = [ 25, 1 ]
        output_filters = [ 5, 1 ] #NOTE: development settings

        output_kernel_size = [ (4,4), (5,5) ]
        output_kernel_size = [ (3,3), (3,3) ] #NOTE: development settings


        OUTPUT_LAYER_PARAMS = [ 
            { "filters":fs, "kernel_size":ks ,  "padding":"same", "activation":'relu' } 
                for fs, ks in zip( output_filters, output_kernel_size )
         ]
        # endregion

        MODEL_VERSION = "1"
        dict_model_version = {1:False,2:False, 3:True, 4:True}
        """
            1: These are non Bayesian Small Models
            2: These are non Bayesian Large Models w/ Deformable & 
            3: These are Bayesian Small Models
            4: These are Bayesian Large Models
        """
        dict_deformable_model_version = {1: False, 2: True, 3:False, 4:True}

        model_type_settings = {
            'stochastic':dict_model_version[ int(list(MODEL_VERSION)[0]) ],
            'Deformable_Conv':dict_deformable_model_version[ int(list(MODEL_VERSION)[0]) ],
            'var_model_type':"Deterministic"
        }

        REC_ADAM_PARAMS = {
            "learning_rate":2e-4 , "warmup_proportion":0.6,
            "min_lr": 1e-4, "beta_1":0.9 , "beta_2": 1.0, "epsilon":1e-5
        }
        LOOKAHEAD_PARAMS = { "sync_period":3 , "slow_step_size":0.6}
        
        self.params = {
            'model_version': MODEL_VERSION,
            'model_name':"THST",
            'model_type_settings':model_type_settings,


            'encoder_params':ENCODER_PARAMS,
            'decoder_params':DECODER_PARAMS,
            'output_layer_params':OUTPUT_LAYER_PARAMS,
            'data_pipeline_params':DATA_PIPELINE_PARAMS,


            'rec_adam_params':REC_ADAM_PARAMS,
            'lookahead_params':LOOKAHEAD_PARAMS,
            'gradients_clip_norm':200

        }


#region vandal
class train_hparameters(HParams):
    def __init__(self, **kwargs):
        super( train_hparameters, self).__init__(**kwargs)

    def _default_params(self):
        # region default params 
        NUM_PARALLEL_CALLS = tf.data.experimental.AUTOTUNE,
        EPOCHS = 342 #equivalent to Vandal
        CHECKPOINTS_TO_KEEP = 50
        CHECKPOINTS_TO_KEEP_EPOCH = 5
        CHECKPOINTS_TO_KEEP_BATCH = 5

        start_date = np.datetime64('1981-01-01')
        end_date = np.datetime64('2015-012-31')
        TOTAL_DATUMS = np.timedelta64( end_date-start_date, 'D').astype(int)

        #need to use this ration ,0.73529, for training
        TRAIN_SET_SIZE_ELEMENTS = int(TOTAL_DATUMS*0.53529411764)
        VAL_SET_SIZE_ELEMENTS = int(TOTAL_DATUMS*0.20)
        BATCH_SIZE = 2
        DATA_DIR = "./Data"
        EARLY_STOPPING_PERIOD = 10
        BOOL_WATER_MASK = pickle.load( open( "Images/water_mask_156_352.dat","rb" ) )

        

        #endregion
        self.params = {
            'batch_size':BATCH_SIZE,
            'epochs':EPOCHS,
            'total_datums':TOTAL_DATUMS,
            'early_stopping_period':EARLY_STOPPING_PERIOD,

            'train_set_size_elements':TRAIN_SET_SIZE_ELEMENTS,
            'train_set_size_batches':TRAIN_SET_SIZE_ELEMENTS//BATCH_SIZE,
            'val_set_size_elements':VAL_SET_SIZE_ELEMENTS,
            'val_set_size_batches':VAL_SET_SIZE_ELEMENTS//BATCH_SIZE,

            'checkpoints_to_keep':CHECKPOINTS_TO_KEEP,
            'checkpoints_to_keep_epoch':CHECKPOINTS_TO_KEEP_EPOCH,
            'checkpoints_to_keep_batch':CHECKPOINTS_TO_KEEP_BATCH,
            
            'dataset_trainval_batch_reporting_freq':0.10,
            'num_parallel_calls':NUM_PARALLEL_CALLS,

            'data_dir': DATA_DIR,

            'bool_water_mask': BOOL_WATER_MASK

        }
        
class test_hparameters(HParams):
    def __init__(self, **kwargs):
        super( test_hparameters, self).__init__(**kwargs)
    
    def _default_params(self):
        NUM_PARALLEL_CALLS = tf.data.experimental.AUTOTUNE
        BATCH_SIZE = 2

        MODEL_RECOVER_METHOD = 'checkpoint_epoch'
    
        trainable = 'test'
        start_date = np.datetime64('1981-01-01')
        end_date = np.datetime64('2015-12-31')
        TOTAL_DATUMS = np.timedelta64( end_date-start_date, 'D').astype(int)

        TEST_SET_SIZE_ELEMENTS = int( TOTAL_DATUMS * (1-0.2-0.53529411764) )
        STARTING_TEST_ELEMENT = TOTAL_DATUMS - TEST_SET_SIZE_ELEMENTS
        
        dates_tss = pd.date_range( end=datetime(2015,12,31), periods=TEST_SET_SIZE_ELEMENTS, freq='D',normalize=True)
        EPOCHS = list ( (dates_tss - pd.Timestamp("1970-01-01") ) // pd.Timedelta('1s') )

        BOOL_WATER_MASK = pickle.load( open( "Images/water_mask_156_352.dat","rb" ) )


        self.params = {
            'batch_size':BATCH_SIZE,
            'starting_test_element':STARTING_TEST_ELEMENT,
            'test_set_size_elements': TEST_SET_SIZE_ELEMENTS,

            'dataset_pred_batch_reporting_freq':0.25,
            'num_parallel_calls':NUM_PARALLEL_CALLS,

            'model_recover_method':MODEL_RECOVER_METHOD,
            'trainable':trainable,

            'script_dir':None,

            'epochs':EPOCHS,

            'bool_water_mask': BOOL_WATER_MASK

        }
# endregion

# region ATI
class train_hparameters_ati(HParams):
    def __init__(self, **kwargs):
        self.lookback_target = kwargs['lookback_target']
        super( train_hparameters_ati, self).__init__(**kwargs)

    def _default_params(self):
        # region -------data pipepline vars
        trainable = True
        MASK_FILL_VALUE = {
                                    "rain":-1.0,
                                    "model_field":-1.0 
        }

        NORMALIZATION_SCALES = {
                                    "rain":200.0,
                                    "model_fields": np.array([1.0,1.0,1.0,1.0,1.0,1.0]) #TODO: Find the appropriate scaling terms for each of the model fields, by inspecting data
                                                #- unknown_local_param_137_128
                                                # - unknown_local_param_133_128,  # - air_temperature, # - geopotential
                                                # - x_wind, # - y_wind
        }

        WINDOW_SHIFT = self.lookback_target
        BATCH_SIZE = 2
        # endregion


        NUM_PARALLEL_CALLS = tf.data.experimental.AUTOTUNE
        EPOCHS = 200
        CHECKPOINTS_TO_KEEP = 3
        CHECKPOINTS_TO_KEEP_EPOCH = 5
        CHECKPOINTS_TO_KEEP_BATCH = 5

        # region ---- data information

        target_start_date = np.datetime64('1950-01-01') + np.timedelta64(10592,'D')
        feature_start_date = np.datetime64('1970-01-01') + np.timedelta64(78888, 'h')
        
        tar_end_date=  target_start_date + np.timedelta64( 14822, 'D')
        feature_end_date  = np.datetime64( feature_start_date + np.timedelta64(16072, '6h'), 'D')
        
        if feature_start_date > target_start_date :
            train_start_date = feature_start_date
        else:
            train_start_date = target_start_date

        if tar_end_date < feature_end_date :
            end_date = tar_end_date
        else:
            end_date = feature_end_date

        #train_start_date = np.max(feature_start_date, target_start_date)
        #end_date = np.min( tar_end_date, feature_end_date)
        val_start_date =    np.datetime64( train_start_date + (end_date - train_start_date)*0.6, 'D' )
        val_end_date =      np.datetime64( train_start_date + (end_date - train_start_date)*0.8, 'D' )

        #TOTAL_DATUMS = int(end_date - start_date)//WINDOW_SHIFT - lookback  #By datums here we mean windows, for the target
        TOTAL_DATUMS_TARGET = np.timedelta64(end_date - train_start_date,'D')  / WINDOW_SHIFT   #Think of better way to get the np.product info from model_params to train params
        TOTAL_DATUMS_TARGET = TOTAL_DATUMS_TARGET.astype(int)
        # endregion

        #TODO: correct the train_set_size_elems
        TRAIN_SET_SIZE_ELEMENTS = int(TOTAL_DATUMS_TARGET*0.6)
        
        VAL_SET_SIZE_ELEMENTS = int(TOTAL_DATUMS_TARGET*0.2)
         
        DATA_DIR = "./Data/Rain_Data_Nov19" 
        EARLY_STOPPING_PERIOD = 10


 
        self.params = {

            'batch_size':BATCH_SIZE,
            'epochs':EPOCHS,
            'total_datums':TOTAL_DATUMS_TARGET,
            'early_stopping_period':EARLY_STOPPING_PERIOD,
            'trainable':trainable,


            'train_set_size_elements':TRAIN_SET_SIZE_ELEMENTS,
            'train_set_size_batches':TRAIN_SET_SIZE_ELEMENTS//BATCH_SIZE,
            'val_set_size_elements':VAL_SET_SIZE_ELEMENTS,
            'val_set_size_batches':VAL_SET_SIZE_ELEMENTS//BATCH_SIZE,

            'checkpoints_to_keep':CHECKPOINTS_TO_KEEP,
            'checkpoints_to_keep_epoch':CHECKPOINTS_TO_KEEP_EPOCH,
            'checkpoints_to_keep_batch':CHECKPOINTS_TO_KEEP_BATCH,


            'dataset_trainval_batch_reporting_freq':0.1,
            'num_parallel_calls':NUM_PARALLEL_CALLS,

            'train_monte_carlo_samples':1,

            'data_dir': DATA_DIR,

            
            'mask_fill_value':MASK_FILL_VALUE,
            'normalization_scales' : NORMALIZATION_SCALES,
            'window_shift': WINDOW_SHIFT,

            'train_start_date':train_start_date,
            'val_start_date':val_start_date,
            'val_end_date':val_end_date,

            'feature_start_date':feature_start_date,
            'target_start_date':target_start_date

        }

class test_hparameters_ati(HParams):
    def __init__(self, **kwargs):
        self.lookback_target = kwargs['lookback_target']
        super( test_hparameters_ati, self).__init__(**kwargs)
    
    def _default_params(self):
        pass
        # region -------data pipepline vars
        trainable = True
        MASK_FILL_VALUE = {
            "rain":-1.0,
            "model_field":-1.0 }

        NORMALIZATION_SCALES = {
            "rain":200.0,
            "model_fields": np.array([1.0,1.0,1.0,1.0,1.0,1.0]) #TODO: Find the appropriate scaling terms for each of the model fields, by inspecting data
                        #- unknown_local_param_137_128
                        # - unknown_local_param_133_128,  # - air_temperature, # - geopotential
                        # - x_wind, # - y_wind
        }

        WINDOW_SHIFT = self.lookback_target
        NUM_PARALLEL_CALLS = tf.data.experimental.AUTOTUNE
        BATCH_SIZE = 2
        N_PREDS = 50
        MODEL_RECOVER_METHOD = 'checkpoint_epoch'
        # endregion


        # region ---- data information

        target_start_date = np.datetime64('1950-01-01') + np.timedelta64(10592,'D')
        feature_start_date = np.datetime64('1970-01-01') + np.timedelta64(78888, 'h')
        
        tar_end_date=  target_start_date + np.timedelta64( 14822, 'D')
        feature_end_date  = feature_start_date + np.timedelta64(16072, '6h') #TODO: change when new data available
        
        if feature_start_date > target_start_date :
            train_start_date = feature_start_date
        else:
            train_start_date = target_start_date

        if tar_end_date < feature_end_date :
            end_date = tar_end_date
        else:
            end_date = feature_end_date

        #train_start_date = np.max(feature_start_date, target_start_date)
        #end_date = np.min( tar_end_date, feature_end_date)
        val_start_date = train_start_date + (end_date - train_start_date)*0.6
        val_end_date = train_start_date + (end_date - train_start_date)*0.8

        test_start_date = train_start_date + (end_date - train_start_date)*0.8
        test_end_date = end_date

        #TOTAL_DATUMS = int(end_date - start_date)//WINDOW_SHIFT - lookback  #By datums here we mean windows, for the target
        TOTAL_DATUMS_TARGET = ( np.timedelta64(end_date - test_start_date,'D') - (self.lookback_target - 1) )  // WINDOW_SHIFT   #Think of better way to get the np.product info from model_params to train params
        TOTAL_DATUMS_TARGET = TOTAL_DATUMS_TARGET.astype(int)
        # endregion

        TEST_SET_SIZE_DATUMS_TARGET = int( TOTAL_DATUMS_TARGET * 0.2)

        date_tss = pd.date_range( end=test_end_date, periods=TEST_SET_SIZE_DATUMS_TARGET, freq='D',normalize=True)
        EPOCHS = list ( (dates_tss - pd.Timestamp("1970-01-01") ) // pd.Timedelta('1s') )

        DATA_DIR = "./Data/Rain_Data_Nov19" 

        self.params = {
            'batch_size':BATCH_SIZE,
            'trainable':trainable,
            
            'total_datums':TOTAL_DATUMS_TARGET,
            'test_set_size_elements':TEST_SET_SIZE_DATUMS_TARGET,
            'num_preds':N_PREDS,

            'model_recover_method':MODEL_RECOVER_METHOD,
            'script_dir':None,
            'data_dir':DATA_DIR,
            
            'epochs':EPOCHS,
            
            'mask_fill_value':MASK_FILL_VALUE,
            'normalization_scales' : NORMALIZATION_SCALES,
            'window_shift': WINDOW_SHIFT,

            'test_start_date':test_start_date,
            'test_end_date':test_end_date,

            'feature_start_date':feature_start_date,
            'feature_start_end':feature_end_date
        }


#endregion ATI
