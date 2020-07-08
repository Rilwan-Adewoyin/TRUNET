import numpy as np
import math
import tensorflow as tf
from operator import itemgetter
import pandas as pd
from datetime import datetime
import pickle
from functools import reduce
from tensorflow.python.training.tracking import data_structures

class HParams():
    """Inheritable class for the parameter classes
        Example of how to use
        hparams = Hparams(**kwargs)
        params_dict = hparams() #dictionary containing parameters
    """    
    def __init__( self ,**kwargs ):  
        
        self._default_params(**kwargs)
        
        if( kwargs != None):
            self.params.update( kwargs)
    
    def __call__(self):
        return self.params
    
    def _default_params(self,**kwargs):
        self.params = {}

class MParams(HParams):
    """Class to be inherited by parameter classes which are designed to return
        parameters for models
    """    
    def __init__(self,**kwargs):
     
        # Parameters related the extraction of the 2D patches of data
        self.regiongrid_param_adjustment()

        super(MParams,self).__init__(**kwargs)
                 
    def regiongrid_param_adjustment(self):
        """Creates a 'region_grid_params' dictionary containing
            information on the sizes and location of patches to be extracted
        """        
        if not hasattr(self, 'params'):
            self.params = {}

        self.params.update(
            {'region_grid_params':{
                'outer_box_dims':[16,16],
                'inner_box_dims':[4,4],
                'vertical_shift':4,
                'horizontal_shift':4,
                'input_image_shape':[100,140]}
            }
        )
        vertical_slides = (self.params['region_grid_params']['input_image_shape'][0] - self.params['region_grid_params']['outer_box_dims'][0] +1 )// self.params['region_grid_params']['vertical_shift']
        horizontal_slides = (self.params['region_grid_params']['input_image_shape'][1] - self.params['region_grid_params']['outer_box_dims'][1] +1 ) // self.params['region_grid_params']['horizontal_shift']
        self.params['region_grid_params'].update({'slides_v_h':[vertical_slides, horizontal_slides]})

class model_TRUNET_hparameters(MParams):
    """Parameters Class for the TRUNET Encoder-Decoder model
    """
    def __init__(self, **kwargs):

        self.conv_ops_qk = kwargs['model_type_settings'].get('conv_ops_qk',False)
        kwargs['model_type_settings'].pop('conv_ops_qk',None)
        
        super( model_TRUNET_hparameters, self ).__init__(**kwargs)

    def _default_params( self, **kwargs ):
        
        model_type_settings = kwargs.get('model_type_settings', {})        

        # region ---  learning/convergence/regularlisation params
        REC_ADAM_PARAMS = {
            "learning_rate":7e-4,   "warmup_proportion":0.65,
            "min_lr":2.5e-4,         "beta_1":0.9,               "beta_2":0.99,
            "amsgrad":True,         "decay":0.0009,              "epsilon":5e-8 } #Rectified Adam params
        
        REC_ADAM_PARAMS = {
            "learning_rate":7e-4,   "warmup_proportion":0.65,
            "min_lr":2.5e-4,         "beta_1":0.6,               "beta_2":0.9,
            "amsgrad":True,         "decay":0.0008,              "epsilon":5e-8 } #Rectified Adam params            
        
        DROPOUT =   model_type_settings.get('do',0.0)
        ido =       model_type_settings.get('ido',0.0) # Dropout for input into GRU
        rdo =       model_type_settings.get('rdo',0.0) # Dropout for recurrent input into GRU
        kernel_reg   = None  #regularlization for input to GRU
        recurrent_reg = None #regularlization for recurrent input to GRU
        bias_reg = tf.keras.regularizers.l2(0.0)
        bias_reg_attn = tf.keras.regularizers.l2(0.00)
        # endregion

        #region --- Key Model Size Settings
        seq_len_for_highest_hierachy_level = 4 # Seq length of GRU operations in highest level of encoder

        seq_len_factor_reduction = [4, 7]    #This represents the reduction in seq_len when going from layer 1 
                                                #to layer 2 and layer 2 to layer 3 in the encoder / decoder
                                                    # 6hrs, 1Day, 1Week
        # endregion

        # region Model Specific Data Generating Params
        target_to_feature_time_ratio = seq_len_factor_reduction[0] 
        lookback_feature = reduce( (lambda x,y: x*y ), seq_len_factor_reduction ) * seq_len_for_highest_hierachy_level #Temporal length of input elements
        DATA_PIPELINE_PARAMS = {
            'lookback_feature':lookback_feature,
            'lookback_target': int(lookback_feature/target_to_feature_time_ratio) #Temporal length of output elements
        }
        # endregion
           
        # region --- ENCODER params 
        enc_layer_count        = len( seq_len_factor_reduction ) + 1
        attn_layers_count = enc_layer_count - 1

        # ConvGRU params
        if model_type_settings.get('large_model',False) == False:
            filters = 72 # no. of filters in all conv operations in ConvGRU units
        else:
            filters = 120

        kernel_size_enc        = [ (4,4) ] * ( enc_layer_count )             
        print("Check appropriate stateful is being used for multi gpu status")
        stateful = False                       

        # Attention params
        attn_heads = [ 8 ]*attn_layers_count            #NOTE:Must be a factor of h or w or c. h,w are dependent on model type so make it a multiple of c = 8
        kq_downscale_stride = [1, 4, 4]                 #[1, 8, 8] 
        kq_downscale_kernelshape = kq_downscale_stride
        key_depth = [filters]*attn_layers_count # Key vector size
        val_depth = [ int( np.prod( self.params['region_grid_params']['outer_box_dims'] ) * filters * 2 )] *attn_layers_count
                  
        attn_layers_num_of_splits = list(reversed((np.cumprod( list( reversed(seq_len_factor_reduction[1:] + [1] ) ) ) *seq_len_for_highest_hierachy_level ).tolist())) 
            #Each encoder layer receives a seq of 3D tensors from layer below. 
            # NUM_OF_SPLITS is how many chunks the incoming tensors are split into

        attn_params_enc = [
            {'bias':None, 'total_key_depth': kd  ,'total_value_depth':vd, 'output_depth': vd   ,
            'num_heads': nh , 'dropout_rate':DROPOUT, 'max_relative_position':None,
            "transform_value_antecedent":True,  "transform_output":True, 
            'implementation':1, 'conv_ops_qk':self.conv_ops_qk,
            "value_conv":{ "filters":int(filters * 2), 'kernel_size':[3,3] ,'use_bias':True, "activation":'relu', 'name':"v", 'bias_regularizer':bias_reg_attn, 'padding':'same' },
            "output_conv":{ "filters":int(filters * 2), 'kernel_size':[3,3] ,'use_bias':True, "activation":'relu', 'name':"outp", 'bias_regularizer':bias_reg_attn,'padding':'same' }
            } 
            for kd, vd ,nh, idx in zip( key_depth, val_depth, attn_heads,range(attn_layers_count) )
        ] #list of param dictionaries for each Inter Layer Cross Attention unit in the encoder
            #Note: bias refers to any attention masking, use_bias refers to bias used in convolutional ops

        attn_downscaling_params_enc = {
            'kq_downscale_stride': kq_downscale_stride,
            'kq_downscale_kernelshape':kq_downscale_kernelshape
        } #List of params for 3D average pooling operations
        
        CGRUs_params_enc = [
            {'filters':filters , 'kernel_size':ks, 'padding':'same', 
                'return_sequences':True, 'dropout':ido, 'recurrent_dropout':rdo,
                'stateful':stateful, 'recurrent_regularizer': recurrent_reg, 'kernel_regularizer':kernel_reg,
                'bias_regularizer':bias_reg, 'implementation':1 ,'layer_norm':None }
             for ks in kernel_size_enc
        ] #list of params for each ConvGRU layer in the Encoder
      
        ENCODER_PARAMS = {
            'enc_layer_count': enc_layer_count,
            'attn_layers_count': attn_layers_count,
            'CGRUs_params' : CGRUs_params_enc,
            'ATTN_params': attn_params_enc,
            'ATTN_DOWNSCALING_params_enc':attn_downscaling_params_enc,
            'seq_len_factor_reduction': seq_len_factor_reduction,
            'attn_layers_num_of_splits': attn_layers_num_of_splits,
            'dropout':DROPOUT
        }
        #endregion

        # region --- DECODER params 
        decoder_layer_count = enc_layer_count-2
                
        kernel_size_dec = kernel_size_enc[ 1:1+decoder_layer_count  ]           
                                              
            #Each decoder layer sends in values into the layer below. 
        CGRUs_params_dec = [
            {'filters':filters , 'kernel_size':ks, 'padding':'same', 
                'return_sequences':True, 'dropout':ido,
                'recurrent_dropout':rdo, 
                'kernel_regularizer':kernel_reg,
                'recurrent_regularizer': recurrent_reg,
                'bias_regularizer':bias_reg,
                'stateful':stateful,
                'implementation':1 ,'layer_norm':[ None, None ]  }
             for ks in kernel_size_dec ] #list of dictionaries containing params for each ConvGRU layer in decoder

        decoder_layers_num_of_splits = attn_layers_num_of_splits[:decoder_layer_count]
            #Each output from a decoder layer is split into n chunks the fed to n different nodes in the layer below. param above tracks teh value n for each dec layer
        seq_len_factor_expansion = seq_len_factor_reduction[-decoder_layer_count:]
        DECODER_PARAMS = {
            'decoder_layer_count': decoder_layer_count,
            'CGRUs_params' : CGRUs_params_dec,
            'seq_len_factor_expansion': seq_len_factor_expansion, #This is written in the correct order
            'seq_len': decoder_layers_num_of_splits,
            'attn_layer_no_splits':attn_layers_num_of_splits,
            'dropout':DROPOUT
        }
        # endregion

        # region --- OUTPUT_LAYER_PARAMS and Upscaling
        output_filters = [  int(  8*(((filters*2)/4)//8)), 1 ] 

        output_kernel_size = [ (3,3), (3,3) ] 
        activations = ['relu','linear']

        OUTPUT_LAYER_PARAMS = [ 
            { "filters":fs, "kernel_size":ks , "padding":"same", "activation":act, 'bias_regularizer': None } 
                for fs, ks, act in zip( output_filters, output_kernel_size, activations )
        ]
        # endregion
        
        self.params.update( {
            'model_name':"TRUNET",
            'model_type_settings':model_type_settings,
    
            'encoder_params':ENCODER_PARAMS,
            'decoder_params':DECODER_PARAMS,
            'output_layer_params':OUTPUT_LAYER_PARAMS,
            'data_pipeline_params':DATA_PIPELINE_PARAMS,

            'rec_adam_params':REC_ADAM_PARAMS,
            'dropout':DROPOUT,
            'clip_norm':6.5
            } )

class model_SimpleConvGRU_hparamaters(MParams):

    def __init__(self, **kwargs):
        self.dc = kwargs.get('model_type_settings',{}).get('discrete_continuous',False)
        self.stoc = kwargs.get('model_type_settings',{}).get('stochastic',False)
        super(model_SimpleConvGRU_hparamaters, self).__init__(**kwargs)
    
    def _default_params(self,**kwargs):
        dropout = kwargs.get('dropout',0.0)

        #region --- ConvLayers
        layer_count = 4 
        filters = 80
        print("Check appropriate stateful is being used for multi gpu status")
        stateful = False
        kernel_sizes = [[4,4]]*layer_count
        paddings = ['same']*layer_count
        return_sequences = [True]*layer_count
        input_dropout = [kwargs.get('inp_dropout',0.0) ]*layer_count #[0.0]*layer_count
        recurrent_dropout = [ kwargs.get('rec_dropout',0.0)]*layer_count #[0.0]*layer_count

        ConvGRU_layer_params = [ { 'filters':filters, 'kernel_size':ks , 'padding': ps,
                                'return_sequences':rs, "dropout": dp , "recurrent_dropout":rdp,
                                'kernel_regularizer': None,
                                'recurrent_regularizer': None,
                                'bias_regularizer':tf.keras.regularizers.l2(0.2),
                                'layer_norm': None,
                                'implementation':1, 'stateful':stateful  }
                                for ks,ps,rs,dp,rdp in zip( kernel_sizes, paddings, return_sequences, input_dropout, recurrent_dropout)  ]

        conv1_layer_params = {'filters': int(  8*(((filters*2)/3)//8)) , 'kernel_size':[3,3], 'activation':'relu','padding':'same','bias_regularizer':tf.keras.regularizers.l2(0.2) }  

        outpconv_layer_params = {'filters':1, 'kernel_size':[3,3], 'activation':'linear','padding':'same','bias_regularizer':tf.keras.regularizers.l2(0.2) }
        #endregion

        #region --- Data pipeline and optimizers
        target_to_feature_time_ratio = 4
        lookback_feature = 28*target_to_feature_time_ratio  
        DATA_PIPELINE_PARAMS = {
            'lookback_feature':lookback_feature,
            'lookback_target': int(lookback_feature/target_to_feature_time_ratio),
            'target_to_feature_time_ratio' :  target_to_feature_time_ratio
        }


        REC_ADAM_PARAMS = {
            "learning_rate":7e-4,   "warmup_proportion":0.65,
            "min_lr":2.5e-4,         "beta_1":0.9,               "beta_2":0.99,
            "amsgrad":True,         "decay":0.0009,              "epsilon":5e-8 } #Rectified Adam params
        
        REC_ADAM_PARAMS = {
            "learning_rate":8e-4,   "warmup_proportion":0.65,
            "min_lr":1.5e-4,         "beta_1":0.6,               "beta_2":0.9,
            "amsgrad":True,         "decay":0.0009,              "epsilon":5e-8 } #Rectified Adam params  

        LOOKAHEAD_PARAMS = { "sync_period":1 , "slow_step_size":0.99 }

        # endregion
        model_type_settings = kwargs.get('model_type_settings',{})

        self.params.update( {
            'model_name':'SimpleConvGRU',
            'layer_count':layer_count,
            'ConvGRU_layer_params':ConvGRU_layer_params,
            'conv1_layer_params':conv1_layer_params,
            'outpconv_layer_params': outpconv_layer_params,
            'dropout': dropout,

            'data_pipeline_params':DATA_PIPELINE_PARAMS,
            'model_type_settings':model_type_settings,

            'rec_adam_params':REC_ADAM_PARAMS,
            'lookahead_params':LOOKAHEAD_PARAMS,
            'clip_norm':6.5
        })

class TRUNET_EF_hparams(HParams):

    def __init__(self, **kwargs):
        """  
        """
        self.big = kwargs.get('model_type_settings',{}).get('big',False)
        self.conv_ops_qk = kwargs['model_type_settings'].get('conv_ops_qk',False)
        kwargs['model_type_settings'].pop('conv_ops_qk',None)

        super( TRUNET_EF_hparams, self ).__init__(**kwargs)

    def _default_params( self, **kwargs ):

        model_type_settings = kwargs.get('model_type_settings',{})        
        
        # region ---  learning/convergence/regularlisation params
        REC_ADAM_PARAMS = {
            "learning_rate":5e-4,   "warmup_proportion":0.65,
            "min_lr":2.5e-4,          "beta_1":0.70,               "beta_2":0.99,
            "amsgrad":True,         "decay":0.0008,              "epsilon":5e-8 }

        DROPOUT =   model_type_settings.get('do',0.0)
        ido =       model_type_settings.get('ido',0.0) # Dropout for input into GRU
        rdo =       model_type_settings.get('rdo',0.0) # Dropout for recurrent input into GRU
        kernel_reg   = None  #regularlization for input to GRU
        recurrent_reg = None #regularlization for recurrent input to GRU
        bias_reg = tf.keras.regularizers.l2(0.0)
        bias_reg_attn = tf.keras.regularizers.l2(0.00)
        # endregion
        
        #region Key Model Size Settings

        seq_len_for_highest_hierachy_level = 4 

        SEQ_LEN_FACTOR_REDUCTION = [4, 5, 3 ]        # 6hrs -> 4days -> 12 days
        # endregion
        
        # region Model Specific Data Generator Params
        target_to_feature_time_ratio = SEQ_LEN_FACTOR_REDUCTION[0] 
        lookback_feature = reduce( (lambda x,y: x*y ), SEQ_LEN_FACTOR_REDUCTION ) * seq_len_for_highest_hierachy_level
        DATA_PIPELINE_PARAMS = {
            'lookback_feature':lookback_feature,
            'lookback_target': int(lookback_feature/target_to_feature_time_ratio),
            'target_to_feature_time_ratio' :  target_to_feature_time_ratio
        }
        # endregion

        # region --------------- ENCODER params -----------------
        enc_layer_count        = len( SEQ_LEN_FACTOR_REDUCTION ) + 1
        h_w_enc = [ [32, 64], [16, 32], [8, 16] ]
        h_w_dec = [ [8, 16], [16, 32], [32, 64] ]

        # region CLSTM params
        _filters = [ 16, 32, 48, 64]
            
        output_filters_enc     = _filters
                
        kernel_size_enc        = [ (4,4), (3,3), (2,2 ), (2,2) ]           
        recurrent_regularizers = [ None ] * (enc_layer_count) 
        kernel_regularizers    = [ None ] * (enc_layer_count)
        bias_regularizers      = [ tf.keras.regularizers.l2(0.00) ]*(enc_layer_count) # [ tf.keras.regularizers.l2(0.02) ] * (enc_layer_count)  #changed
        recurrent_dropouts     = [ kwargs.get('rec_dropout',0.0) ]*(enc_layer_count)
        input_dropouts         = [ kwargs.get('inp_dropout',0.0) ]*(enc_layer_count)
        stateful               = True                       #True if testing on single location , false otherwise
        layer_norms            = lambda: None               #lambda: tf.keras.layers.LayerNormalization(axis=[-1], center=False, scale=False ) #lambda: None

        attn_layers_count = enc_layer_count - 1
        attn_heads = [ 8 ]*attn_layers_count                #[ 8 ]*attn_layers_count        #[5]  #NOTE:Must be a factor of h or w or c. h,w are dependent on model type so make it a multiple of c = 8
        
        kq_downscale_stride = [1, 4, 4]                 #[1, 8, 8] 
        kq_downscale_kernelshape = kq_downscale_stride
        key_depth = _filters[1:]
        val_depth = [ int( np.prod( h_w ) * output_filters_enc[idx] * 2 ) for h_w in h_w_enc  ]
                  
            
        ATTN_LAYERS_NUM_OF_SPLITS = list(reversed((np.cumprod( list( reversed(SEQ_LEN_FACTOR_REDUCTION[1:] + [1] ) ) ) *seq_len_for_highest_hierachy_level ).tolist())) 
            #Each encoder layer receives a seq of 3D tensors from layer below. NUM_OF_SPLITS codes in how many chunks to devide the incoming data. NOTE: This is defined only for Encoder-Attn layers
        ATTN_params_enc = [
            {'bias':None, 'total_key_depth': kd  ,'total_value_depth':vd, 'output_depth': vd   ,
            'num_heads': nh , 'dropout_rate':DROPOUT, 'max_relative_position':None,
            "transform_value_antecedent":True,  "transform_output":True, 
            'implementation':1,
            "value_conv":{ "filters":int(output_filters_enc[idx] * 2), 'kernel_size':[3,3] ,'use_bias':True, "activation":'relu', 'name':"v", 'bias_regularizer':tf.keras.regularizers.l2(0.00002), 'padding':'same' },
            "output_conv":{ "filters":int(output_filters_enc[idx] * 2), 'kernel_size':[3,3] ,'use_bias':True, "activation":'relu', 'name':"outp", 'bias_regularizer':tf.keras.regularizers.l2(0.00002),'padding':'same' }
            } 
            for kd, vd ,nh, idx in zip( key_depth, val_depth, attn_heads,range(attn_layers_count) )
        ] 
        #Note: bias refers to masking attention places

        ATTN_DOWNSCALING_params_enc = {
            'kq_downscale_stride': kq_downscale_stride,
            'kq_downscale_kernelshape':kq_downscale_kernelshape
        }
        
        CGRUs_params_enc = [
            {'filters':f , 'kernel_size':ks, 'padding':'same', 
                'return_sequences':True, 'return_state':True ,'dropout':ido, 'recurrent_dropout':rd,
                'stateful':stateful, 'recurrent_regularizer': rr, 'kernel_regularizer':kr,
                'bias_regularizer':br, 'implementation':1 ,'layer_norm':layer_norms() }
             for f, ks, rr, kr, br, rd, ido in zip( output_filters_enc, kernel_size_enc, recurrent_regularizers, kernel_regularizers, bias_regularizers, recurrent_dropouts, input_dropouts )
        ]
        # endregion
        
        ENCODER_PARAMS = {
            'enc_layer_count': enc_layer_count,
            'attn_layers_count': attn_layers_count,
            'CGRUs_params' : CGRUs_params_enc,
            'ATTN_params': ATTN_params_enc,
            'ATTN_DOWNSCALING_params_enc':ATTN_DOWNSCALING_params_enc,
            'seq_len_factor_reduction': SEQ_LEN_FACTOR_REDUCTION,
            'attn_layers_num_of_splits': ATTN_LAYERS_NUM_OF_SPLITS,
            'dropout':DROPOUT,
            'h_w_enc':h_w_enc

        }
        #endregion

        # region --------------- DECODER params -----------------
        decoder_layer_count = enc_layer_count-2

        output_filters_dec = output_filters_enc[:1] + output_filters_enc[ :decoder_layer_count ]  #The first part list needs to be changed, only works when all convlstm layers have the same number of filters
                
        kernel_size_dec = kernel_size_enc[ 1:1+decoder_layer_count  ]           
                                              
            #Each decoder layer sends in values into the layer below. 
        CGRUs_params_dec = [
            {'filters':f , 'kernel_size':ks, 'padding':'same', 
                'return_sequences':True, 'dropout':ido,
                'recurrent_dropout':rdo, 
                'kernel_regularizer':kr,
                'recurrent_regularizer': rr,
                'bias_regularizer':br,
                'stateful':False,
                'implementation':1 ,'layer_norm':[ layer_norms(),layer_norms() ]  }
             for f, ks, ido, rdo, rr, kr, br  in zip( output_filters_dec, kernel_size_dec, input_dropouts, recurrent_dropouts, recurrent_regularizers, kernel_regularizers, bias_regularizers)
        ]
        DECODER_LAYERS_NUM_OF_SPLITS = ATTN_LAYERS_NUM_OF_SPLITS[:decoder_layer_count]
            #Each output from a decoder layer is split into n chunks the fed to n different nodes in the layer below. param above tracks teh value n for each dec layer
        SEQ_LEN_FACTOR_EXPANSION = SEQ_LEN_FACTOR_REDUCTION[-decoder_layer_count:]
        DECODER_PARAMS = {
            'decoder_layer_count': decoder_layer_count,
            'CGRUs_params' : CGRUs_params_dec,
            'seq_len_factor_expansion': SEQ_LEN_FACTOR_EXPANSION, #This is written in the correct order
            'seq_len': DECODER_LAYERS_NUM_OF_SPLITS,
            'attn_layer_no_splits':ATTN_LAYERS_NUM_OF_SPLITS,
            'dropout':DROPOUT,
            'h_w_dec':h_w_dec
        }
        # endregion

        # region --------------- OUTPUT_LAYER_PARAMS and Upscaling-----------------

 
        output_filters = [  int(  8*(((output_filters_dec[-1]*2)/4)//8)) ] + [ 1 ]  #[ 2, 1 ]   # [ 8, 1 ]

        output_kernel_size = [ (3,3), (2,2) ] 

        activations = [ tf.keras.layers.PRelu() , 'linear' ]

        OUTPUT_LAYER_PARAMS = [ 
            { "filters":fs, "kernel_size":ks , "padding":"same", "activation":act, 'bias_regularizer':bias_regularizers[0]  } 
                for fs, ks, act in zip( output_filters, output_kernel_size, activations )
        ]
  
        self.params.update( {
            'model_name':"THST",
            'model_type_settings':model_type_settings,
    
            'encoder_params':ENCODER_PARAMS,
            'decoder_params':DECODER_PARAMS,
            'output_layer_params':OUTPUT_LAYER_PARAMS,
            'data_pipeline_params':DATA_PIPELINE_PARAMS,

            'rec_adam_params':REC_ADAM_PARAMS,
            'lookahead_params':LOOKAHEAD_PARAMS,
            'dropout':DROPOUT
            } )



class train_hparameters_ati(HParams):
    """ Parameters for testing """
    def __init__(self, **kwargs):
        self.lookback_target = kwargs.get('lookback_target',None)
        self.batch_size = kwargs.get("batch_size",None)
        self.dd = kwargs.get("data_dir") 
        
        # data formulation method
        self.custom_train_split_method = kwargs.get('ctsm') 
            
        if self.custom_train_split_method == "4ds_10years":
            self.four_year_idx_train = kwargs['fyi_train'] #index for training set

        kwargs.pop('batch_size')
        kwargs.pop('lookback_target')
        
        super( train_hparameters_ati, self).__init__(**kwargs)

    def _default_params(self, **kwargs):
        # region ------- Masking, Standardisation, temporal_data_size
        trainable = True
        MASK_FILL_VALUE = {
                                    "rain":0.0,
                                    "model_field":0.0 
        }
        vars_for_feature = ['unknown_local_param_137_128', 'unknown_local_param_133_128', 'air_temperature', 'geopotential', 'x_wind', 'y_wind' ]
        NORMALIZATION_SCALES = {
                                    "rain":4.69872+0.5,
                                    "model_fields": np.array([6.805,
                                                              0.001786,
                                                              5.458,
                                                              1678.2178,
                                                                5.107268,
                                                                4.764533]) }
        NORMALIZATION_SHIFT = {
                                    "rain":2.844,
                                    "model_fields": np.array([15.442,
                                                                0.003758,
                                                                274.833,
                                                                54309.66,
                                                                3.08158,
                                                                0.54810]) 
        }
        WINDOW_SHIFT = self.lookback_target
        BATCH_SIZE = self.batch_size
        # endregion

        NUM_PARALLEL_CALLS = tf.data.experimental.AUTOTUNE
        EPOCHS = 500
        CHECKPOINTS_TO_KEEP = 5

        # region ---- data formulation strategies
        target_start_date = np.datetime64('1950-01-01') + np.timedelta64(10592,'D')
        feature_start_date = np.datetime64('1970-01-01') + np.timedelta64(78888, 'h')

        if self.custom_train_split_method == "4ds_10years":
            #Dividing the dataset into 4 sets of 10 years for Experiement Varied Time Span
            li_start_dates = [ np.datetime64( '1979-01-01','D'), np.datetime64( '1989-01-01','D'), np.datetime64( '1999-01-01','D'), np.datetime64( '2009-01-01','D')   ]
            li_end_dates = [ np.datetime64( '1988-12-31','D'), np.datetime64( '1998-12-31','D'), np.datetime64( '2008-12-31','D'), np.datetime64( '2019-07-31','D') ]

            start_date = li_start_dates[ self.four_year_idx_train ]
            end_date = li_end_dates[ self.four_year_idx_train ]

            val_start_date =( pd.Timestamp(start_date) + pd.DateOffset( months = 8*12 ) ).to_numpy()  #8 year train set size, 2 year val set size
            val_end_date = end_date

            total_datums_target = np.timedelta64(end_date - start_date,'D')  / WINDOW_SHIFT  
            total_datums_target = total_datums_target.astype(int)

            TRAIN_SET_SIZE_ELEMENTS = ( np.timedelta64(val_start_date - start_date,'D')  // WINDOW_SHIFT   ).astype(int) 
            VAL_SET_SIZE_ELEMENTS = ( np.timedelta64( end_date - val_start_date,'D')  // WINDOW_SHIFT    ).astype(int)         
        
        elif type(self.custom_train_split_method) == str:
            # a string containing four dates seperated by underscores
            # The numbers correspond to trainstart_trainend_valstart_valend

            dates_str = self.custom_train_split_method.split("_")
            start_date = np.datetime64(dates_str[0],'D')
            train_end_date = (pd.Timestamp(dates_str[1]) - pd.DateOffset(seconds=1) ).to_numpy()
            val_start_date = np.datetime64(dates_str[1],'D')
            val_end_date = (pd.Timestamp(dates_str[2]) - pd.DateOffset(seconds=1) ).to_numpy()
            
            TRAIN_SET_SIZE_ELEMENTS = ( np.timedelta64(train_end_date - start_date,'D')  // WINDOW_SHIFT  ).astype(int) 
            VAL_SET_SIZE_ELEMENTS   = ( np.timedelta64(val_end_date - val_start_date,'D')  // WINDOW_SHIFT  ).astype(int)               
                
        else:
            raise ValueError("Invalid value passed for arg -cstm (custom_train_split_method) ")

        # endregion
        
        DATA_DIR = self.dd
        EARLY_STOPPING_PERIOD = 120
 
        self.params = {
            'batch_size':BATCH_SIZE,
            'epochs':EPOCHS,
            'early_stopping_period':EARLY_STOPPING_PERIOD,
            'trainable':trainable,
            'lookback_target':self.lookback_target,

            'train_batches': TRAIN_SET_SIZE_ELEMENTS//BATCH_SIZE, 
                #Note TRAIN_SET_SIZE_ELEMENTS refers to the number of sequences of days that are passed to TRU_NET as oppose dot every single day
            'val_batches': VAL_SET_SIZE_ELEMENTS//BATCH_SIZE,

            'checkpoints_to_keep':CHECKPOINTS_TO_KEEP,
            'reporting_freq':0.25,
            'num_parallel_calls':NUM_PARALLEL_CALLS,

            'train_monte_carlo_samples':1,
            'data_dir': DATA_DIR,
            
            'mask_fill_value':MASK_FILL_VALUE,
            'vars_for_feature':vars_for_feature,
            'normalization_scales' : NORMALIZATION_SCALES,
            'normalization_shift': NORMALIZATION_SHIFT,
            'window_shift': WINDOW_SHIFT,

            'start_date':start_date,
            'val_start_date':val_start_date,
            'val_end_date':val_end_date,

            'feature_start_date':feature_start_date,
            'target_start_date':target_start_date,
        }

class test_hparameters_ati(HParams):
    """ Parameters for testing """
    def __init__(self, **kwargs):
        self.lookback_target = kwargs['lookback_target']
        self.batch_size = kwargs.get("batch_size",2)
        
        self.dd = kwargs.get('data_dir')
        self.custom_test_split_method = kwargs.get('ctsm_test')
        
        if self.custom_test_split_method == "4ds_10years":
            self.four_year_idx_train = kwargs['fyi_train'] #index for training set
            self.four_year_idx_test = kwargs['fyi_test']
            assert self.four_year_idx_train != self.four_year_idx_test
        
        # kwargs.pop('batch_size')
        # kwargs.pop('lookback_target')
        #kwargs.pop('ctsm')
        super( test_hparameters_ati, self).__init__(**kwargs)
    
    def _default_params(self, **kwargs):
        
        # region --- data pipepline vars
        trainable = False

        # Standardisation and masking variables
        MASK_FILL_VALUE = {
                                    "rain":0.0,
                                    "model_field":0.0 
        }
        vars_for_feature = ['unknown_local_param_137_128', 'unknown_local_param_133_128', 'air_temperature', 'geopotential', 'x_wind', 'y_wind' ]
        NORMALIZATION_SCALES = {
                                    "rain":4.69872+0.5,
                                    "model_fields": np.array([6.805,
                                                              0.001786,
                                                              5.458,
                                                              1678.2178,
                                                                5.107268,
                                                                4.764533]) 
                                                #- unknown_local_param_137_128
                                                # - unknown_local_param_133_128,  
                                                # # - air_temperature, 
                                                # # - geopotential
                                                # - x_wind, 
                                                # # - y_wind
        }
        NORMALIZATION_SHIFT = {
                                    "rain":2.844,
                                    "model_fields": np.array([15.442,
                                                                0.003758,
                                                                274.833,
                                                                54309.66,
                                                                3.08158,
                                                                0.54810]) 
        }

        
        WINDOW_SHIFT = self.lookback_target # temporal shift for window to evaluate
        BATCH_SIZE = self.batch_size
        # endregion

        # region ---- Data Formaulation

        target_start_date = np.datetime64('1950-01-01') + np.timedelta64(10592,'D') #E-obs recording start from 1950
        feature_start_date = np.datetime64('1970-01-01') + np.timedelta64(78888, 'h') #ERA5 recording start from 1979
        
        tar_end_date =  target_start_date + np.timedelta64( 14822, 'D')
        feature_end_date  = np.datetime64( feature_start_date + np.timedelta64(59900, '6h'), 'D')
        
        start_date = feature_start_date if (feature_start_date > target_start_date) else target_start_date
        end_date = tar_end_date if (tar_end_date < feature_end_date) else feature_end_date     


        if self.custom_test_split_method == "4ds_10years":
            # Ease helper for the 4 dataset experiement: "Varied Time Span"
            # Allows user to pass two argumnets fyi_train and fyi_test, starting which model to use and which 
                # 10 year chunk to test on and which mod 

            li_start_dates = [ np.datetime64( '1979-01-01','D'), np.datetime64( '1989-01-01','D'), np.datetime64( '1999-01-01','D'), np.datetime64( '2009-01-01','D')   ]
            li_end_dates = [ np.datetime64( '1988-12-31','D'), np.datetime64( '1998-12-31','D'), np.datetime64( '2008-12-31','D'), np.datetime64( '2019-07-31','D') ]

            start_date = li_start_dates[self.four_year_idx_test]
            test_end_date = li_end_dates[self.four_year_idx_test]

            TEST_SET_SIZE_DAYS_TARGET = np.timedelta64( test_end_date - start_date, 'D' ).astype(int)
        
        elif type(self.custom_test_split_method) == str:
            # User must pass in two dates seperated by underscore such as 
                # 1985_2005, or 1985-02-04_2005_11_20
            dates_str = self.custom_test_split_method.split("_")
            start_date = np.datetime64(dates_str[0],'D')
            test_end_date = (pd.Timestamp(dates_str[1]) - pd.DateOffset(seconds=1) ).to_numpy()
            
            TEST_SET_SIZE_DAYS_TARGET = np.timedelta64( test_end_date - start_date, 'D' ).astype(int)

        # endregion

        # timesteps for saving predictions
        date_tss = pd.date_range( end=test_end_date, start=start_date, freq='D', normalize=True)
        timestamps = list ( (date_tss - pd.Timestamp("1970-01-01") ) // pd.Timedelta('1s') )

        DATA_DIR = self.dd

        self.params = {
            'batch_size':BATCH_SIZE,
            'trainable':trainable,
            
            'test_batches': TEST_SET_SIZE_DAYS_TARGET//(WINDOW_SHIFT*BATCH_SIZE),
                        
            'script_dir':None,
            'data_dir':DATA_DIR,
            
            'timestamps':timestamps,
            
            'mask_fill_value':MASK_FILL_VALUE,
            'vars_for_feature':vars_for_feature,
            'normalization_scales' : NORMALIZATION_SCALES,
            'normalization_shift': NORMALIZATION_SHIFT,
            'window_shift': WINDOW_SHIFT,

            'start_date':start_date,
            'test_end_date':test_end_date,

            'feature_start_date':feature_start_date,
            'target_start_date':target_start_date,
            #'train_test_size':self.tst,
        }

