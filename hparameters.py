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
    
    def __init__( self ,**kwargs ):  
        
        self._default_params()
        
        #TODO: remove this from code functionality
        if( kwargs != None):
            self.params.update( kwargs)
    
    def __call__(self):
        return self.params
    
    def _default_params(self):
        self.params = {}

class MParams(HParams):
    def __init__(self,**kwargs):
        
        #if kwargs['model_type_settings']['location'] == "region_grid":
        if type( kwargs['model_type_settings']['location'][:] ) in [list, data_structures.ListWrapper] and (kwargs.get('downscaled_input') == False ):
            self.regiongrid_param_adjustment()
        else:
            self.params = {}
        
        self._default_params(**kwargs)
        #super(MParams, self).__init__(**kwargs)
                 
        if type( kwargs['model_type_settings']['location'][:] ) in [list, data_structures.ListWrapper] and (kwargs.get('downscaled_input') == False ) :
            self.params['lookahead_params']['sync_period'] == int( np.prod( self.params['region_grid_params']['slides_v_h']  * min( [ self.params['lookahead_params']['sync_period'] // 2, 1] ) ) )

    def regiongrid_param_adjustment(self):
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

class model_deepsd_hparameters(MParams):
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

            CONV1_params = {    'filters':300, #512
                                'kernel_size': [7,7] , #[7,7] #TODO:use size from paper later
                                'activation':'relu',
                                'padding':'same',
                                'data_format':'channels_last',
                                'name':"Conv1" }
            CONV1_params.update(self.conv1_param_custom)
        if type(self.conv2_param_custom) == dict :
            conv2_kernel_size = np.ceil( np.ceil( np.array(output_dims)/np.array(input_dims) )*1.5 )  #This makes sure that each filter in conv2, sees at least two of the real non zero values. The zero values occur due to the upscaling
            CONV2_params = {    'filters':300, #512
                                'kernel_size':  conv2_kernel_size.astype(np.int32).tolist() , #TODO:use size from paper later
                                #each kernel covers 2 non original values from the upsampled tensor
                                'activation':'relu',
                                'padding':'same',
                                'data_format':'channels_last',
                                "name":"Conv2" }
            CONV2_params.update(self.conv2_param_custom)
                
        CONV3_params = {
                            'filters':1,
                            'kernel_size': [5,5], #[5,5] #TODO:use size from paper later
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
            "learning_rate":1e-5 , "warmup_proportion":0.5,
            "min_lr": 1e-6, "beta_1":0.99 , "beta_2": 0.99, "decay":0.005 }
        LOOKAHEAD_PARAMS = { "sync_period":5 , "slow_step_size":0.85}

        model_type_settings = {'stochastic':False ,'stochastic_f_pass':10,
                                'distr_type':"Normal", 'discrete_continuous':True,
                                'precip_threshold':0.5 , 'var_model_type':"flipout",
                                'model_version': "1" }
        
        self.params.update( {
            'model_name':"DeepSD",
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
            
            'rec_adam_params':REC_ADAM_PARAMS,
            'lookahead_params':LOOKAHEAD_PARAMS
        })

class model_THST_hparameters(MParams):

    def __init__(self, **kwargs):
        """  
        """
        self.dc = kwargs.get('model_type_settings',{}).get('discrete_continuous',False)
        self.stoc = kwargs.get('model_type_settings',{}).get('stochastic',False)
        self.di = kwargs.get('downscaled_input',False)
        super( model_THST_hparameters, self ).__init__(**kwargs)

    def _default_params( self, **kwargs ):
        # region learning/convergence params
        REC_ADAM_PARAMS = {
            "learning_rate":5e-3, "warmup_proportion":0.65,
            "min_lr":5e-4, "beta_1":0.50 , "beta_2":0.95,
            "amsgrad":True, "decay":0.007, "epsilon":0.0005 }

        DROPOUT = kwargs.get('dropout',0.0)
        LOOKAHEAD_PARAMS = { "sync_period":1, "slow_step_size":0.99 }
        # endregion
        
        #region Key Model Size Settings
        seq_len_for_highest_hierachy_level = 4   # 2  
        SEQ_LEN_FACTOR_REDUCTION = [4, 7]        # [ 4, 2 ]
            #This represents the rediction in seq_len when going from layer 1 to layer 2 and layer 2 to layer 3 in the encoder / decoder
            # 6hrs, 1Day, 1Week, 1Month
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

        # region CLSTM params
        if DROPOUT != 0.0 and self.stoc==True :
            #_filter = 112
            _filter = 72
        else:
            #_filter = 96
            _filter = 72
            #_filter = 112

        output_filters_enc     = [ _filter ]*(enc_layer_count-1)                     
        output_filters_enc     = output_filters_enc + output_filters_enc[-1:]   # the last two layers in the encoder must output the same number of channels
        
        if self.di == False:
            kernel_size_enc        = [ (4,4) ] * ( enc_layer_count )                
        else:
            kernel_size_enc        = [ (3,3) ] * ( enc_layer_count )                

        recurrent_regularizers = [ None ] * (enc_layer_count) 
        kernel_regularizers    = [ None ] * (enc_layer_count)
        bias_regularizers      = [ tf.keras.regularizers.l2(0.2) ] * (enc_layer_count) 
        recurrent_dropouts     = [ kwargs.get('rec_dropout',0.0) ]*(enc_layer_count)
        input_dropouts         = [ kwargs.get('inp_dropout',0.0) ]*(enc_layer_count)
        stateful               = True                       #True if testing on single location , false otherwise
        layer_norms            = lambda: None               #lambda: tf.keras.layers.LayerNormalization(axis=[-1], center=False, scale=False ) #lambda: None

        attn_layers_count = enc_layer_count - 1
        attn_heads = [ 8 ]*attn_layers_count                #[ 8 ]*attn_layers_count        #[5]  #NOTE:Must be a factor of h or w or c. h,w are dependent on model type so make it a multiple of c = 8
        
        if 'region_grid_params' in self.params.keys():
            kq_downscale_stride = [1, 4, 4]                 #[1, 8, 8] 
            kq_downscale_kernelshape = kq_downscale_stride

            #This keeps the hidden representations equal in size to the incoming tensors
            val_depth = [ int( np.prod( self.params['region_grid_params']['outer_box_dims'] ) * output_filters_enc[idx] * 2 ) for idx in range(attn_layers_count)  ]
                                    
            if kq_downscale_stride == [1,8,8]:
                key_depth = [72]*attn_layers_count
                #key_depth = [320]*attn_layers_count
            elif kq_downscale_stride == [1,4,4]:
                #key_depth = [128]*attn_layers_count
                key_depth = [72]*attn_layers_count

        #elif 'downscale_input_factor' in kwargs:
        elif self.di == True:

            _dims = [18 , 18 ]
            kq_downscale_stride = [1, _dims[0]//4, _dims[1]//4 ]
            kq_downscale_kernelshape = kq_downscale_stride

            val_depth = [ int( np.prod(_dims )*output_filters_enc[idx]*2) for idx in range(attn_layers_count)  ]
            key_depth = [72]*attn_layers_count
            

        else:
            kq_downscale_stride = [1, 13, 13]
            kq_downscale_kernelshape = [1, 13, 13]

            #This keeps the hidden representations equal in size to the incoming tensors
            key_depth = [ 72 ]*attn_layers_count
            val_depth = [ int(100*140*output_filters_enc[idx]*2) for idx in range(attn_layers_count)  ]
                
            #The keydepth for any given layer will be equal to (h*w*c/avg_pool_strideh*avg_pool_stridew)
                # where h,w = 100,140 and c is from the output_filters_enc from the layer below
            

        ATTN_LAYERS_NUM_OF_SPLITS = list(reversed((np.cumprod( list( reversed(SEQ_LEN_FACTOR_REDUCTION[1:] + [1] ) ) ) *seq_len_for_highest_hierachy_level ).tolist())) 
            #Each encoder layer receives a seq of 3D tensors from layer below. NUM_OF_SPLITS codes in how many chunks to devide the incoming data. NOTE: This is defined only for Encoder-Attn layers

        ATTN_params_enc = [
            # {'bias':None, 'total_key_depth': kd  ,'total_value_depth':vd, 'output_depth': vd   ,
            # 'num_heads': nh , 'dropout_rate':DROPOUT, 'max_relative_position':None,
            # "transform_value_antecedent":False ,  "transform_output":False,
            # 'implementation':1  } 
            
            {'bias':None, 'total_key_depth': kd  ,'total_value_depth':vd, 'output_depth': vd   ,
            'num_heads': nh , 'dropout_rate':DROPOUT, 'max_relative_position':None,
            "transform_value_antecedent":True,  "transform_output":True, 
            'implementation':1, "value_conv":{ "filters":int(output_filters_enc[idx] * 2), 'kernel_size':[3,3] ,'use_bias':True, "activation":'relu', 'name':"v", 'bias_regularizer':tf.keras.regularizers.l2(0.2), 'padding':'same' },
            "output_conv":{ "filters":int(output_filters_enc[idx] * 2), 'kernel_size':[3,3] ,'use_bias':True, "activation":'relu', 'name':"outp", 'bias_regularizer':tf.keras.regularizers.l2(0.2),'padding':'same' }
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
                'return_sequences':True, 'dropout':ido, 'recurrent_dropout':rd,
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
            'dropout':DROPOUT
        }
        #endregion

        # region --------------- DECODER params -----------------
        decoder_layer_count = enc_layer_count-2
        
        output_filters_dec = output_filters_enc[:1] + output_filters_enc[ :decoder_layer_count ]  #The first part list needs to be changed, only works when all convlstm layers have the same number of filters
        kernel_size_dec = kernel_size_enc[ 1:1+decoder_layer_count  ]                             # This is written in the correct order
        
            #Each decoder layer sends in values into the layer below. 
        CGRUs_params_dec = [
            {'filters':f , 'kernel_size':ks, 'padding':'same', 
                'return_sequences':True, 'dropout':ido,
                'recurrent_dropout':rdo, 
                'kernel_regularizer':kr,
                'recurrent_regularizer': rr,
                'bias_regularizer':br,
                'stateful':stateful,
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
            'dropout':DROPOUT
        }
        # endregion

        # region --------------- OUTPUT_LAYER_PARAMS -----------------
        if self.di:
            _upscale_target = [100,140]
            _input_dims = [18, 18]
            _strides =( np.floor_divide(_upscale_target,_input_dims).astype( np.int32) ).tolist()  #( np.ceil( np.array(_upscale_target)/np.array(_input_dims)).astype(np.int32)  ).tolist()
            _kernel_size = (np.array(_strides)*2+1 ).tolist()
            
            # _output_padding = (np.array(_input_dims)*(np.array(_strides)) )  #This provides a methodology to understand the outer_padding number, https://stackoverflow.com/a/54891027/7497927 
            # _output_padding = (np.array(_upscale_target) - _output_padding )
            # _output_padding = (_output_padding/2).astype(np.int32).tolist()
            _output_padding = np.array( [4,6] )
            #_output_padding = ( (np.array(_kernel_size)-1)//2 )

            conv_upscale_params = {'filters': int(  8*(((output_filters_dec[-1]*2)/4)//8)), 'kernel_size':_kernel_size, 'strides':_strides, 'output_padding': _output_padding,
                                    'activation':'relu','padding':'valid','bias_regularizer':tf.keras.regularizers.l2(0.2)  }  

            #Note: Stride larger than filter size may lead to middle areas being assigned a zero value
            self.params.update( { 'conv_upscale_params': conv_upscale_params } )

        output_filters = [  int(  8*(((output_filters_dec[-1]*2)/3)//8)), 1 ]  #[ 2, 1 ]   # [ 8, 1 ]
        output_kernel_size = [ (3,3), (3,3) ] 
        activations = ['relu','linear']

        OUTPUT_LAYER_PARAMS = [ 
            { "filters":fs, "kernel_size":ks , "padding":"same", "activation":act, 'bias_regularizer':tf.keras.regularizers.l2(0.2)  } 
                for fs, ks, act in zip( output_filters, output_kernel_size, activations )
         ]
        # endregion

        _mts = {  }
        model_type_settings = kwargs.get('model_type_settings',_mts)

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

class model_SimpleGRU_hparameters(MParams):

    def __init__(self, **kwargs):
        super(model_SimpleGRU_hparameters, self).__init__(**kwargs)
    
    def _default_params(self, **kwargs):
        #model
        dropout = kwargs.get('dropout',0.0)

        input_dropout = kwargs.get('inp_dropout',0.0)
        recurrent_dropout = kwargs.get('rec_dropout',0.0)
        layer_count = 3

        if dropout == 0.0:
            units = 160
        else:
            #units = int(160*2.0)
            units = int(160*1.4)
        li_units = [units]*layer_count
        
        li_rs =     [True]*layer_count
        ln = [ tf.keras.layers.LayerNormalization(axis=-1) for _idx in range(layer_count) ]
        for _ln in ln: _ln._dtype = 'float32'

        LAYER_PARAMS = [
            {'units': un, 'dropout':input_dropout, 'recurrent_dropout':recurrent_dropout,
                'return_sequences':rs, 'stateful':True,
                'kernel_regularizer': None,
                'recurrent_regularizer': None,
                'bias_regularizer':tf.keras.regularizers.l2(0.2),
                'layer_norm':_ln }
                for un, rs,_ln in zip(li_units, li_rs, ln)
        ]

        # LAYER_PARAMS = [
        #     {'units': un, 'dropout':0.2, 'recurrent_dropout':0.25,
        #         'return_sequences':rs, 'stateful':True,
        #         'kernel_regularizer': None,
        #         'recurrent_regularizer': None,
        #         'bias_regularizer':tf.keras.regularizers.l2(0.2) }
        #         for un, rs in zip(li_units, li_rs)
        # ]

        dense1_layer_params = { 'units':80, 'activation':'relu', 'bias_regularizer':tf.keras.regularizers.l2(0.2) }
        output_dense_layer_params = {'units':1, 'activation':'linear','bias_regularizer':tf.keras.regularizers.l2(0.2) }

        #data pipeline
        target_to_feature_time_ratio = 4
        lookback_feature = 30*target_to_feature_time_ratio   #30     
        DATA_PIPELINE_PARAMS = {
            'lookback_feature':lookback_feature,
            'lookback_target': int(lookback_feature/target_to_feature_time_ratio),
            'target_to_feature_time_ratio' :  target_to_feature_time_ratio
        }

        #training proc

        REC_ADAM_PARAMS = {
            "learning_rate":2e-3, "warmup_proportion":0.75,
            "min_lr":1e-3, "beta_1":0.3, "beta_2":0.95, "decay":0.005,
            "amsgrad":True, "epsilon":5e-3
            } #for multile optimizers asymettric 

        LOOKAHEAD_PARAMS = { "sync_period":1 , "slow_step_size":0.99 }

        _mts = { }
        model_type_settings = kwargs.get('model_type_settings',_mts)

        self.params.update({
            'model_name':'SimpleGRU',
            'layer_count': layer_count,
            'layer_params': LAYER_PARAMS,
            'dense1_layer_params':dense1_layer_params,
            'output_dense_layer_params':output_dense_layer_params,
            'dropout':dropout,
            
            'data_pipeline_params': DATA_PIPELINE_PARAMS,
            'model_type_settings': model_type_settings,

            'rec_adam_params':REC_ADAM_PARAMS,
            'lookahead_params':LOOKAHEAD_PARAMS
        })

class model_SimpleConvGRU_hparamaters(MParams):

    def __init__(self, **kwargs):
        self.dc = kwargs.get('model_type_settings',{}).get('discrete_continuous',False)
        self.stoc = kwargs.get('model_type_settings',{}).get('stochastic',False)
        self.di = kwargs.get('downscaled_input',False)
        super(model_SimpleConvGRU_hparamaters, self).__init__(**kwargs)
    
    def _default_params(self,**kwargs):
        #Other
        dropout = kwargs.get('dropout',0.0)

        #region ConvLayers
        layer_count = 3 #TODO: Shi uses 2 layers
        if dropout != 0.0 and self.stoc==True :            
            _filter = int(80*1.4)
            #_filter = 80
        else:
            _filter = 80

        filters = [_filter]*layer_count #[128]*layer_count #Shi Precip nowcasting used
        kernel_sizes = [[4,4]]*layer_count
        paddings = ['same']*layer_count
        return_sequences = [True]*layer_count
        input_dropout = [kwargs.get('inp_dropout',0.0) ]*layer_count #[0.0]*layer_count
        recurrent_dropout = [ kwargs.get('rec_dropout',0.0)]*layer_count #[0.0]*layer_count

        # if self.params['model_type_settings']['location'] == "region_grid":
        #     _st = False
        # else:
        #     _st = True
        _st = True #Note; this will only work when doing 2d on a specific region

        ConvGRU_layer_params = [ { 'filters':fs, 'kernel_size':ks , 'padding': ps,
                                'return_sequences':rs, "dropout": dp , "recurrent_dropout":rdp,
                                'kernel_regularizer': None,
                                'recurrent_regularizer': None,
                                'bias_regularizer':tf.keras.regularizers.l2(0.2),
                                'layer_norm': None, #tf.keras.layers.LayerNormalization(axis=[-1]),
                                'implementation':1, 'stateful':_st  }
                                for fs,ks,ps,rs,dp,rdp in zip(filters, kernel_sizes, paddings, return_sequences, input_dropout, recurrent_dropout)  ]

        conv1_layer_params = {'filters': int(  8*(((filters[0]*2)/3)//8)) , 'kernel_size':[3,3], 'activation':'relu','padding':'same','bias_regularizer':tf.keras.regularizers.l2(0.2) }  

        if self.di:
            _upscale_target = [100,140]
            _input_dims = [18, 18]
            _strides =( np.floor_divide(_upscale_target,_input_dims).astype( np.int32) ).tolist()  #( np.ceil( np.array(_upscale_target)/np.array(_input_dims)).astype(np.int32)  ).tolist()
            _kernel_size = (np.array(_strides)*2+1 ).tolist()
      
            _output_padding = np.array( [4,6] )
            
            conv2_layer_params = {'filters': int(  8*(((filters[-1]*2)/4)//8)), 'kernel_size':_kernel_size, 'strides':_strides, 'output_padding': _output_padding,
                                    'activation':'relu','padding':'valid','bias_regularizer':tf.keras.regularizers.l2(0.2)  }  

            #Note: Stride larger than filter size may lead to middle areas being assigned a zero value
            self.params.update( { 'conv2_layer_params': conv2_layer_params } )

        outpconv_layer_params = {'filters':1, 'kernel_size':[3,3], 'activation':'linear','padding':'same','bias_regularizer':tf.keras.regularizers.l2(0.2) }


        #endregion

        #region data pipeline
        target_to_feature_time_ratio = 4
        lookback_feature = 30*target_to_feature_time_ratio  #TODO: Try with longer sequence if it fits into memory       
        DATA_PIPELINE_PARAMS = {
            'lookback_feature':lookback_feature,
            'lookback_target': int(lookback_feature/target_to_feature_time_ratio),
            'target_to_feature_time_ratio' :  target_to_feature_time_ratio
        }
        # endregion
        #region training proc
        REC_ADAM_PARAMS = {
            "learning_rate":7e-3 , "warmup_proportion":0.75,
            "min_lr":1e-3, "beta_1":0.35, "beta_2":0.85, "decay":0.006, "amsgrad":True,
            'epsilon':0.005
            }

        LOOKAHEAD_PARAMS = { "sync_period":1 , "slow_step_size":0.99 }
        # endregion
        _mts = {}
        model_type_settings = kwargs.get('model_type_settings',_mts)


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
            'lookahead_params':LOOKAHEAD_PARAMS
        })

class model_SimpleDense_hparameters(MParams):

    def __init__(self, **kwargs):
        super(model_SimpleDense_hparameters, self).__init__(**kwargs)
    
    def _default_params(self):
        
        #data pipeline
        target_to_feature_time_ratio = 4
        lookback_feature = 4*target_to_feature_time_ratio        
        DATA_PIPELINE_PARAMS = {
            'lookback_feature':lookback_feature,
            'lookback_target': int(lookback_feature/target_to_feature_time_ratio),
            'target_to_feature_time_ratio' :  target_to_feature_time_ratio
        }

        #training proc
        REC_ADAM_PARAMS = {
            "learning_rate":1e-3 , "warmup_proportion":0.6,
            "min_lr":1e-4, "beta_1":0.99, "beta_2":0.99,"decay":0.005
            }

        LOOKAHEAD_PARAMS = { "sync_period":1 , "slow_step_size":0.999 }

        model_type_settings = { }

        self.params.update({
            'model_name':'SimpleDense',
            
            'data_pipeline_params': DATA_PIPELINE_PARAMS,
            'model_type_settings': model_type_settings,

            'rec_adam_params':REC_ADAM_PARAMS,
            'lookahead_params':LOOKAHEAD_PARAMS
        })

class train_hparameters(HParams):
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get("batch_size",None)
        kwargs.pop('batch_size')
        super( train_hparameters, self).__init__(**kwargs)

    def _default_params(self):
        # region default params 
        NUM_PARALLEL_CALLS = tf.data.experimental.AUTOTUNE
        EPOCHS = 342 #equivalent to Vandal
        CHECKPOINTS_TO_KEEP = 50
        CHECKPOINTS_TO_KEEP_EPOCH = 5
        CHECKPOINTS_TO_KEEP_BATCH = 5

        start_date = np.datetime64('1981-01-01')
        end_date = np.datetime64('2015-12-31')
        TOTAL_DATUMS = np.timedelta64( end_date-start_date, 'D').astype(int)

        #need to use this ration, 0.73529, for training
        TRAIN_SET_SIZE_ELEMENTS = int(TOTAL_DATUMS*0.53529411764)
        VAL_SET_SIZE_ELEMENTS = int(TOTAL_DATUMS*0.20)
        BATCH_SIZE = self.batch_size
        DATA_DIR = "./Data"
        EARLY_STOPPING_PERIOD = 5
        BOOL_WATER_MASK = pickle.load( open( "Images/water_mask_156_352.dat","rb" ) )

        #endregion

        self.params = {
            'batch_size':BATCH_SIZE,
            'epochs':EPOCHS,
            'total_datums':TOTAL_DATUMS,
            'early_stopping_period':EARLY_STOPPING_PERIOD,
            'trainable':True,

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
        self.batch_size = kwargs.get("batch_size",None)
        kwargs.pop('batch_size')
        super( test_hparameters, self).__init__(**kwargs)
    
    def _default_params(self):
        NUM_PARALLEL_CALLS = tf.data.experimental.AUTOTUNE
        BATCH_SIZE = self.batch_size

        MODEL_RECOVER_METHOD = 'checkpoint_epoch'
    
        trainable = False
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

            'dataset_pred_batch_reporting_freq':0.05,
            'num_parallel_calls':NUM_PARALLEL_CALLS,

            'model_recover_method':MODEL_RECOVER_METHOD,
            'trainable':trainable,

            'script_dir':None,

            'epochs':EPOCHS,

            'bool_water_mask': BOOL_WATER_MASK
        }

class train_hparameters_ati(HParams):
    def __init__(self, **kwargs):
        self.lookback_target = kwargs.get('lookback_target',None)
        self.batch_size = kwargs.get("batch_size",None)
        self.strided_dataset_count = kwargs.get("strided_dataset_count", 1)
        self.di = kwargs.get("downscaled_input")
        self.dd = kwargs.get("data_dir") 
        kwargs.pop('batch_size')
        kwargs.pop('lookback_target')
        kwargs.pop('strided_dataset_count')
        super( train_hparameters_ati, self).__init__(**kwargs)

    def _default_params(self):
        # region -------data pipepline vars
        trainable = True
        MASK_FILL_VALUE_v1 = {
                                    "rain":0.0,
                                    "model_field":0.0 
        }

        NORMALIZATION_SCALES_v1 = {
                                    "rain":4.69872+0.5,
                                    "model_fields": np.array([6.805,
                                                              0.001786,
                                                              5.458,
                                                              1678.2178,
                                                                5.107268,
                                                                4.764533]) 
                                                #- unknown_local_param_137_128
                                                # - unknown_local_param_133_128,  # - air_temperature, # - geopotential
                                                # - x_wind, # - y_wind
        }
        NORMALIZATION_SHIFT_v1 = {
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
        CHECKPOINTS_TO_KEEP = 3
        CHECKPOINTS_TO_KEEP_EPOCH = 5
        CHECKPOINTS_TO_KEEP_BATCH = 5

        # region ---- data information

        if self.di == False:
            target_start_date = np.datetime64('1950-01-01') + np.timedelta64(10592,'D')
            feature_start_date = np.datetime64('1970-01-01') + np.timedelta64(78888, 'h')
        
            tar_end_date = target_start_date + np.timedelta64( 14822, 'D')
            feature_end_date  = np.datetime64( feature_start_date + np.timedelta64(16072, '6h'), 'D')
        
        elif self.di == True:
            target_start_date = np.datetime64('1950-01-01') + np.timedelta64(10592,'D')
            feature_start_date = np.datetime64('1970-01-01') + np.timedelta64(78888, 'h')
        
            tar_end_date =  target_start_date + np.timedelta64( 14822, 'D')
            feature_end_date  = np.datetime64( feature_start_date + np.timedelta64(59900, '6h'), 'D')


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
        
        #DATA_DIR = "./Data/Rain_Data_Nov19" 
        DATA_DIR = self.dd

        EARLY_STOPPING_PERIOD = 30
 
        self.params = {
            'batch_size':BATCH_SIZE,
            'epochs':EPOCHS,
            'early_stopping_period':EARLY_STOPPING_PERIOD,
            'trainable':trainable,
            'lookback_target':self.lookback_target,

            'strided_dataset_count': self.strided_dataset_count,
            'train_set_size_elements_b4_sdc_multlocation': TRAIN_SET_SIZE_ELEMENTS,
            'val_set_size_elements_b4_sdc_multlocation':VAL_SET_SIZE_ELEMENTS,
            # 'train_set_size_batches': (TRAIN_SET_SIZE_ELEMENTS//BATCH_SIZE) #*self.strided_dataset_count - ( self.strided_dataset_count - 1) ,
            # 'val_set_size_batches':(VAL_SET_SIZE_ELEMENTS//BATCH_SIZE) #*self.strided_dataset_count - (self.strided_dataset_count - 1),

            'checkpoints_to_keep':CHECKPOINTS_TO_KEEP,
            'checkpoints_to_keep_epoch':CHECKPOINTS_TO_KEEP_EPOCH,
            'checkpoints_to_keep_batch':CHECKPOINTS_TO_KEEP_BATCH,

            'dataset_trainval_batch_reporting_freq':0.1,
            'num_parallel_calls':NUM_PARALLEL_CALLS,

            'train_monte_carlo_samples':1,
            'data_dir': DATA_DIR,

            'mask_fill_value':MASK_FILL_VALUE_v1,
            'normalization_scales' : NORMALIZATION_SCALES_v1,
            'normalization_shift': NORMALIZATION_SHIFT_v1,
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
        self.batch_size = kwargs.get("batch_size",2)
        kwargs.pop('batch_size')
        #kwargs.pop('lookback_target')
        self.di = kwargs.get('downscaled_input')
        self.dd = kwargs.get('data_dir')
        

        super( test_hparameters_ati, self).__init__(**kwargs)
    
    def _default_params(self):
        pass
        # region -------data pipepline vars
        trainable = False

        MASK_FILL_VALUE_v1 = {
                                    "rain":0.0,
                                    "model_field":0.0 
        }

        NORMALIZATION_SCALES_v1 = {
                                    "rain":4.69872+0.5,
                                    "model_fields": np.array([6.805,
                                                              0.001786,
                                                              5.458,
                                                              1678.2178,
                                                                5.107268,
                                                                4.764533]) 
                                                #- unknown_local_param_137_128
                                                # - unknown_local_param_133_128,  # - air_temperature, # - geopotential
                                                # - x_wind, # - y_wind
        }
        NORMALIZATION_SHIFT_v1 = {
                                    "rain":2.844,
                                    "model_fields": np.array([15.442,
                                                                0.003758,
                                                                274.833,
                                                                54309.66,
                                                                3.08158,
                                                                0.54810]) 
        }

        WINDOW_SHIFT = self.lookback_target
        NUM_PARALLEL_CALLS = tf.data.experimental.AUTOTUNE
        BATCH_SIZE = self.batch_size
        N_PREDS = 25
        MODEL_RECOVER_METHOD = 'checkpoint_epoch'
        # endregion

        if self.di == False:
            target_start_date = np.datetime64('1950-01-01') + np.timedelta64(10592,'D')
            feature_start_date = np.datetime64('1970-01-01') + np.timedelta64(78888, 'h')
            
            tar_end_date=  target_start_date + np.timedelta64( 14822, 'D')
            feature_end_date  = np.datetime64( feature_start_date + np.timedelta64(16072, '6h'), 'D')
        
        elif self.di == True:
            target_start_date = np.datetime64('1950-01-01') + np.timedelta64(10592,'D')
            feature_start_date = np.datetime64('1970-01-01') + np.timedelta64(78888, 'h')
        
            tar_end_date =  target_start_date + np.timedelta64( 14822, 'D')
            feature_end_date  = np.datetime64( feature_start_date + np.timedelta64(59900, '6h'), 'D')

        if feature_start_date > target_start_date:
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
        TOTAL_DATUMS_TARGET = np.timedelta64(end_date - train_start_date,'D')   #Think of better way to get the np.product info from model_params to train params
        TOTAL_DATUMS_TARGET = TOTAL_DATUMS_TARGET.astype(int)

        test_start_date = train_start_date + (end_date - train_start_date)*0.8
        test_end_date = end_date

        TEST_SET_SIZE_DATUMS_TARGET = int( TOTAL_DATUMS_TARGET * 0.2)
        ## endregion

        date_tss = pd.date_range( end=test_end_date, start=test_start_date, freq='D',normalize=True)
        EPOCHS = list ( (date_tss - pd.Timestamp("1970-01-01") ) // pd.Timedelta('1s') )

        DATA_DIR = self.dd

        self.params = {
            'batch_size':BATCH_SIZE,
            'trainable':trainable,
            
            'total_datums':TOTAL_DATUMS_TARGET,
            'test_set_size_elements':TEST_SET_SIZE_DATUMS_TARGET,
            'num_preds':N_PREDS,
            'dataset_pred_batch_reporting_freq':0.01,

            'model_recover_method':MODEL_RECOVER_METHOD,
            'script_dir':None,
            'data_dir':DATA_DIR,
            
            'epochs':EPOCHS,
            
            'mask_fill_value':MASK_FILL_VALUE_v1,
            'normalization_scales' : NORMALIZATION_SCALES_v1,
            'normalization_shift': NORMALIZATION_SHIFT_v1,
            'window_shift': WINDOW_SHIFT,

            'test_start_date':test_start_date,
            'test_end_date':test_end_date,

            'feature_start_date':feature_start_date,
            'target_start_date':target_start_date,
            'feature_start_end':feature_end_date
        }

