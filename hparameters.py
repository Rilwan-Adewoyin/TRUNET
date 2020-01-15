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

    def __init__(self, **kwargs):
        self.input_dims = kwargs['input_dims']
        self.output_dims = kwargs['output_dims']
        super( model_deepsd_hparameters, self ).__init__(**kwargs)


    def _default_params(self):
        # region params
        input_dims = self.input_dims #[39, 88]
        output_dims = self.output_dims #[156,352]

        #TODO: (change filter sizes back to the ones used in the paper)

        CONV1_params = {    'filters':45,
                            'kernel_size': [5,5] , #TODO:use size from paper later
                            'activation':'relu',
                            'padding':'same',
                            'data_format':'channels_last',
                            'name':"Conv1" }

        conv2_kernel_size = np.ceil( np.ceil( np.array(output_dims)/np.array(input_dims) )*1.5 )  #This makes sure that each filter in conv2, sees at least two of the real non zero values. The zero values occur due to the upscaling
        CONV2_params = {    'filters':25,
                            'kernel_size':  conv2_kernel_size.astype(np.int32).tolist() , #TODO:use size from paper later
                            #each kernel covers 2 non original values from the upsampled tensor
                            'activation':'relu',
                            'padding':'same',
                            'data_format':'channels_last',
                            "name":"Conv2" }
       
        # CONV21_params = {   'filters':2,
        #                     'kernel_size':  conv2_kernel_size.astype(np.int32).tolist() , #TODO:use size from paper later
        #                     #each kernel covers 2 non original values from the upsampled tensor
        #                     'activation':'relu',
        #                     'padding':'same',
        #                     'data_format':'channels_last',
        #                     "name":"Conv21" }https://www.google.com/search?client=ubuntu&channel=fs&q=localhost%3A%2F6006&ie=utf-8&oe=utf-8
        

        CONV3_params = {
                            'filters':1,
                            'kernel_size':[3,3], #TODO:use size from paper later
                            'activation':'relu',
                            'padding':'same',
                            'data_format':'channels_last',
                            "name":"Conv3"  }

        var_model_type = "horseshoe_factorized"

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

        self.params = {
            'model_name':"DeepSD",
            'input_dims':input_dims,
            'output_dims':output_dims,
            'var_model_type':var_model_type,

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
            'conv3_inp_channels':conv3_inp_channels
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
        SEQ_LEN_FACTOR_REDUCTION = [30, 4 ] #This represents the rediction in seq_len when going from layer 1 to layer 2 and layer 2 to layer 3 in the encoder / decoder
        seq_len_for_highest_hierachy_level = 5
        NUM_OF_SPLITS = [ seq_len_for_highest_hierachy_level*SEQ_LEN_FACTOR_REDUCTION[1] , seq_len_for_highest_hierachy_level ]
        
        # end region
        
        # region Model Specific Data Generator Params
        
        DATA_PIPELINE_PARAMS = {
            'lookback': reduce( (lambda x,y: x*y ), SEQ_LEN_FACTOR_REDUCTION ) * seq_len_for_highest_hierachy_level
        }
        # end region


        # region --------------- ENCODER params -----------------
        encoder_layers = 3

        # region CLSTM params
        output_filters_enc = [50, 50, 50] #output filters for each convLSTM2D layer in the encoder
        kernel_size_enc = [ (4,4) , (4,4) , (4,4)]

        CLSTMs_params_enc = [
            {'filters':f , 'kernel_size':ks, 'padding':'same', 
                'return_sequences':True, 'dropout':DROPOUT, 'recurrent_dropout':DROPOUT }
             for f, ks in zip( output_filters_enc, kernel_size_enc)
        ]
        # endregion

        # region Attn Params
        """ These are the params for each attn layer in the encoder"""
        attn_layers = encoder_layers - 1
        key_depth = [0, 0] #This will be updated dynamically during the first iteration of the model
        attn_heads = [ 8, 8 ]

        ATTN_params_enc = [
            {'bias':None, 'total_key_depth': kd , 'total_value_depth':kd , 'output_depth': kd   ,
            'num_heads': nh , 'dropout_rate':DROPOUT, 'attention_type':"dot_product_relative_v2" }
            for kd, nh in zip( key_depth , attn_heads )
        ]
        # endregion

        ENCODER_PARAMS = {
            'encoder_layers': encoder_layers,
            'attn_layers': attn_layers,
            'CLSTMs_params' : CLSTMs_params_enc,
            'ATTN_params': ATTN_params_enc,
            'seq_len_factor_reduction': SEQ_LEN_FACTOR_REDUCTION,
            'num_of_splits': NUM_OF_SPLITS
            'dropout':DROPOUT
        }
        #endregion

        # region --------------- DECODER params -----------------
        decoder_layers = 2
        
        output_filters_dec = [50, 50] #output filters for each convLSTM2D layer in the encoder
        kernel_size_dec = [ (4,4) , (4,4) ]

        CLSTMs_params_dec = [
            {'filters':f , 'kernel_size':ks, 'padding':'same', 
                'return_sequences':True, 'dropout':0.05, 'gates_version':2 }
             for f, ks in zip( output_filters_dec, kernel_size_dec)
        ]

        DECODER_PARAMS = {
            'decoder_layers': decoder_layers,
            'CLSTMs_params' : CLSTMs_params_dec,
            'seq_len_factor_reduction': SEQ_LEN_FACTOR_REDUCTION,
            'num_of_splits': NUM_OF_SPLITS,
            'dropout':DROPOUT
        }
        # endregion

        # region --------------- OUTPUT_LAYER_PARAMS -----------------
        output_filters = [ 25, 1 ]
        output_kernel_size = [ (4,4), (5,5) ]

        OUTPUT_LAYER_PARAMS = [ 
            { "filters":fs, "kernel_size":ks ,  "padding":"same", "activation":'relu' } 
                for fs, ks in zip( output_filters, output_kernel_size )
         ]
        # endregion

        self.params = {
            'encoder_params':ENCODER_PARAMS,
            'decoder_params':DECODER_PARAMS,
            'output_layer_params':OUTPUT_LAYER_PARAMS
            'data_pipeline_params':DATA_PIPELINE_PARAMS
        }



class train_hparameters(HParams):
    def __init__(self, **kwargs):
        super( train_hparameters, self).__init__(**kwargs)

    def _default_params(self):
        # region default params 
        NUM_PARALLEL_CALLS = tf.data.experimental.AUTOTUNE,
        EPOCHS = 200
        CHECKPOINTS_TO_KEEP = 3
        MODEL_VERSION = 4
        TOTAL_DATUMS = 3650 #10 years worth of data apparently
        TRAIN_SET_SIZE_ELEMENTS = int(TOTAL_DATUMS*0.6)
        VAL_SET_SIZE_ELEMENTS = int(TOTAL_DATUMS*0.2)
        BATCH_SIZE = 15
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

            'model_version':MODEL_VERSION,
            
            'dataset_trainval_batch_reporting_freq':0.5,
            'num_parallel_calls':NUM_PARALLEL_CALLS,

            'gradients_clip_norm':50.0,

            'train_monte_carlo_samples':1,

            'data_dir': DATA_DIR,

            'bool_water_mask': BOOL_WATER_MASK

        }
        
class test_hparameters(HParams):
    def __init__(self, **kwargs):
        super( test_hparameters, self).__init__(**kwargs)
    
    def _default_params(self):
        NUM_PARALLEL_CALLS = tf.data.experimental.AUTOTUNE
        MODEL_VERSION = 4
        BATCH_SIZE = 15
        N_PREDS = 5

        MODEL_RECOVER_METHOD = 'checkpoint_epoch'
    
        TRAINING = 'test'
        TOTAL_DATUMS = 3650
        TEST_SET_SIZE_ELEMENTS = int( TOTAL_DATUMS * 0.2)
        STARTING_TEST_ELEMENT = TOTAL_DATUMS - TEST_SET_SIZE_ELEMENTS
        
        DATE_TSS = pd.date_range( end=datetime(2015,12,31), periods=TEST_SET_SIZE_ELEMENTS, freq='D',normalize=True).astype('int64').tolist()

        BOOL_WATER_MASK = pickle.load( open( "Images/water_mask_156_352.dat","rb" ) )


        self.params = {
            'batch_size':BATCH_SIZE,
            'starting_test_element':STARTING_TEST_ELEMENT,
            'test_set_size_elements': TEST_SET_SIZE_ELEMENTS,

            'model_version':MODEL_VERSION,

            'dataset_pred_batch_reporting_freq':0.25,
            'num_parallel_calls':NUM_PARALLEL_CALLS,

            'num_preds': N_PREDS,

            'model_recover_method':MODEL_RECOVER_METHOD,
            'training':TRAINING,

            'script_dir':None,

            'dates_tss':DATE_TSS,

            'bool_water_mask': BOOL_WATER_MASK

        }
        


    
    
