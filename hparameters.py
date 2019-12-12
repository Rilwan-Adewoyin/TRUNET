import numpy as np
import tensorflow as tf

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
        super( model_deepsd_hparameters, self ).__init__()
        
    
    def _default_params(self):
        # region params
        input_dims = [39, 88]
        output_dims = [156,352]

        #TODO: (change filter sizes back to the ones used in the paper)

        CONV1_params = {    'filters':4,
                            'kernel_size': [3,3] , #TODO:use size from paper later
                            'activation':'relu',
                            'padding':'same',
                            'data_format':'channels_last',
                            'name':"Conv1" }

        conv2_kernel_size = np.ceil( np.ceil( np.array(output_dims)/np.array(input_dims) )*1.5 )  #This makes sure that each filter in conv2, sees at least two of the real non zero values. The zero values occur due to the upscaling
        CONV2_params = {    'filters':4,
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
        #                     "name":"Conv21" }
        

        CONV3_params = {
                            'filters':1,
                            'kernel_size':[2,2], #TODO:use size from paper later
                            'activation':'relu',
                            'padding':'same',
                            'data_format':'channels_last',
                            "name":"Conv3"  }

        var_model_type = "reparam"

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


class train_hparameters(HParams):
    def __init__(self, **kwargs):
        super( train_hparameters, self).__init__()

    def _default_params(self):
        # region default params 
        NUM_PARALLEL_CALLS = tf.data.experimental.AUTOTUNE,
        CHECKPOINTS_TO_KEEP = 10
        MODEL_VERSION = 1
        EPOCHS = 2
        BATCH_SIZE = 10


        #endregion
        self.params = {
            'batch_size':BATCH_SIZE,
            'epochs':EPOCHS,

            'train_set_size_batches':64*100,
            'val_set_size_batches':64*20,

            'checkpoints_to_keep':CHECKPOINTS_TO_KEEP,

            'model_version':MODEL_VERSION,
            
            'dataset_trainval_batch_reporting_freq':0.1,
            'num_parallel_calls':NUM_PARALLEL_CALLS,

            'gradients_clip_norm':75.0,

            'train_monte_carlo_samples':1

        }
        

    
    
