import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed
import layers
import layers_ConvLSTM2D
import layers_ConvGRU2D
import layers_gru
import copy


def model_loader(train_params,model_params ):
    
    if type(model_params)==list:
        model_name = model_params[0]['model_name']
    else:
        model_name = model_params['model_name']
    
    if(model_name=="THST"):
        model =THST(train_params, model_params)
    
    elif(model_name=="DeepSD"):
        model = SuperResolutionModel( train_params, model_params)
    
    elif(model_name=="SimpleLSTM"):
        model = SimpleLSTM(train_params, model_params)

    elif(model_name=="SimpleDense"):
        model = SimpleDense(train_params, model_params)
        
    elif(model_name == "SimpleConvLSTM"):
        model = SimpleConvLSTM(train_params, model_params)

    elif(model_name=="SimpleConvGRU"):
        model = SimpleConvGRU(train_params, model_params)
    
    return model

class SuperResolutionModel( tf.keras.Model ):
    def __init__(self, train_params, model_params ):
        super(SuperResolutionModel, self).__init__()
        self.train_params = train_params
        self.model_params = model_params
        self.SRCNN_1 = layers.SRCNN( train_params, model_params )
        self.float32_output = tf.keras.layers.Activation('linear',dtype='float32')
        # self.SRCNN_2 = layers.SRCNN( train_params, model_params[1] )
        #self.SRCNN_3 = layers.SRCNN( train_params, model_params[2] )
    
    #@tf.function
    def call(self, inputs, training=False):
        x = self.SRCNN_1(inputs, training)
        x = self.float32_output(x)
        # x = self.SRCNN_2( x, pred )
        return x
    
    def predict( self, inputs, n_preds, pred=True):
        """
            Produces N predictions for each given input
        """
        preds = []
        #for count in tf.range(n_preds):
        for count in range(n_preds):
            pred = self.call( inputs, training=pred ) #shape ( batch_size, output_h, output_w, 1 )
            preds.append( pred )
        
        return preds

class THST(tf.keras.Model):
    """
    Temporal Hierarchical Spatial Transformer 
    """
    def __init__(self, train_params, model_params):
        super(THST, self).__init__()

        if model_params['model_type_settings']['location'] == "region_grid":
            h_w = model_params['region_grid_params']['outer_box_dims']
        else:
            h_w = [100,140]
        #TODO: in bidirectional layers explicity add the go_backwards line and second LSTM / GRU layer
        self.encoder = layers.THST_Encoder( train_params, model_params['encoder_params'], h_w )
        self.decoder = layers.THST_Decoder( train_params, model_params['decoder_params'], h_w )
        self.output_layer = layers.THST_OutputLayer( train_params, model_params['output_layer_params'], model_params['model_type_settings']  )

        self.outp_layer2 = layers.OutputReluFloat32(train_params)
        # self.output_activation = layers.CustomRelu_maker(train_params)
        # self.float32_output = tf.keras.layers.Activation('linear', dtype='float32')

    #@tf.function
    def call(self, _input, tape=None, training=False):
        
        #old
        # hidden_states_2_enc, hidden_states_3_enc, hidden_4_enc, hidden_5_enc = self.encoder( _input, training )
        # hidden_states_dec = self.decoder( hidden_states_2_enc, hidden_states_3_enc, hidden_4_enc, hidden_5_enc, training )
        # output = self.output_layer(hidden_states_dec, training)
        # output = self.float32_output(output)
        #with tf.device('/GPU:0'):
        hs_list_enc = self.encoder(_input, training=training)
        #with tf.device('/GPU:0'):#: with tf.device('/GPU:1'):
        hs_dec = self.decoder(hs_list_enc, training=training)
        
        output = self.output_layer(hs_dec, training)
        output = self.outp_layer2(output)

        return output

    def predict( self, inputs, n_preds, training=True):
        """
            Produces N predictions for each given input
        """
        preds = []
        for count in tf.range(n_preds):
            pred = self.call( inputs, training=True ) #shape ( batch_size, output_h, output_w, 1 ) or # (pred_count, bs, seq_len, h, w)
            preds.append( pred )
        return preds

class SimpleLSTM(tf.keras.Model):
    
    def __init__(self, train_params, model_params):
        super(SimpleLSTM, self).__init__()

        self.model_params = model_params

        if model_params['model_type_settings']['model_version'] == "24":
            self.LSTM_layers=[]
            for idx in range( model_params['layer_count'] ):
                _params=  copy.deepcopy(model_params['layer_params'][idx] )
                _params.pop('return_sequences')
                _params.pop('stateful')
                
                peephole_lstm_cell = tf.keras.experimental.PeepholeLSTMCell(**_params)
                layer = tf.keras.layers.Bidirectional( tf.keras.layers.RNN( peephole_lstm_cell, return_sequences=True, stateful=True ), merge_mode='concat' )
                self.LSTM_layers.append(layer) 
        elif model_params['model_type_settings']['model_version'] in ["25","27","28","30","33","35"]:
            self.LSTM_layers = [ tf.keras.layers.Bidirectional( tf.keras.layers.GRU( **model_params['layer_params'][idx] ), merge_mode='concat' ) for idx in range( model_params['layer_count'] ) ] 
        elif int(model_params['model_type_settings']['model_version'])>= 44 or model_params['model_type_settings']['model_version']  in ["34","36"]:
            self.LSTM_layers = [ tf.keras.layers.Bidirectional( layers_gru.GRU_LN_v2( **model_params['layer_params'][idx]), merge_mode='concat' ) for idx in range(model_params['layer_count'] ) ]
        else:
            self.LSTM_layers = [ tf.keras.layers.Bidirectional( tf.keras.layers.LSTM( **model_params['layer_params'][idx] ), merge_mode='concat' ) for idx in range( model_params['layer_count'] ) ]

        self.dense1 =  TimeDistributed( tf.keras.layers.Dense(model_params['dense1_layer_params'] ) )
        self.output_dense = TimeDistributed( tf.keras.layers.Dense( model_params['output_dense_layer_params']) )
        
        if model_params['model_type_settings']['model_version'] in ["23","27"]:
            self.output_activation = layers.LeakyRelu_mkr(train_params)
        else:
            #self.output_activation = layers.CustomRelu_maker(train_params, dtype='float16')
            self.output_activation = layers.CustomRelu_maker(train_params, dtype='float32')
            #self.output_activation._dtype = 'float32'


        self.float32_output = tf.keras.layers.Activation('linear', dtype='float32')
        self.do=tf.keras.layers.Dropout(0.05 )#, noise_shape=None, seed=None, **kwargs

        self.new_shape = tf.TensorShape( [train_params['batch_size'], model_params['data_pipeline_params']['lookback_target'], int(6*4)] )

    @tf.function
    def call(self, _input, training=True):
        #_input shape (bs, seq_len, c)
        #shape = _input.shape

        x = tf.reshape( _input, self.new_shape )
        #x =  self.dense0(self.do(x,training=training) )

        for idx in range(self.model_params['layer_count']):
            if idx==0:
                x0 = self.LSTM_layers[idx](inputs=x,training=training )
                x = x0
            else:
                x = x + self.LSTM_layers[idx](inputs=x,training=training )
        
        #x = self.dense1(self.do( (x + x0)/2,training=training) )
        x = self.dense1( self.do( tf.concat( [x,x0] , axis=-1 ), training=training ) ) #new for model 36,37
        outp = self.output_dense( x )
        #outp = self.output_activation(outp)
        outp = self.float32_output(outp)
        outp = self.output_activation(outp)
        return outp
        
    def predict( self, inputs, n_preds, training=True):
            """
                Produces N predictions for each given input
            """
            preds = []
            for count in tf.range(n_preds):
                pred = self.call( inputs, training=True ) #shape ( batch_size, output_h, output_w, 1 ) or # (pred_count, bs, seq_len, h, w)
                preds.append( pred )
            
            return preds

class SimpleDense(tf.keras.Model):
    
    def __init__(self, train_params, model_params):
        super(SimpleDense, self).__init__()

        self.model_params = model_params

        self.dense0 = TimeDistributed( tf.keras.layers.Dense(units= 128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)) )
        self.dense1 = TimeDistributed( tf.keras.layers.Dense(units= 128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)) )
        self.dense2 = TimeDistributed( tf.keras.layers.Dense(units= 128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)) )
        self.dense3 = TimeDistributed( tf.keras.layers.Dense(units= 128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)) )

        self.output_dense = TimeDistributed( tf.keras.layers.Dense(units=1, activation='linear') )
        self.output_activation = layers.CustomRelu_maker(train_params)
        self.float32_output = tf.keras.layers.Activation('linear', dtype='float32')

        self.new_shape = tf.TensorShape( [train_params['batch_size'], model_params['data_pipeline_params']['lookback_target'], int(6*4)] )
        self.do=tf.keras.layers.Dropout(0.15 )#, noise_shape=None, seed=None, **kwargs

    @tf.function
    def call(self, _input, training=True):
        #_input shape (bs, seq_len, c)
        #shape = _input.shape

        x = tf.reshape( _input, self.new_shape )
        # for idx in range(self.model_params['layer_count']):
        #     x = self.LSTM_layers[idx](inputs=x,training=training )  
        
        x0 = self.dense0(x)
        x1 = self.dense1(self.do(x0,training=training))
        x2 = self.dense2(self.do(x1,training=training))
        x3 = self.dense3(self.do( x2, training=training ))
        outp = self.output_dense( x3 )
        outp = self.output_activation(outp)
        outp = self.float32_output(outp)
        return outp
        
    def predict( self, inputs, n_preds, training=True):
            """
                Produces N predictions for each given input
            """
            preds = []
            for count in tf.range(n_preds):
                pred = self.call( inputs, training=True ) #shape ( batch_size, output_h, output_w, 1 ) or # (pred_count, bs, seq_len, h, w)
                preds.append( pred )
            
            return preds

class SimpleConvLSTM(tf.keras.Model):

    def __init__(self, train_params, model_params):
        super(SimpleConvLSTM, self).__init__()

        self.model_params = model_params
        self.ConvLSTM_layers = [ tf.keras.layers.Bidirectional( layers_ConvLSTM2D.ConvLSTM2D( **self.model_params['ConvLSTM_layer_params'][idx] ), merge_mode='concat' )  for idx in range( model_params['layer_count'] ) ]
               
        self.output_conv = self.conv_output = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **self.model_params['outpconv_layer_params'] ) )

        self.do = tf.keras.layers.TimeDistributed( tf.keras.layers.SpatialDropout2D( rate=0.10, data_format = 'channels_last' ) )

        self.output_activation = layers.CustomRelu_maker(train_params)

        self.float32_output = tf.keras.layers.Activation('linear', dtype='float32')

        self.new_shape1 = tf.TensorShape( [train_params['batch_size'],model_params['region_grid_params']['outer_box_dims'][0], model_params['region_grid_params']['outer_box_dims'][1],  train_params['lookback_target'] ,int(6*4)] )
    
    @tf.function
    def call(self, _input, training):
        
        x = tf.transpose( _input, [0, 2,3,1,4])     # moving time axis next to channel axis
        x = tf.reshape( x, self.new_shape1 )        # reshape time and channel axis
        x = tf.transpose( x, [0,3,1,2,4 ] )   # converting back to bs, time, h,w, c

        # for idx in range(self.model_params['layer_count']):
        #     x = self.ConvLSTM_layers[idx](inputs=x, training=training)

        for idx in range(self.model_params['layer_count']):
            if idx==0:
                x0 = self.ConvLSTM_layers[idx](inputs=x,training=training )
                x = x0
            else:
                x = x + self.ConvLSTM_layers[idx](inputs=x,training=training )

        x = self.output_conv( self.do( (x + x0)/2),training=training)
        outp = self.output_activation(x)
        x = self.float32_output(outp)
        return x

class SimpleConvGRU(tf.keras.Model):
    def __init__(self, train_params, model_params):
        super(SimpleConvGRU, self).__init__()

        self.model_params = model_params
        
        
        #region old version 
        self.ConvGRU_layers = [ tf.keras.layers.Bidirectional( layer= layers_ConvGRU2D.ConvGRU2D( **self.model_params['ConvGRU_layer_params'][idx] ), 
                                                                backward_layer= layers_ConvGRU2D.ConvGRU2D( go_backwards=True,**self.model_params['ConvGRU_layer_params'][idx] ) ,
                                                                merge_mode='concat' )  for idx in range( model_params['layer_count'] ) ]
         
        #self.do = tf.keras.layers.TimeDistributed( tf.keras.layers.SpatialDropout2D( rate=model_params['dropout'], data_format = 'channels_last' ) )
        #self.do1 = tf.keras.layers.TimeDistributed( tf.keras.layers.SpatialDropout2D( rate=model_params['dropout'], data_format = 'channels_last' ) )

        #self.conv1 = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **self.model_params['conv1_layer_params'] ) )
        # self.output_conv = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **self.model_params['outpconv_layer_params'] ) )

        # self.float32_output = tf.keras.layers.Activation('linear', dtype='float32')

        # self.output_activation = layers.CustomRelu_maker(train_params, dtype='float32')

        # self.new_shape1 = tf.TensorShape( [train_params['batch_size'],model_params['region_grid_params']['outer_box_dims'][0], model_params['region_grid_params']['outer_box_dims'][1],  train_params['lookback_target'] ,int(6*4)] )
        #endregion

        #region 1 layer version
        model_params['layer_count'] = 1
        self.ConvGRU_layers = [ tf.keras.layers.Bidirectional( layer= layers_ConvGRU2D.ConvGRU2D( **self.model_params['ConvGRU_layer_params'][idx] ), 
                                                                backward_layer= layers_ConvGRU2D.ConvGRU2D( go_backwards=True,**self.model_params['ConvGRU_layer_params'][idx] ) ,
                                                                merge_mode='concat' )  for idx in range( model_params['layer_count'] ) ]

        self.output_conv = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **self.model_params['outpconv_layer_params'] ) )

        self.float32_output = tf.keras.layers.Activation('linear', dtype='float32')

        self.output_activation = layers.CustomRelu_maker(train_params, dtype='float32')

        self.new_shape1 = tf.TensorShape( [train_params['batch_size'],model_params['region_grid_params']['outer_box_dims'][0], model_params['region_grid_params']['outer_box_dims'][1],  train_params['lookback_target'] ,int(6*4)] )
        #endregion

        #region new version
        # self.linear_dim_expansion = tf.keras.layers.Dense( units = int(self.model_params['ConvGRU_layer_params'][0]['filters']*2 ), activation='linear', use_bias=False )
        # self.ConvGRU_layers = [ tf.keras.layers.Bidirectional( layer= layers_ConvGRU2D.ConvGRU2D( **self.model_params['ConvGRU_layer_params'][idx] ), 
        #                                                         backward_layer= layers_ConvGRU2D.ConvGRU2D( go_backwards=True,**self.model_params['ConvGRU_layer_params'][idx] ) ,
        #                                                         merge_mode='concat' )  for idx in range( model_params['layer_count'] ) ]
        # self.output_conv = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **self.model_params['outpconv_layer_params'] ) )

        # self.float32_output = tf.keras.layers.Activation('linear', dtype='float32')

        # self.output_activation = layers.CustomRelu_maker(train_params, dtype='float32')

        # self.new_shape1 = tf.TensorShape( [train_params['batch_size'],model_params['region_grid_params']['outer_box_dims'][0], model_params['region_grid_params']['outer_box_dims'][1],  train_params['lookback_target'] ,int(6*4)] )
        # endregion

    @tf.function
    def call(self, _input, training):
        
        x = tf.transpose( _input, [0, 2,3,1,4])     # moving time axis next to channel axis
        x = tf.reshape( x, self.new_shape1 )        # reshape time and channel axis
        x = tf.transpose( x, [0,3,1,2,4 ] )   # converting back to bs, time, h,w, c


        # for idx in range(self.model_params['layer_count']):
        #     if idx==0:
        #         x0 = self.ConvGRU_layers[idx](inputs=x,training=training )
        #         x = x0
        #     else:
        #         x = x + self.ConvGRU_layers[idx](inputs=x,training=training )
        
        
        # x = self.conv1( self.do( tf.concat([x,x0] ,axis=-1) ), training=training )
        # outp = self.output_conv( self.do1( x ), training=training )
        # outp = self.float32_output(outp)
        # outp = self.output_activation(outp)


        x = self.ConvGRU_layers[0](inputs=x,training=training )
        outp = self.output_conv( x , training=training )
        outp = self.float32_output(outp)
        outp = self.output_activation(outp)

        
        # x0 = self.linear_dim_expansion( _input )
        # for idx in range( self.model_params['layer_count'] ):
        #     if idx==0:
        #         x = self.ConvGRU_layers[idx](inputs=x+x0,training=training )
        
        # outp = self.output_conv( x , training=training )
        # outp = self.float32_output(outp)
        # outp = self.output_activation(outp)
        return outp
        