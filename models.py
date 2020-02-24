import tensorflow as tf
import layers


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
        
    elif(model_name == "SimpleConvLSTM"):
        raise NotImplementedError

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
    
    @tf.function
    def call(self, inputs, pred=False):
        x = self.SRCNN_1(inputs, pred)
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
            pred = self.call( inputs, pred=pred ) #shape ( batch_size, output_h, output_w, 1 )
            preds.append( pred )
        
        return preds

class THST(tf.keras.Model):
    """
    Temporal Hierarchical Spatial Transformer 
    """
    def __init__(self, train_params, model_params):
        super(THST, self).__init__()

        self.encoder = layers.THST_Encoder( train_params, model_params['encoder_params'] )
        self.decoder = layers.THST_Decoder( train_params, model_params['decoder_params'] )
        self.output_layer = layers.THST_OutputLayer( train_params, model_params['output_layer_params'], model_params['model_type_settings']  )

        self.float32_output = tf.keras.layers.Activation('linear', dtype='float32')

    @tf.function
    def call(self, _input, tape=None, training=False):
        
        hidden_states_2_enc, hidden_states_3_enc, hidden_4_enc, hidden_5_enc = self.encoder( _input, training )
        hidden_states_dec = self.decoder( hidden_states_2_enc, hidden_states_3_enc, hidden_4_enc, hidden_5_enc, training )
        output = self.output_layer(hidden_states_dec, training)
        output = self.float32_output(output)
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
        self.LSTM_layers = [ tf.keras.layers.Bidirectional( tf.keras.layers.LSTM( **model_params['layer_params'][idx] ), merge_mode='concat' ) for idx in range( model_params['layer_count'] ) ]
        self.LSTM_init_state = [ [tf.Variable(tf.zeros( (train_params['batch_size'],model_params['layer_params'][idx]['units']), dtype=tf.float16)) ]*2 for idx in range( model_params['layer_count']) ]
        for idx in range(model_params['layer_count']):
            #layers._dtype = 'float16'
            #self.LSTM_layers[idx].states = self.LSTM_init_state[idx]
            self.LSTM_layers[idx].forward_layer.states = self.LSTM_init_state[idx]
            self.LSTM_layers[idx].backward_layer.states = self.LSTM_init_state[idx]
            #layers.reset_states()
        self.output_dense = tf.keras.layers.Dense(units=1, activation='relu')
        self.float32_output = tf.keras.layers.Activation('linear', dtype='float32')

        self.new_shape = tf.TensorShape( [train_params['batch_size'], train_params['lookback_target'], int(6*4)] )

    def call(self, _input, training=False):
        #_input shape (bs, seq_len, c)
        #shape = _input.shape
        x = tf.reshape( _input, self.new_shape )
        for idx in tf.range(self.model_params['layer_count']):
            x = self.LSTM_layers[idx](inputs=x ) #, initial_state=self.LSTM_init_state[idx] ) #returns hidden states at each level
            #x = self.LSTM_layers[idx](inputs=x) #returns hidden states at each level
        x = self.output_dense(x)
        x = self.float32_output(x)
        return x
        
    def predict( self, inputs, n_preds, training=True):
            """
                Produces N predictions for each given input
            """
            preds = []
            for count in tf.range(n_preds):
                pred = self.call( inputs, training=True ) #shape ( batch_size, output_h, output_w, 1 ) or # (pred_count, bs, seq_len, h, w)
                preds.append( pred )
            
            return preds

