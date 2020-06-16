import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TimeDistributed
import layers
import layers_convgru2D
import copy


def model_loader(t_params,m_params ):
    
    model_name = m_params['model_name']
    
    if(model_name=="THST"):
        model =THST(t_params, m_params)
    
    elif(model_name=="SimpleConvGRU"):
        model = SimpleConvGRU(t_params, m_params)
    
    elif(model_name=="TRUNET_EF"):
        model = TRUNET_EF(t_params, m_params) 

    return model

class TRUNET_EF(tf.keras.Model):
    """
    Temporal Hierarchical Spatial Transformer 

    """
    def __init__(self, t_params, m_params, **kwargs):
        super(TRUNET_EF, self).__init__()

        self.input_shape = m_params['input_shape']

        h_w_enc = m_params['encoder_params']['h_w_enc']
        h_w_dec = m_params['decoder_params']['h_w_dec']

        if m_params['model_type_settings']['location'] not in ["wholeregion"]:
            h_w_enc = h_w_dec = m_params['region_grid_params']['outer_box_dims']

        self.encoder = layers.TRUNET_EF_Encoder( t_params, m_params['encoder_params'], h_w_enc)

        self.decoder = layers.TRUNET_EF_Forecaster( t_params, m_params['decoder_params'], h_w_dec )

        self.output_layer_z = layers.THST_OutputLayer( t_params, m_params['output_layer_params'], m_params['model_type_settings'],
            m_params['dropout'] )

        self.output_layer_t = layers.THST_OutputLayer( t_params, m_params['output_layer_params'], m_params['model_type_settings'],
            m_params['dropout']  )
        #self.float32_custom_relu = layers.OutputReluFloat32(t_params) 
        
        ##Output Lyaer
        # self.conv_hidden1 = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **layer_params[0] ) )
        # self.conv_hidden2 = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **layer_params[0] ) )
        # self.conv_output = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **layer_params[1] ) )
        # self.conv_output2 = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **layer_params[1] ) )
        # self.do1 = tf.keras.layers.TimeDistributed( tf.keras.layers.SpatialDropout2D( rate=dropout_rate, data_format = 'channels_last' ) )
        # self.do2 = tf.keras.layers.TimeDistributed( tf.keras.layers.SpatialDropout2D( rate=dropout_rate, data_format = 'channels_last' ) )
        # self.outputf32 = tf.keras.layers.Activation('linear', dtype='float32')


    @tf.function
    def call(self, _input, tape=None, training=False):
        _input = tf.keras.Input( self.input_shape)

        hs_list_enc = self.encoder(_input, training=training)
        hs_dec = self.decoder(hs_list_enc, training=training)
        
        outp1 = self.conv_hidden1( self.do1(hs_dec,training=training), training=training )
        outp2 = self.conv_hidden2( self.do2(hs_dec,training=training), training=training ) 	
		
        outp1 = self.conv_output( outp1, training=training ) #shape (bs, height, width)
        outp2 = self.conv_output( outp2, training=training ) #shape (bs, height, width)

        outp1 = self.float32_custom_relu(outp1)   
        outp2 = self.float32_custom_relu(outp2)   

        output = tf.concat([outp1,outp2],axis=-1)
        
        #output = self.float32_custom_relu(output)   
        return output


    def predict(self, inputs, n_preds, training=True):
        """
            Produces N predictions for each given input
        """
        preds = []
        for count in tf.range(n_preds):
            pred = self.call( inputs, training=True ) #shape ( batch_size, output_h, output_w, 1 ) or # (pred_count, bs, seq_len, h, w)
            preds.append( pred )
        return preds

class SimpleConvGRU(tf.keras.Model):
    def __init__(self, t_params, m_params):
        super(SimpleConvGRU, self).__init__()

        self.dc = np.asarray( [m_params['model_type_settings']['discrete_continuous']],dtype=np.bool )
        self.dsif = np.asarray( [ 'downscale_input_factor' in m_params ] , dtype=np.bool )
        self.layer_count = m_params['layer_count']
               
        
        self.ConvGRU_layers = [ tf.keras.layers.Bidirectional( layer= layers_convgru2D.ConvGRU2D( **m_params['ConvGRU_layer_params'][idx] ), 
                                                                backward_layer= layers_convgru2D.ConvGRU2D( go_backwards=True,**copy.deepcopy(m_params['ConvGRU_layer_params'][idx]) ) ,
                                                                merge_mode='concat' )  for idx in range( m_params['layer_count'] ) ]
         
        self.do = tf.keras.layers.TimeDistributed( tf.keras.layers.SpatialDropout2D( rate=m_params['dropout'], data_format = 'channels_last' ) )
        self.do1 = tf.keras.layers.TimeDistributed( tf.keras.layers.SpatialDropout2D( rate=m_params['dropout'], data_format = 'channels_last' ) )
        self.do2 = tf.keras.layers.TimeDistributed( tf.keras.layers.SpatialDropout2D( rate=m_params['dropout'], data_format = 'channels_last' ) )

        if m_params['model_type_settings']['discrete_continuous']:
            self.conv1_val = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **m_params['conv1_layer_params'] ) )
            self.conv1_prob = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **m_params['conv1_layer_params'] ) )

            if self.dsif:
                self.conv2_val = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2DTranspose( **m_params['conv2_layer_params'] ) )
                self.conv2_prob = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2DTranspose( **m_params['conv2_layer_params'] ) )
            
            self.output_conv_val = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **m_params['outpconv_layer_params'] ) )
            self.output_conv_prob = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **m_params['outpconv_layer_params'] ) )


            self.float32_output = tf.keras.layers.Activation('linear', dtype='float32')
            self.output_activation_val = layers.CustomRelu_maker(t_params, dtype='float32')
            self.output_activation_prob = tf.keras.layers.Activation('sigmoid', dtype='float32')

        else:
            self.conv1 = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **m_params['conv1_layer_params'] ) )
            self.output_conv = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **m_params['outpconv_layer_params'] ) )

            if self.dsif:
                self.conv2 = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2DTranspose(**m_params['conv2_layer_params']) )
                
            self.float32_output = tf.keras.layers.Activation('linear', dtype='float32')

            self.output_activation = layers.CustomRelu_maker(t_params, dtype='float32')

        self.new_shape1 = tf.TensorShape( [t_params['batch_size'],m_params['region_grid_params']['outer_box_dims'][0], m_params['region_grid_params']['outer_box_dims'][1],  t_params['lookback_target'] ,int(6*4)] )

    @tf.function
    def call(self, _input, training):
        
        x = tf.transpose( _input, [0, 2,3,1,4])     # moving time axis next to channel axis
        x = tf.reshape( x, self.new_shape1 )        # reshape time and channel axis
        x = tf.transpose( x, [0,3,1,2,4 ] )   # converting back to bs, time, h,w, c

        
        for idx in range(self.layer_count):
            if idx==0:
                x0 = self.ConvGRU_layers[idx](inputs=x,training=training )
                x = x0
            else:
                x = x + self.ConvGRU_layers[idx](inputs=x,training=training )
        
        
        if self.dc ==  False :
            x = self.conv1( self.do( tf.concat([x,x0] ,axis=-1),training=training ), training=training )
            if self.dsif == True:
                x = self.conv2( self.do( x, training ), training=training )
            outp = self.output_conv( self.do1( x, training=training ), training=training )
            outp = self.float32_output(outp)
            outp = self.output_activation(outp)
        
        else:
            x_vals = self.conv1_val( self.do( tf.concat([x,x0] ,axis=-1),training=training ), training=training )
            x_prob = self.conv1_prob( self.do( tf.concat([x,x0] ,axis=-1),training=training ), training=training )

            if self.dsif == True:
                x_vals = self.conv2_vals( self.do1( x_vals, training ), training=training )
                x_prob = self.conv2_probs( self.do1( x_prob, training ), training=training )

            outp_vals = self.output_conv_val( self.do2( x_vals, training=training ), training=training )
            outp_prob = self.output_conv_prob( self.do2( x_prob, training=training ), training=training )

            outp_vals = self.float32_output(outp_vals)
            outp_prob = self.float32_output(outp_prob)

            outp_vals = self.output_activation_val(outp_vals)
            outp_prob = self.output_activation_prob(outp_prob)

            outp = tf.stack([outp_vals, outp_prob],axis=0)

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

class THST(tf.keras.Model):
    """
        TRU-NET Encoder Decoder Model

    """
    def __init__(self, t_params, m_params, **kwargs):
        """
        Args:
            t_params (dict): params related to training/testing
            m_params ([type]): params related to model
        """        
        super(THST, self).__init__()

        
        self.mg = m_params['model_type_settings'].get('mult_gpu',False)
        h_w_enc = h_w_dec = m_params['region_grid_params']['outer_box_dims']
        
        # Encoder
        self.encoder = layers.THST_Encoder( t_params, m_params['encoder_params'], h_w_enc,  
            attn_ablation=m_params['model_type_settings'].get('attn_ablation',0) )

        #Decoder
        self.decoder = layers.THST_Decoder( t_params, m_params['decoder_params'], h_w_dec )
        
        #Output Layer
        self.output_layer = layers.THST_OutputLayer( t_params, m_params['output_layer_params'], 
                                m_params['model_type_settings'], m_params['dropout'])

    @tf.function
    def call(self, _input, tape=None, training=False):
    
        hs_list_enc = self.encoder(_input, training=training)
        hs_dec = self.decoder(hs_list_enc, training=training)
        output = self.output_layer(hs_dec, training=training)
        return output


    def predict( self, inputs, n_preds, training=True):
        """
            Produces N predictions for a each input
        """
        preds = []
        for count in tf.range(n_preds):
            pred = self.call( inputs, training=True ) #shape ( batch_size, output_h, output_w, 1 ) or # (pred_count, bs, seq_len, h, w)
            preds.append( pred )
        return preds
