import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TimeDistributed, Conv2D, Dropout, MaxPooling2D, UpSampling2D, SpatialDropout2D
import layers
import layers_convgru2D
import copy


def model_loader(t_params,m_params ):
    
    model_name = m_params['model_name']
    
    if(model_name=="TRUNET"):
        model = TRUNET(t_params, m_params)
    
    elif(model_name=="SimpleConvGRU"):
        model = SimpleConvGRU(t_params, m_params)
    
    elif(model_name=="UNET"):
        model = UNET(t_params, m_params )

    return model

class SimpleConvGRU(tf.keras.Model):
    def __init__(self, t_params, m_params):
        super(SimpleConvGRU, self).__init__()

        self.dc = np.asarray( [m_params['model_type_settings']['discrete_continuous']],dtype=np.bool )
        
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
            
            self.output_conv_val = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **m_params['outpconv_layer_params'] ) )
            self.output_conv_prob = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **m_params['outpconv_layer_params'] ) )


            self.float32_output = tf.keras.layers.Activation('linear', dtype='float32')
            self.output_activation_val = layers.CustomRelu_maker(t_params, dtype='float32')
            self.output_activation_prob = tf.keras.layers.Activation('sigmoid', dtype='float32')

        else:
            self.conv1 = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **m_params['conv1_layer_params'] ) )
            self.output_conv = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **m_params['outpconv_layer_params'] ) )
                
            self.float32_output = tf.keras.layers.Activation('linear', dtype='float32')

            self.output_activation = layers.CustomRelu_maker(t_params, dtype='float32')
        
        self.new_shape1 = [0,m_params['region_grid_params']['outer_box_dims'][0], m_params['region_grid_params']['outer_box_dims'][1],  t_params['lookback_target'] ,int(6*4)]
    
    @tf.function
    def call(self, _input, training):
        
        x = tf.transpose( _input, [0, 2,3,1,4])     # moving time axis next to channel axis
        self.new_shape1[0] = _input.shape[0]
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

            outp = self.output_conv( self.do1( x, training=training ), training=training )
            outp = self.float32_output(outp)
            outp = self.output_activation(outp)
        
        else:
            x = self.do(tf.concat([x,x0], axis=-1), training=training)

            x_vals = self.conv1_val( x )
            x_prob = self.conv1_prob( x )

            outp_vals = self.output_conv_val( self.do2( x_vals, training=training ))
            outp_prob = self.output_conv_prob( self.do2( x_prob, training=training ))

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

class TRUNET(tf.keras.Model):
    """
        TRU-NET Encoder Decoder Model

    """
    def __init__(self, t_params, m_params, **kwargs):
        """
        Args:
            t_params (dict): params related to training/testing
            m_params ([type]): params related to model
        """        
        super(TRUNET, self).__init__()

        h_w_enc = h_w_dec = m_params['region_grid_params']['outer_box_dims']
        
        # Encoder
        self.encoder = layers.TRUNET_Encoder( t_params, m_params['encoder_params'], h_w_enc,  
            attn_ablation=m_params['model_type_settings'].get('attn_ablation',0) )

        #Decoder
        self.decoder = layers.TRUNET_Decoder( t_params, m_params['decoder_params'], h_w_dec )
        
        #Output Layer
        self.output_layer = layers.TRUNET_OutputLayer( t_params, m_params['output_layer_params'], 
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

class UNET(tf.keras.Model):
    
    def __init__(self, t_params, m_params, **kwargs):
        """
        Args:
            t_params (dict): params related to training/testing
            m_params ([type]): params related to model
        """        
        super(UNET, self).__init__()

        #inputs = Input(input_size)
        self.dc = np.asarray( [m_params['model_type_settings']['discrete_continuous']],dtype=np.bool )

        self.conv2d_1 = Conv2D( 80, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        self.pool_1 = MaxPooling2D(pool_size=(2, 2))

        self.conv2d_2 = Conv2D(80, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        self.pool_2 = MaxPooling2D(pool_size=(2, 2))
        
        self.conv2d_3 = Conv2D(64, 4, activation='relu', padding = 'same', kernel_initializer = 'he_normal' ) 
        self.pool_3 = MaxPooling2D(pool_size=(2, 2))
        self.dropout_3 = SpatialDropout2D(0.5)

        self.upsample_4 = UpSampling2D(size=(2,2))
        self.conv2d_4 =  Conv2D(80, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')

        self.upsample_5 = UpSampling2D(size=(2,2))
        self.conv2d_5 =  Conv2D(80, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')

        self.conv_2d_6val =  Conv2D(80, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        self.conv_2d_6prob =  Conv2D(80, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')

        self.float32_output = tf.keras.layers.Activation('linear', dtype='float32')
        self.output_activation_val = layers.CustomRelu_maker(t_params, dtype='float32')
        
        if m_params['model_type_settings']['discrete_continuous']:
            self.output_activation_prob = tf.keras.layers.Activation('sigmoid', dtype='float32')
    

    @tf.function
    def call(self, _input, training=False):

        x1 = self.conv2d_1(_input)
        x1 = self.pool_1(x1)

        x2 = self.conv2d_2(x1)
        x2 = self.pool_2(x2)

        x3 = self.conv2d_3(x2)
        x3 = self.pool_3(x3)

        drop3 = self.dropout_3(x3, training=training)

        x4_up = self.upsample_4(drop3)
        x4 = self.conv2d_4( x4_up )
        merge4 = tf.concat([drop3, x4], axis = 3)

        x5_up = self.upsample_5(merge4)
        x5 = self.conv2d_5( x5_up )
        merge5 = tf.concat([x2, x5], axis = 3)

        
        x6_val = self.conv_2d_6val( merge5 )
        outp = self.float32_output(x6_val)
        outp = self.output_activation(x6_val)

        if self.dc == True:
            x6_prob = self.conv_2d_6prob( merge5 )
            outp_prob = self.float32_output(x6_prob)
            outp_prob = self.output_activation_prob(outp_prob)
                 
            outp = tf.stack([outp_vals, outp_prob],axis=0)
            
        return outp

    def predict( self, inputs, n_preds, training=True):
        """
            Produces N predictions for a each input
        """
        preds = []
        for count in tf.range(n_preds):
            pred = self.call( inputs, training=True ) #shape ( batch_size, time, output_h, output_w, 1 ) or # (pred_count, time, bs, seq_len, h, w)
            preds.append( pred )
        return preds