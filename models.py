import tensorflow as tf
import layers


def model_loader(train_params,model_params ):
    
    if type(model_params)==list:
        model_name = model_params[0]['model_name']
    else:
        model_name = model_params['model_name']

    if(model_name=="THST"):
        return THST(train_params, model_params)
    
    elif(model_name=="DeepSD"):
            return SuperResolutionModel( train_params, model_params)
    

    


class SuperResolutionModel( tf.keras.Model ):
    def __init__(self, train_params, model_params ):
        super(SuperResolutionModel, self).__init__()
        
        self.SRCNN_1 = layers.SRCNN( train_params, model_params )
        # self.SRCNN_2 = layers.SRCNN( train_params, model_params[1] )
        #self.SRCNN_3 = layers.SRCNN( train_params, model_params[2] )
         
    def call(self, inputs, tape=None, pred=False):
        x = self.SRCNN_1(inputs, tape, pred)
        # x = self.SRCNN_2( x, pred )
        
        return x
    
    def predict( self, inputs, n_preds, pred=True):
        """
            Produces N predictions for each given input
        """
        preds = []
        for count in tf.range(n_preds):
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

    #@tf.function
    def call(self, _input, tape=None, pred=False):
        
        hidden_states_2_enc, hidden_states_3_enc, hidden_4_enc, hidden_5_enc = self.encoder( _input )
        hidden_states_dec = self.decoder( hidden_states_2_enc, hidden_states_3_enc, hidden_4_enc, hidden_5_enc )
        output = self.output_layer(hidden_states_dec)
        
        return output

    def predict( self, inputs, n_preds):
        """
            Produces N predictions for each given input
        """
        preds = []
        for count in tf.range(n_preds):
            pred = self.call( inputs, pred=True ) #shape ( batch_size, output_h, output_w, 1 )
            preds.append( pred )
        
        return preds
