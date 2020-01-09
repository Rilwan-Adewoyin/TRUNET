import tensorflow as tf
import layers


def model_loader(train_params, model_params ):
    if(model_params[0]['model_name']=="DeepSD"):
        return SuperResolutionModel( train_params, model_params)

class SuperResolutionModel( tf.keras.Model ):
    def __init__(self, train_params, model_params ):
        super(SuperResolutionModel, self).__init__()
        
        self.SRCNN_1 = layers.SRCNN( train_params, model_params[0] )
        self.SRCNN_2 = layers.SRCNN( train_params, model_params[1] )
        #self.SRCNN_3 = layers.SRCNN( train_params, model_params[2] )
         
    def call(self, inputs, tape=None, pred=False):
        x = self.SRCNN_1(inputs, tape, pred)
        x = self.SRCNN_2( x, pred )
        
        #self.losses += self.SRCNN_1.losses()
        #self.losses += self.SRCNN_1.posterior_entropy()
        #self.losses += self.SRCNN_1.prior_cross_entropy()
        return x
    
    def predict( self, inputs, n_preds):
        """
            Produces N predictions for each given input
        """
        preds = []
        for count in tf.range(n_preds):
            pred = self.call( inputs, pred=True ) #shape ( batch_size, output_h, output_w, 1 )
            preds.append( pred )
        
        return preds

class THST(tf.keras.Model):
    """
    Temporal Hierachical Spatial Transformer 
    """
    def __init__(self, train_params, model_params):
        super(THST, self).__init__()

        self.encoder = layers.THST_Encoder( model_params['encoder_params'] )
        self.decoder = layers.THST_Decoder( model_params['decoder_params'] )

    def call(self, input, tape, pred):
        
        encdr_last_cell_state_1, encdr_last_hstate_1, encdr_last_cell_state_2, encdr_last_hstate_2, encdr_last_cell_state_3, encdr_last_hstate_3 = self.encoder( _input ) 
        

    def predict( self, inputs, n_preds):
        """
            Produces N predictions for each given input
        """
        preds = []
        for count in tf.range(n_preds):
            pred = self.call( inputs, pred=True ) #shape ( batch_size, output_h, output_w, 1 )
            preds.append( pred )
        
        return preds
