import tensorflow as tf
import layers


def model_loader(train_params, model_params ):
    if(model_params['model_name']=="DeepSD"):
        return SuperResolutionModel( train_params, model_params)

class SuperResolutionModel( tf.keras.Model ):
    def __init__(self, train_params, model_params ):
        super(SuperResolutionModel, self).__init__()
        
        self.SRCNN_1 = layers.SRCNN( train_params, model_params )
         
    def call(self, inputs, tape):
        x = self.SRCNN_1(inputs, tape=tape)
        #self.losses += self.SRCNN_1.losses()
        #self.losses += self.SRCNN_1.posterior_entropy()
        #self.losses += self.SRCNN_1.prior_cross_entropy()
        return x
    
    #TODO: override keras model method for performing testing
    #def predict(self, inputs):



class TrajGRU(tf.keras.Model):
    def __init__(self, hparams):
        super(TrajGRU, self).__init__()

    def call(self, inputs, tape):
        raise NotImplementedError
