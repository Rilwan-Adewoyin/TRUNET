class HParams():
    
    def __init__( self ,**kwargs ):    
        pass

    def __call__(self):
        return self.__dict__


class model_hparameters(HParams):

    def __init__(self, **kwargs):
        super( model_hparameters, self ).__init__()

    def __call__(self):


    return hparam

class train_hparameters():
    
    batch_size = 10
