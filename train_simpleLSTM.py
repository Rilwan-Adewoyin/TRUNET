import sys
import tensorflow as tf

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional, Reshape, Input

import data_generators

import utility
def main(t_params, m_params):

    ds_train = data_generators.load_data_ati( train_params, model_params, day_to_start_at=train_params['train_start_date'], data_dir=train_params['data_dir'])
    ds_val = data_generators.load_data_ati( train_params, model_params, day_to_start_at=train_params['val_start_date'], data_dir=train_params['data_dir'] )

    n_steps = model_params['data_pipeline_params']['feature_lookback']
    n_features = 6

    model = Sequential()
    model.add(Reshape( -1,24), input_shape= ( n_steps,6) )
    model.add(Bidirectional(LSTM(64, activation='relu',dropout=0.1, recurrent_dropout=0.1)))
    model.add(Bidirectional(LSTM(64, activation='relu',dropout=0.1, recurrent_dropout=0.1)))
    model.add(Bidirectional(LSTM(64, activation='relu',dropout=0.1, recurrent_dropout=0.1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(x=ds_train, validation_split=0.2, epochs=200, verbose=0)

if __name__ == "__main__":
    s_dir = utility.get_script_directory(sys.argv[0])
    args_dict = utility.parse_arguments(s_dir)
    train_params, model_params = utility.load_params_train_model(args_dict)
    
    main(train_params(), model_params )