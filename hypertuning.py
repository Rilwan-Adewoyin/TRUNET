from netCDF4 import Dataset, num2date
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import argparse
import ast
import gc
import logging
import math

import sys
import time

import numpy as np
import pandas as pd
import psutil

#import tensorflow as tf
#from tensorflow.keras.mixed_precision import experimental as mixed_precision

# try:
#     import tensorflow_addons as tfa
# except Exception as e:
#     tfa = None

import data_generators
import custom_losses as cl
import hparameters
import models
import utility

# tf.keras.backend.set_floatx('float16')
# tf.keras.backend.set_epsilon(1e-3)

# try:
#     gpu_devices = tf.config.list_physical_devices('GPU')
# except Exception as e:
#     gpu_devices = tf.config.experimental.list_physical_devices('GPU')

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

import subprocess

# Structure of code

# User passes in: 
    # kwargs that will be pass on to train.py

# Already Defined
    # kwargs that will be tested in a given range

# Make a list of the subprocess. commands that will be run for hyper-parameter training
# Make a list of the subprocess commands that will be run for hyper-parameter testing
# 	- Need to adjust the trianing script to take new hyper-parameter == "htuning".
# 		-if htuning=True: Changing Model naming, Model Saving [Put this in new folder called model hypertuning]
# 			include ( (max lr,min lr), B_1, B_2, min lr, rec_dropout and input_dropout ) in 

#   - Create a hyper-parameter testing pandas dictionary  for which after each model is train/tested the final_train_loss&val_loss/test_loss is appended
        # new directory called hyptertuning
#       - Append final_train_loss, final_val_loss, test_loss to a 
 
# Training Example
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py -mn "TRUNET" -ctsm "1979_2009_2014" -mts "{'stochastic':False,'stochastic_f_pass':1,'distr_type':'Normal','discrete_continuous':True,'var_model_type':'mc_dropout','do':0.2,'ido':0.2,'rdo':0.3,'location':['Cardiff','London','Glasgow','Birmingham','Lancaster','Manchester','Liverpool','Bradford','Edinburgh','Leeds','Dublin', 'Norwich', 'Truro', 'Newry','Plymouth','Bangor'] }" -dd "/media/Data3/akanni/Rain_Data_Mar20" -bs 64

# Testing Example 
# Chain the below together using and statements


#CUDA_VISIBLE_DEVICES=2 python3 predict.py -mn "TRUNET" -ctsm "2009_2016_2018-12-31" -ctsm_test '1979_1988-12-31', -mts "{'stochastic':True,'stochastic_f_pass':2,'distr_type':'Normal','discrete_continuous':True,'var_model_type':'mc_dropout','do':0.2,'ido':0.2,'rdo':0.3,'location':['Cardiff','London','Glasgow','Birmingham','Lancaster','Manchester','Liverpool','Bradford','Edinburgh','Leeds','Dublin', 'Norwich', 'Truro', 'Newry','Plymouth','Bangor'],'location_test':['All'],'value_dropout':True}" -ts "{'region_pred':True}" -dd "/media/Data3/akanni/Rain_Data_Mar20" -bs 130


def main(m_params):

    #defining hyperam range

    lrs_max_min = [ ( 1e-3, 1e-4) , (1e-4,1e-5)]
    b1s = [0.75, 0.9]
    b2s = [0.99]
    
    inp_dropouts = [0.2,0.3]
    rec_dropouts = [0.15,0.25,0.35]

    counter =  0

    f =  open("hypertune.txt","w")
    for lr in lrs_max_min:
        for b1 in b1s:
            for b2 in b2s:
                for inpd in inp_dropouts:
                    for recd in rec_dropouts:

                        print(f"\n\n Training model v{counter}")
                        train_cmd = train_cmd_maker( m_params['model_name'], lr, b1, b2, inpd, recd, counter )
                        # try:
                        #     outp = subprocess.run( train_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, check=True )
                        # except subprocess.CalledProcessError as e:
                        #     print(e.args)
                        #     print("\n\n")
                        #     print(e.stderr)
                        
                        f.write(f'{train_cmd}\n && ')
                            

                        # popen = subprocess.Popen( train_cmd, stdout=subprocess.PIPE, shell=True, check=True )
                        # for stdout_line in iter(popen.stdout.readline, ""):
                        #     yield stdout_line 
                        # return_code = popen.wait()
                        # if return_code:
                        #     raise subprocess.CalledProcessError(return_code, train_cmd)
                        print(f" Testing model v{counter}")
                        test_cmd = test_cmd_maker( m_params['model_name'], inpd, recd, counter )
                        f.write(f'{train_cmd}\n && ')
                        
                        # outp = subprocess.run( test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, check=True )
                        # counter = counter + 1
    f.close()

def train_cmd_maker( mn ,lr_min_max, b1, b2, inp_drop, rec_drop, counter):
    cmd = [
        "CUDA_VISIBLE_DEVICES=1,2,3",
        "python3", "train.py","-mn",f"{mn}",
        "-ctsm", "1994_2009_2014", "-mts",
        f"\"{{'htuning':True, 'htune_version':{counter},'stochastic':False,'stochastic_f_pass':1,'discrete_continuous':True,'var_model_type':'mc_dropout','do':0.2,'ido':{inp_drop},'rdo':{rec_drop}, 'b1':{b1}, 'b2':{b2}, 'lr_max':{lr_min_max[0]}, 'lr_min':{lr_min_max[1]}, 'location':['Cardiff','London','Glasgow','Birmingham','Lancaster','Manchester','Liverpool','Bradford','Edinburgh','Leeds'] }}\"",
        "-dd", "/media/Data3/akanni/Rain_Data_Mar20", "-bs", "48"]
    
    cmd2 = ' '.join(cmd)
    return cmd2

def test_cmd_maker( mn,inp_drop, rec_drop, counter):
    cmd = [ 
        "CUDA_VISIBLE_DEVICES=1",
        #"export", "CUDA_VISIBLE_DEVICES=1", "&&",
        "python 3", "predict.py", "-mn", f"{mn}", "-ctsm", "1994_2009_2014", "-ctsm_test", "2014_2019-07-04", "-mts",
    f"\"{{'htuning':True, 'htune_version':{counter},'stochastic':True,'stochastic_f_pass':2,'distr_type':'Normal','discrete_continuous':True,'var_model_type':'mc_dropout', 'do':0.2,'ido':{inp_drop},'rdo':{rec_drop}, 'location':['Cardiff','London','Glasgow','Birmingham','Lancaster','Manchester','Liverpool','Bradford','Edinburgh','Leeds'],'location_test':['Cardiff','London','Glasgow','Birmingham','Lancaster','Manchester','Liverpool','Bradford','Edinburgh','Leeds']}}\"",
    "-ts", "{'region_pred':True}", "-dd", "/Data/Rain_Data_Mar20", "-bs", f"{71}" ]

    return cmd
    
    



if __name__ == "__main__":
    s_dir = utility.get_script_directory(sys.argv[0])
    args_dict = utility.parse_arguments(s_dir)

    # get training and model params
    #_, m_params = utility.load_params(args_dict)
    
    
    main( args_dict )
    

    #CUDA_VISIBLE_DEVICES=0,1,2,3 python3 hypertuning.py -mn "SimpleConvGRU" -mts "{}' -ctsm ""    
