# TRU-NET: Precipitation Prediction

Code for predicting precipitation using model field data from IFS-ERA5 (temperature, geopotential, wind velocity, etc.) as predictors for sub-areas across the British Isle.
The Data used in this project is available at the following Google Drive link. Users must download and decompress the Rain_Data_Mar20 folder. 
Keywords: TRU_NET, Downscaling, Cross Attention, Hierarchical GRU

## Training Scripts
The code below can be used to train a TRU-NET model. The list the follows the code explains the role of the arguments.

`python3 train.py -mn "THST" -ctsm "1998_2010_2010_2012" -mts "{'stochastic':False,'stochastic_f_pass':1,'distr_type':'Normal','discrete_continuous':True,'var_model_type':'mc_dropout','location':['Cardiff','London','Glasgow']}" -dd "/Data/Rain_Data_Mar20" -bs 13 -do 0.2 -ido 0.2 -rdo 0.3`

* mn = string : model_name, can be THST or SimpleConvGRU
* ctsm = string : dates for training and validation in format `trainstart_trainend_valstart_val_end` eg. "1979_1981_1982_1983"
* mts = dictionary :specific settings
*   var_model_type = str: Whether to train BNN or deterministic model `mc_dropout` or `Deterministic`
*	stochastic = Bool: To train model with multiple forward passes
*	stochastic_f_pass = int : how many forward passes if stochastic=True
*	distr_type = Normal/LogNormal: Distribution for rain in non CC distribution or Distribution for g in CC distribution
*	discrete_continuous = Bool: Whether or not to use CC distribution
*	location: list: Locations to train on. To train on whole UK use `["All"]`
* dd = string : data directory
* bs = int : batch size
* do = float : dropout
* ido = float : input_dropout
* rdo = float : recurrent_dropout

Locations can be chosen from the following list: London, Cardiff, Glasgow, Lancaster, Bradford, Manchester, Birmingham, Liverpool, Leeds, Edinburgh, Belfast, Dublin, LakeDistrict, Newry, Preston, Truro, Bangor, Plymouth, Norwich. Alternatively using `["All"]` as a location trains on the whole UK.
A distinct modelcode string is created for each model based on the arguments used when during initialising of the training script.
Model Checkpoints are saved in a './checkpoints/modelcode' folder
Dictionary containing information on the model trained are saved in a './saved_params/modelcode' folder

## Prediction Scripts 
The code below can be used to make predictions provided you have trained a TRU-NET model using the previous script. The list that follows the code, explains the role of the arguments. For brevity only new arguments not present during training are detailed. 

`python3 predict.py -mn "THST" -ctsm "1998_2010_2010_2012" -ctsm_test "2012_2014" -mts "{'stochastic':False,'stochastic_f_pass':50,'distr_type':'Normal','discrete_continuous':True,'var_model_type':'mc_dropout','location':['Cardiff','London','Glasgow'], 'location_test':['Cardiff','London','Glasgow']}" -dd "/Data/Rain_Data_Mar20" -bs 13 -do 0.2 -ido 0.2 -rdo 0.3`

* ctsm = string : to identify which trained model to use.
* ctsm_test = string : date range to predict on. Format `teststart_testend` eg. "1979_1981"
*	distr_type = Normal/LogNormal: Distribution for rain in non CC distribution alternatively the continuous distribution (g) in CC distribution
*	location: list: to identify which trained model to use.
*	location_test: list: Locations to test on. If no value passed, the values for `location` is used

The above code will save a pickled tuple (predictions, true_values, timesteps) to a folder Output/modelcode
Currently, to retrieve scoring metrics the 'Visualization.ipynb' notebook must be used on the output. Examples of how to do so are included in the Visualization.ipynb script. In the scoring will be worked into the predict.py

## Evaluating ERA5 Predictions
To  Be Added

##Notes for Developers

##Requirements (Python 3 and Linux)
* Tensorflow 2
* Tensorflow-addons
* numpy
* pandas
* scipy
* matplotlib
* pupygrib
* xarray
* netCDF4 (Note: netCDF4 must be imported before TensorFlow in scripts)
* argparse

##License
`TRU-NET: Precipitation Prediction` is licensed under the MIT License
