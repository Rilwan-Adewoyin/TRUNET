# TRU-NET: Precipitation Prediction

Code for predicting precipitation using model field data from IFS-ERA5 (temperature, geopotential, wind velocity, etc.) as predictors for sub-areas across the British Isle.
The Data used in this project is available at the following Google Drive link. Users must download and decompress the Rain_Data_Mar20 folder. 
Keywords: TRU_NET, Downscaling, Cross Attention, Hierarchical GRU

## Getting Started / Recreate
The instructions below include commands to enter in command line to get started quickly and train your own model that takes in model field data and outputs rain prediction: 

1.  make directory on your computer where you want to create this project 
    *  mkdir Project_RainForecast && cd Project_RainForecast
2.  Download / Fork the repository
    *  git clone https://github.com/Akanni96/TRUNET.git 
    *  mkdir TRUNET/Data
3.  make python >3.6 virtual environment and install requirements
    *  python -m venv venv_trunet
    *  source venv_trunet/bin/activate
    *  pip3 install -r TRUNET/requirements.txt
    *  pip3 install tensorflow-gpu==2.3
4.  Download and Extract preprocessed Data Files
    *  gdown https://drive.google.com/uc?id=1543TTVz6gAGjpZ4lTqyVX_r0aa3jJAbm
    *  unzip Data.zip -d Data
5.  Train on 1979-2009, Predict 2009-2014, Evaluate Performance
    *  Use example codes in the following train, predict and Evaluate section
  
#### train
`python3 train.py -mn "TRUNET" -ctsm "1979_2009_2014" -mts "{'stochastic':False,'stochastic_f_pass':1,'discrete_continuous':True,'var_model_type':'mc_dropout','do':0.2,'ido':0.2,'rdo':0.3,'location':['London','Cardiff','Glasgow','Lancaster','Bradford','Manchester','Birmingham','Liverpool','Leeds','Edinburgh','Belfast','Dublin','LakeDistrict','Newry','Preston','Truro','Bangor','Plymouth','Norwich','StDavids','Swansea','Lisburn','Salford','Aberdeen','Stirling','Hull']}" -dd "./Data" -bs 64`

#### predictions
`python3 predict.py -mn "TRUNET" -ctsm "1979_2009_2014" -ctsm_test "2014_2019-07-04" -mts "{'stochastic':True,'stochastic_f_pass':25,'discrete_continuous':True,'var_model_type':'mc_dropout', 'do':0.2,'ido':0.2,'rdo':0.3, 'location':['London','Cardiff','Glasgow','Lancaster','Bradford','Manchester','Birmingham','Liverpool','Leeds','Edinburgh','Belfast','Dublin','LakeDistrict','Newry','Preston','Truro','Bangor','Plymouth','Norwich','StDavids','Swansea','Lisburn','Salford','Aberdeen','Stirling','Hull'], 'location_test':['All']}" -ts "{'region_pred':True}" -dd "./Data" -bs 71`

Note: The maximum batch size that can be used during prediction is proportional to number of days covered by the test set. As explained in the paper, one prediction from TRUNET is one sequence of 28 values relating to the rain in the corresponding 28 days. As such each batch of predictions will cover 28 days * batch size. Using the prediction code from above as an example, each geographic region under evaluation has predictions produced for the time period 2014 till 2019-07-04. This is equal to 2010 days. When predicting for a single region, the upper limit on batch size should be 2010/28 days. This is roughly equal to 71.79. 

#### evaluation
Use the Evaluation.ipynb


## Training Scripts
The code below can be used to train a TRU-NET model. The list the follows the code explains the role of the arguments.

`python3 train.py -mn "THST" -ctsm "1998_2010_2012" -mts "{'stochastic':False,'stochastic_f_pass':1,'discrete_continuous':True,'var_model_type':'mc_dropout','do':0.2,'ido':0.2,'rdo':0.3,'location':['Cardiff','London','Glasgow']}" -dd "./Data" -bs 64`

* mn = string : model_name, can be THST or HCGRU
* ctsm = string : dates for training and validation in format `trainstart_trainend/valstart_valend` eg. "1979_1992_1995"
* mts = dictionary :specific settings
*   var_model_type = str: Whether to train BNN or deterministic model `mc_dropout` or `Deterministic`
*   stochastic = Bool: To train model with multiple forward passes
*	  stochastic_f_pass = int : how many forward passes if stochastic=True
*	  discrete_continuous = Bool: Whether or not to use CC distribution
*   do = float : dropout
*   ido = float : input_dropout (Dropout to input parts of RNN based layers)
*   rdo = float : recurrent_dropout (Dropout to recurrent parts of RNN based layers)
*	  location = list: Locations to train on. To train on whole UK use `["All"]`
* dd = string : data directory
* bs = int : batch size

A distinct modelcode string is created for each model based on the arguments used when during initialising of the training script. This modelcode is utilised when saving results, models, illustrations related to any given model.

Model Checkpoints are saved in a './checkpoints/modelcode' folder
Dictionary containing information on the model trained are saved in a './saved_params/modelcode' folder

Locations can be chosen from the following list: London, Cardiff, Glasgow, Lancaster, Bradford, Manchester, Birmingham, Liverpool, Leeds, Edinburgh, Belfast, Dublin, LakeDistrict, Newry, Preston, Truro, Bangor, Plymouth, Norwich. Alternatively using `["All"]` as a location trains on the whole UK.

## Prediction Scripts 
The code below can be used to make predictions provided you have trained a TRU-NET model using the previous script. The list that follows the code, explains the role of the arguments. For brevity only new arguments not present during training are detailed. 

`python3 predict.py -mn "TRUNET" -ctsm "1998_2010_2012" -ctsm_test "2012_2014" -mts "{'stochastic':True,'stochastic_f_pass':25,'discrete_continuous':True,'var_model_type':'mc_dropout', 'do':0.2,'ido':0.2,'rdo':0.3, 'location':['Cardiff','London','Glasgow'], 'location_test':['Cardiff','London','Glasgow']}" -ts "{'region_pred':True}" -dd "./Data" -bs 71`

* ctsm = string : to identify which trained model to use.
* mts = dictionary :specific model settings
*   stochastic = Bool : To predict with multiple forward passes
*   ctsm_test = string : date range to predict on. Format `teststart_testend` eg. "1979_1981"
*	  location: list: to identify which trained model to use.
*	  location_test: list: Locations to test on. If no value passed, the values for `location` is used. To test on whole UK use `["All"]`
* ts = Bool : specific test settings
*   region_pred = Bool : If True predict on the 16 by 16 stencil around a region. If False predict for the single point within the 16 by 16 region

After running the predict sript, a pickled tuple (predictions, true_values, timesteps) will be saved to a folder './Output/modelcode/Predictions'. If region_pred=True this file will be called region.dat, if region_pred=False this file will be called local.dat

## Evaluation Scripts 
Currently, to retrieve any scoring metrics or illustrations the 'Evaluation.ipynb' notebook must be used. Examples of how to do so are included in the file Visualization.ipynb. The output from Evaluation.ipynb is generally saved to the './Output/Experiments/' directory. 

## Producing IFS-ERA5 Predictions
IFS-ERA5 is the numerical weather system which is used as the baseline in our paper. The code below can be used to create prediction results from the IFS-ERA 5 model

`python3 predict_ifs.py -dd "./Data" -sd "2014" -ed "2019-06-11" -lo "['All']" -reg False`

* dd = string : data directory (Note: do not use a child directory)
* sd = string: start_date for predictions. Can be of the form 'YYYY' or 'yy-mm-dd'.
* ed = string: end_date for predictions. Can be of the form 'YYYY' or 'yy-mm-dd'.
* location = list: Locations to extract prediction for. To train on whole UK use `["All"]`
* reg = Bool:  If True extract predictions for the 16 by 16 stencil around a region. If False extract for the central point within the 16 by 16 region

Predictions will again be saved in the .Output/Predictions file.

## Data Download
The preprocessed data used for experiments related to the paper can be found at this link https://drive.google.com/file/d/1543TTVz6gAGjpZ4lTqyVX_r0aa3jJAbm/view?usp=sharing. Users must extract the contents from the zip folder, into the root directory associated with their TRUNET repository. This Data contains 6-hourly data for 6 model fields defined on a 100,140 grid over the UK for the years 1979 through to 2019. 

* IFS rain predictions between 1979 and 2019 defined on a 102,142 grid over the UK
* Eobs true rainfall between 1979 to 2019 (netcdf4)
* model field feature data between 1979 to 2019 defined on a 16x20 (netcdf4)
* model field feature data between 1979 to 2019, upscaled to 100x140 grid over the UK (netcdf4)

  contains 6-hourly data for 6 model fields defined on a 100,140 grid over the UK for the years 1979 through to 2019.

To download the preprocessed IFS precipitation data which forms a benchmark for our paper, please download the following datafile: https://drive.google.com/file/d/1543TTVz6gAGjpZ4lTqyVX_r0aa3jJAbm/view?usp=sharing. This contains 12 hourly predictions for rainfall over the UK for the years 1979 through to 2020.

**Problem**: 
- When you try to download a file with .rar extension from google drive and you get access denied error. 

**Solution**: 
- you can fix the error by opening a new private/incognito window.
- Copy the url of the file and paste that url in new private window. 
- Hit enter the file will automatically start to download.

## Key Requirements (Python 3 and Linux)
A requirements.txt file is included which manages most requirements. Below we list key requirments

* Python 3.6 
* Tensorflow 2.2 - 2.4 [ Depending on the cuda/cudnn versions present on your computer.]




# TODO: 
- Add a requirements.txt
- Make IFS data openly available
- Dockerize whole solution with automatic downloads of data
- Include a description of the IFS data

##License
`TRU-NET: Precipitation Prediction` is licensed under the MIT License
