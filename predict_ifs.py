import netCDF4
from netCDF4 import Dataset, num2date
import data_generators
import argparse
import ast
import datetime as dt
import os
import pickle
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

"""Example of how to use

    For Evaluation of IFS predictive scores between the periods 1987-10-20 till 1989-11-20, for the single points representing region Cardiff and London:
        python3 evaluate.py -dd "./Data" -sd "1987-10-20" -ed "1989-11-20" -lo ["Cardiff","London"] -mth 3 

    For Evaluation of IFS predictive scores between the periods 1987-10-20 till 1989-11-20, for the regions representing region Cardiff and London:
        python3 evaluate.py -dd "./Data" -sd "1987-10-20" -ed "1989-11-20" -lo ["Cardiff","London"] -reg True

    For Evaluation of IFS predictive scores between the periods 1988-10-20 till 2000-11-20, for the whole UK:
        python3 evaluate.py -dd "./Data" -sd "1988-10-20" -ed "2000-11-20" -lo ["All"] -mth 3 -reg "False"

    For infomration on the True rainfall statistics for London between the periods 2008-01-20 till 2015-11-20
        python3 evaluate.py -dd "./Data" -sd "2008-01-20" -ed "2015-11-20" -lo ["London"] -mth 3 -rfs True -reg True
        -This returns information such as Average Rainfall, Percentage of R10 Events and Average rainfall given an R10 event occurs.

"""


def main(date_start_str, date_end_str, location, data_dir="./", rain_fall_stats=False, region=False ):
    """
        :str date_start: start evaluation date as a string in the following format YYYY-MM-DD
        :str date_end: end evaluation date as a string in the following format YYYY-MM-DD
        :str location: Location to evaluate, pass "All to evaluate whole country
        :bool rain_fall_stats: boolean indicating whether to return statistics explaining the true rainfall of a region
    """
    
    date_start = np.datetime64(date_start_str,'D')
    date_end = np.datetime64(date_end_str,'D')
    
    #Extract the IFS Predictions for a given location and time range
    ifs_preds = ifs_pred_extractor(data_dir, date_start, date_end, location, region )
    
    #Extract the True rainfall for a given location and time range
    true_rain, rain_mask = true_rain_extractor( data_dir, date_start, date_end, location, region )

    #Model fields extractor
    mf = model_field_extractor(data_dir, date_start, date_end, location, region )

    #Creating a list of the epoch timestamps relating to the days we study. i.e. if we tested from 1978-01-20 till 2000-03-01 
        # this would be a list such as [254102400 ,......,  951868800]
    date_tss = pd.date_range( end=date_end, start=date_start, freq='D',normalize=True)
    timestamp_epochs =  list ( (date_tss - pd.Timestamp("1970-01-01") ) // pd.Timedelta('1s') )
    
    #Inserting nans in masked values
    true_rain = np.where(rain_mask, true_rain, np.nan )
    ifs_preds = np.where(rain_mask, ifs_preds, np.nan )
    
    #Save the extracted IFS prediction, true rainfall for optional Visualization using Evaluation.ipynb
    preds = ifs_preds
    true_rain = true_rain
    f_dir = "./Output/ERA5/preds/"
    fn = "{}_{}_{}".format(location, date_start_str, date_end_str)
    if region ==True:
        fn+= "regional"
    fn += "_pred.dat"
    fp = f_dir+fn
    if not os.path.isdir(f_dir):
        os.makedirs( f_dir, exist_ok=True  )

    pickle.dump( [np.array(timestamp_epochs), np.array(ifs_preds,dtype=np.float64) , np.array(true_rain) ], 
        open(fp,"wb")  )
    
    #Version that also saves associated model field data
    f_dir1 = "./Output/ERA5/preds_w_mf/"
    fn1 = "{}_{}_{}".format(location, date_start_str, date_end_str)
    if region ==True:
        fn1+= "regional"
    fn1 += "_pred.dat"
    fp1 = f_dir1+fn1
    if not os.path.isdir(f_dir1):
        os.makedirs( f_dir1, exist_ok=True  )

    _dat = [np.array(timestamp_epochs), np.array(ifs_preds,dtype=np.float64) , np.array(true_rain), mf ]
    _dat = {
        'timestamps':np.array(timestamp_epochs),
        'ERA5':np.array(ifs_preds,dtype=np.float64),
        'true_rain':np.array(true_rain),
        'model_field':mf
    }
    pickle.dump( _dat , open(fp1,"wb")  )
    

    #Create a Plot of IFS predictions against True Rain values, for a quick check if I have aligned the IFS prediction and True rain correctly
    if location != "All" and region==False:
        plot_ifs_preds( ifs_preds, true_rain, date_start, date_end, data_dir, location  ) 
    
    # Masking out data which is either invalid or does not represent land (e.g. the sea)
    true_rain = np.float64( true_rain[rain_mask] )
    ifs_preds = np.float64( ifs_preds[rain_mask]  )

    # Option to output simply the rainfall statistics of an area or to evaluate IFS predictions on this area
    if rain_fall_stats == False:
        #IFS predictive performance metrics
        rmse = rmse_aggregate( ifs_preds, true_rain )
        r10_rmse = r10rmse_aggregate(ifs_preds, true_rain )
        _dataframe = pd.DataFrame( {'RMSE':[rmse],"R10_RMSE":[r10_rmse] } )
        
        #Saved scores to file in scores sub-directory
        f_dir = "./Output/ERA5/scores"
        
        #if not os.path.isdir(f_dir):
        os.makedirs( f_dir, exist_ok=True  )

        fn = "{}_{}till{}_scores.csv".format(location,date_start_str, date_end_str)
        fp = f_dir+"/"+fn
        _dataframe.to_csv( fp, index=False)
        print(_dataframe)

    else:
        #Rainfall statistics of an area.
        avg_precip = np.nanmean(true_rain)
        days_r10 = np.size(true_rain[true_rain>=10]) / np.size(true_rain)
        avg_precip_r10 = np.nanmean( true_rain[true_rain>=10] )
        _dataframe = pd.DataFrame( { 'Avg_precip':[avg_precip], 'days_r10':[days_r10*100], "Avg_precip_r10":[avg_precip_r10] })

        print(_dataframe)

    return True
    
def ifs_pred_extractor( data_dir, target_start_date, target_end_date, location="London", region=False ):
    """
        This method extracts the IFS data from file. And performs any grouping/preproc neccesary. 
    """
    ifs_fn = data_dir + "/IFS_precip_preds/ana_tp_1234.grib"

    str_start_date = "1979-01-02"
    str_end_date = '2019-12-31'

    ifs_start_date = np.datetime64(str_start_date,'D') #IFS preds starts from 1979-01-02
    ifs_end_date = np.datetime64(str_end_date,'D')
    if target_end_date >ifs_end_date  or target_start_date < ifs_start_date:
        #rejects user request for results outside of relevant time span
        raise ValueError(f"Invalid Datespan, please stick within range: {str_start_date} to {str_end_date}")
        
    #Extracting and reshapping and IFS Data
    ifs_preds_24hr = pickle.load( open(ifs_fn,"rb"))
    ifs_preds_24hr = ifs_preds_24hr[:, 2:-2, 2:-2] 

    #Calculating start and end index at which to slice the IFS data.
    cut_idx_s = np.timedelta64( target_start_date - ifs_start_date, 'D' ).astype(int) 
    cut_idx_e = np.timedelta64( target_end_date - ifs_start_date, 'D' ).astype(int) + 1
    ifs_preds_24hr = ifs_preds_24hr[cut_idx_s:cut_idx_e]

    #Scaling the IFS data to match that of the true  precip observations
    ifs_preds_24hr = ifs_preds_24hr * 1000

    #If a location is passed, this extracts the single point ,representing a city, from the 100,140 map 
    ifs_preds_24hr = data_craft(ifs_preds_24hr, location, region)
        
    return ifs_preds_24hr

def true_rain_extractor(data_dir, target_start_date, target_end_date, location, region):
    
    # Valid Date Check
    rain_start_date = np.datetime64('1979-01-01','D')
    rain_end_date = np.datetime64('2019-08-01','D')
    if target_end_date >=rain_end_date  or target_start_date < rain_start_date:
        raise ValueError("Invalid Datespan, please stick within range: {rain_start_date} to {rain_end_date}")
        
    #Extracting from NETCDF4 file 
    fn_rain_Mar =   data_dir+ "/Rain_Data_Mar20/rr_ens_mean_0.1deg_reg_v20.0e_197901-201907_uk.nc"
    dataset_rain =  Dataset( fn_rain_Mar,'r', format="NETCDF4")
    data_rain =     dataset_rain.variables['rr'][:]

    #Selecting Only relevant time period
    cut_idx_s =     np.timedelta64( target_start_date - rain_start_date, 'D' ).astype(int)
    cut_idx_e =     np.timedelta64( target_end_date - rain_start_date, 'D' ).astype(int) + 1
    data_rain =     data_rain[cut_idx_s:cut_idx_e, :, :]
    
    #Aligning Data. The rain data is essential upside down
    data_rain = data_rain[:, ::-1, :]
        
    #Selecting location of interest
    data_rain = data_craft( data_rain, location, region)
    
    #Selecting the rain_mask and actual rain_data
    rain_mask = np.logical_not( np.ma.getmask(data_rain) )
    data_rain = np.ma.getdata( data_rain )
    
    return data_rain, rain_mask

def model_field_extractor(data_dir,target_start_date, target_end_date, location, region ):
   # Valid Date Check
    feature_start_date = np.datetime64('1970-01-01') + np.timedelta64(78888, 'h')
    feature_end_date  = np.datetime64( feature_start_date + np.timedelta64(59900, '6h'), 'D')

    if target_end_date >=feature_end_date  or target_start_date < feature_start_date:
        raise ValueError("Invalid Datespan, please stick within range: {feature_end_date} to {feature_start_date}")    

    #Instatiating model field data generator
    fn_mf =  os.path.join(data_dir,'Rain_Data_Mar20',"ana_input_intrp_linear.nc")

    vars_for_feature = ['unknown_local_param_137_128', 'unknown_local_param_133_128', 'air_temperature', 'geopotential', 'x_wind', 'y_wind' ]
    all_at_once = True
    seq_len =     np.timedelta64( target_end_date - target_start_date, '6h' ).astype(int)
    
    mf_data_gen = data_generators.Generator_mf(fp=fn_mf, vars_for_feature=vars_for_feature, 
                all_at_once=all_at_once, seq_len=seq_len )
    
    #Extracting dta
    mf_data_gen.start_idx = np.timedelta64(target_start_date - feature_start_date,'6h').astype(int)
    mf_data_gen.end_idx = np.timedelta64(target_end_date - feature_start_date,'6h').astype(int) + 4

    mf_data = mf_data_gen()
    
    #Cropping spatial bounds of data t
    mf_array = data_craft(mf_data, location, region, mf=True)

    return mf_array

def data_craft( data, location, region=False, mf=False ):
    # location of cities/regions of interest
    city_latlon = {
        "London": [51.5074, -0.1278],
        "Cardiff": [51.4816 + 0.15, -3.1791 -0.05], #1st Rainiest
        "Glasgow": [55.8642,  -4.2518], #3rd rainiest
        "Lancaster":[54.466, -2.8007], #2nd hieghest
        "Bradford": [53.7960, -1.7594], #3rd highest
        "Manchester":[53.4808, -2.2426], #15th rainiest
        "Birmingham":[52.4862, -1.8904], #25th
        "Liverpool":[53.4084 , -2.9916 +0.1 ], #18th rainiest
        "Leeds":[ 53.8008, -1.5491 ], #8th
        "Edinburgh": [55.9533, -3.1883],
        "Belfast": [54.5973, -5.9301], #25
        "Dublin": [53.3498, -6.2603],
        "LakeDistrict":[54.4500,-3.100],
        "Newry":[54.1751, -6.3402],
        "Preston":[53.7632, -2.7031 ],
        "Truro":[50.2632, -5.0510],
        "Bangor":[54.2274 - 0, -4.1293 - 0.3],
        "Plymouth":[50.3755 + 0.1, -4.1427],
        "Norwich": [52.6309, 1.2974],
        "StDavids":[51.8812+0.05, -5.2660+0.05] ,
        "Swansea":[51.6214+0.05,-3.9436],
        "Lisburn":[54.5162,-6.058],
        "Salford":[53.4875, -2.2901],
        "Aberdeen":[57.1497,-2.0943-0.05],
        "Stirling":[56.1165, -3.9369],
        "Hull":[53.7676+0.05, 0.3274]
        }

    #Selects the closest grid point to the location of the city
    if location in city_latlon.keys():
        latitude_array = np.linspace(58.95,49.05, 100)
        longitude_array = np.linspace(-10.95, 2.95, 140)
        coordinates = city_latlon[location]
        
        latitude_index =    np.abs(latitude_array - coordinates[0]).argmin()
        longitude_index =   np.abs(longitude_array - coordinates[1]).argmin()

        if region==False and mf==False:
            data = data[:,latitude_index, longitude_index]

        if region==False and mf==True:
            slice_lat = slice(latitude_index,latitude_index+1)
            slice_lon = slice(longitude_index,longitude_index+1)
            data = data.isel(latitude=slice_lat,longitude=slice_lon)
            
            data = data.to_array()
            data = data.transpose('time','latitude','longitude','variable').values

        elif region == True and mf==False:
            data = data[:,latitude_index-2:latitude_index+2, longitude_index-2:longitude_index+2].astype(np.float64)

        elif region == True and mf==True:
            slice_lat = slice(latitude_index-8,latitude_index+8)
            slice_lon = slice(longitude_index-8,longitude_index+8)
            data = data.isel(latitude=slice_lat,longitude=slice_lon)
            
            data = data.to_array()
            data = data.transpose('time','latitude','longitude','variable').values

    elif location == "All":
        pass
    else:
        raise ValueError("Invalid Location")
        
    return data

def plot_ifs_preds( ifs_preds, true_val, date_start, date_end, data_dir,loc):
    """"
        Creates a plot of IFS preds against rain precipitation
    """
    ts_formated = [ d.strftime("%d-%m-%y") for d in pd.date_range(start=date_start, end=date_end, freq='D' ) ] 
    
    start = pd.to_datetime( date_start)
    end = pd.to_datetime( date_end + np.timedelta64(1,'D') )

    ts = mdates.drange(start,end,dt.timedelta(days=1))

    fig, (ax) = plt.subplots(1)
    ax.plot(ts, true_val,color='black',linewidth=1)    
    ax.plot(ts, ifs_preds.reshape([-1]), color='blue')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=20) )
            
    img_dir = "./Output/ERA5/preds/Illustrations"
    fn_name = "{}_{}_{}.png".format(loc,ts_formated[0], ts_formated [-1])
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        
    plt.savefig(img_dir+"/"+fn_name )

def rmse_aggregate( preds_mean, true_vals):
    return np.sqrt( np.square(np.subtract(preds_mean, true_vals)).mean() )

def r10rmse_aggregate(preds_mean ,true_vals, N=10):
    """ Returns the RN rmse, by default N = 10"""
    mask_r10 = true_vals >= N
    if np.count_nonzero(mask_r10) == 0:
        return np.NaN
    preds_filt = preds_mean[mask_r10]
    true_vals_filtr = true_vals[mask_r10]
    
    return np.sqrt(np.mean((preds_filt-true_vals_filtr)**2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receive input params")

    parser.add_argument('-dd','--data_dir', type=str, help='the directory for the Data', required=False, default="./Data/")
    
    parser.add_argument('-sd','--date_start_str', type=str, required=True)

    parser.add_argument('-reg','--region', type=eval, required=False, default='True', choices=[True,False])

    parser.add_argument('-ed','--date_end_str', type=str, required=False, default='2019-07-31')

    parser.add_argument('-lo','--location', type=str, required=True, default="['London']", help="List of locations to evaluation on")
    
    parser.add_argument('-rfs','--rain_fall_stats', type=bool, required=False, default=False, help="Pass True to return statistics on the true rainfall of an for the areas of interest")

    args_dict = vars(parser.parse_args() )

    li_loc = ast.literal_eval( args_dict.pop('location') )
    for loc in li_loc:
        print(f"Evaluating {loc}")
        main( location=loc, **args_dict )
        print("\n")
