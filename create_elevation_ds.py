import numpy as np
import math
import utility
import glob
import pickle

def main():
    """
        This script saves a numpy array representing the elevation data on a 1km by 1km grid with the same boundaries as the untouched precipitation files
    """
    # reading in elevation files
    elev_bil_paths = list( glob.glob("Data/USGS/elevation/*/*.DEM") )
    elev_bil_paths.sort(reverse=True)
    #order is w180n90, w180n40, W140n90, W140N40, W100N90, W100N40
    
    #joining seperate maps
    li_elevation_maps = [ utility.read_prism_elevation(_bil_path) for _bil_path in elev_bil_paths]
    arr_elevation_map_north = np.concatenate( (li_elevation_maps[0],li_elevation_maps[2],li_elevation_maps[4]),axis=1 )
    arr_elevation_map_south = np.concatenate( (li_elevation_maps[1],li_elevation_maps[3],li_elevation_maps[5]),axis=1 )
    arr_elevation_map = np.concatenate( (arr_elevation_map_north, arr_elevation_map_south),axis=0 )

    #Values from HDR files
    ELEV_TOP_LEFT_x = -179.99583333333334
    ELEV_TOP_LEFT_y = 89.99583333333334
    ELEV_dim_x = 0.00833333333333 #dimension of a pixel
    ELEV_dim_y = 0.00833333333333
    ELEV_dim_NO_ROWS = 6000*2
    ELEV_dim_NO_COLS = 4800*3

    PRECIP_TOP_LEFT_x = -125
    PRECIP_TOP_LEFT_y = 49.9166666666664
    PRECIP_dim_x = 0.0416666666667 #dimension of a pixel
    PRECIP_dim_y = 0.0416666666667
    PRECIP_dim_NO_ROWS = 621
    PRECIP_dim_NO_COLS = 1405

    # geographical borders for precipitation data
    precip_left_border = PRECIP_TOP_LEFT_x - PRECIP_dim_x/2
    precip_right_border = precip_left_border + (PRECIP_dim_NO_COLS+1)*PRECIP_dim_x
    precip_top_border = PRECIP_TOP_LEFT_y + PRECIP_dim_y/2
    precip_bottom_border = precip_top_border - (PRECIP_dim_NO_ROWS+1)*PRECIP_dim_y

    #geographical borders for elevation data
    elev_left_border = ELEV_TOP_LEFT_x - ELEV_dim_x/2
    elev_right_border = elev_left_border + (ELEV_dim_NO_COLS+1)*ELEV_dim_x
    elev_top_border = ELEV_TOP_LEFT_y + ELEV_dim_y/2
    elev_bottom_border = elev_top_border - (ELEV_dim_NO_ROWS+1)*ELEV_dim_y

    #Calculating how many cells to crop for up, left, down, right
    elev_left_cells_to_crop = abs(elev_left_border - precip_left_border) / ELEV_dim_x
    elev_right_cells_to_crop = abs( elev_left_border-precip_right_border ) / ELEV_dim_x

    elev_top_cells_to_crop = abs( elev_top_border - precip_top_border) / ELEV_dim_y
    elev_bottom_cells_to_crop = abs(elev_top_border - precip_bottom_border) / ELEV_dim_y

    elev_left_cells_to_crop =  math.ceil(elev_left_cells_to_crop)
    elev_right_cells_to_crop = math.ceil(elev_right_cells_to_crop)

    elev_top_cells_to_crop = math.ceil(elev_top_cells_to_crop)
    elev_bottom_cells_to_crop = math.ceil(elev_bottom_cells_to_crop)

    arr_elevation_map = arr_elevation_map[ elev_top_cells_to_crop:elev_bottom_cells_to_crop,
                                        elev_left_cells_to_crop:elev_right_cells_to_crop]

    #reducing resolution to match 4km by 4km precipitation data
    ##Precip data has images of shape b=(621,1405) with each reading circa 0.9km
    ## arr_elevation_map has shape a=(3110,7030)
    ## np.array(a)/np.array(b) = array([5.00805153, 5.00355872])

    arr_elevation_map = arr_elevation_map[::5,::5] # shape (622, 1406)
    
    arr_elevation_map = arr_elevation_map[1:,1:] # shape (621, 1405)

    #np.save( "Data/Preprocessed/elevation.npy", arr_elevation_map, fix_imports=False, allow_pickle=False )
    with open("Data/Preprocessed/elevation.pkl","wb") as f:
        pickle.dump(arr_elevation_map, f, protocol=pickle.HIGHEST_PROTOCOL )


if __name__ == "__main__":
    main()
