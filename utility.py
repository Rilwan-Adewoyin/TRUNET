import numpy as np
import math

#precip data import - use in data pipeline
def read_prism_precip(bil_path, hdr_path=None, hdr_known=True, tensorf = True):
    """
        Read an array from ESRI BIL raster file using Info from the hdr file too
        https://pymorton.wordpress.com/2016/02/26/plotting-prism-bil-arrays-without-using-gdal/
    """
    if( tensorf ):
        bil_path = bil_path.numpy()

    if(hdr_known):
        NROWS = 621
        NCOLS = 1405
        NODATA_VAL = float(-9999)
    else:
        hdr_dict = read_hdr(hdr_path)
        NROWS = int(hdr_dict['NROWS'])
        NCOLS = int(hdr_dict['NCOLS'])
        NODATA_VAL = hdr_dict['NODATA']
    # For now, only use NROWS, NCOLS, and NODATA
    # Eventually use NBANDS, BYTEORDER, LAYOUT, PIXELTYPE, NBITS
 
    prism_array = np.fromfile(bil_path, dtype='<f4')
    prism_array = prism_array.astype( np.float32 )
    prism_array = prism_array.reshape( NROWS , NCOLS )
    prism_array[ prism_array == float(NODATA_VAL) ] = np.nan
    return prism_array

#prism data import
def read_prism_elevation(bil_path, hdr_path=None, hdr_known=True):
    if(hdr_known):
        NROWS = 6000
        NCOLS = 4800
        NODATA_VAL = float(-9999)
    else:
        hdr_dict = read_hdr(hdr_path)
        NROWS = int(hdr_dict['NROWS'])
        NCOLS = int(hdr_dict['NCOLS'])
        NODATA_VAL = int(hdr_dict['NODATA'])
    # For now, only use NROWS, NCOLS, and NODATA
    # Eventually use NBANDS, BYTEORDER, LAYOUT, PIXELTYPE, NBITS
 
    _array = np.fromfile(open(bil_path,"rb"), dtype='>i2')
    _array = _array.astype(dtype=np.float32)
    _array = _array.reshape( NROWS , NCOLS )
    _array[ _array == NODATA_VAL ] = np.nan
    return _array

def read_hdr(hdr_path):
    """Read an ESRI BIL HDR file"""
    with open(hdr_path, 'r') as input_f:
        header_list = input_f.readlines()
    return dict(item.strip().split() for item in header_list)
