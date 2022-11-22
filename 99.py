# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 09:57:25 2022

@author: Karl.Piltz
"""

import numpy as np
#import gdal
from osgeo import gdal
import os
from osgeo.gdalconst import *
gdal.AllRegister()
rasterfolder = r"D:\Germany\torun" #input folder containing images
raster = []
prc=0; #temp variable to hold the 99th percentile value for each image
for root, folder, files in os.walk(rasterfolder): #loop through the raster folder
    for file in files:
        if file.endswith('.tif'): #for every file with the extention .tif ... action
           #fullname = os.path.join(root, file)
            
            a=(root+ "\\" + file) #read raster as an array and writing new raster using a gdal driver
            inDs = gdal.Open(a)
            band1 = inDs.GetRasterBand(1) #get the raster band that you want to work with more intresting if RGB
            #dtype=gdal.GetDataTypeName(inDs.GetRasterBand(1).DataType)
            rows = inDs.RasterYSize #define output raster size, same as the input image
            cols = inDs.RasterXSize
            myarray = band1.ReadAsArray(0,0,cols,rows) #make the raster band into numpy array to work on
            driver = inDs.GetDriver() #create driver to write new raster with implemented changes
            outDs = driver.Create("D:/torun/ger99/" + file, cols, rows, 1, gdal.GDT_Int16) #define output raster using properties from the imput image
            outBand = outDs.GetRasterBand(1)
            #outData = numpy.zeros((rows,cols), numpy.int16)
            prc=np.percentile(myarray, 99) #find the 99th percentile value for each image
            myarray[myarray > prc] = -1 #set every value larger than the 99th to no data
            outBand.WriteArray(myarray, 0, 0)
            outBand.FlushCache()
            outBand.SetNoDataValue(-1) 
            outDs.SetGeoTransform(inDs.GetGeoTransform())
            outDs.SetProjection(inDs.GetProjection())