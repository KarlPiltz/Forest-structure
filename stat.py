# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 12:58:06 2022

@author: Karl.Piltz
"""
import matplotlib.pyplot as plt
import arcpy
import pandas as pd
import numpy as np

bb='D:/Germany/Disturbance1_0.shp' #input of disturbance occurance location later used for masking cell statistics
rFolder=r'D:\Germany\torun\2009_1m_laea1.tif' #folder containing imegary
arcpy.env.overwriteOutput = True #Allow overwriting
arcpy.env.workspace=rFolder #set working environment to image input

pnt_array = arcpy.Array() #Empty array to contain occurance point data for active image
zstat=[] #Empty array to store cell statistics appended from masking
stat=['Max','Mean','Std','Disturbance'] #Names of stat types, will become column names when storing as dataframe


for ThisRas in arcpy.ListRasters():  #Using arcpy raster iteration to loop through input folder
    arcpy.env.extent = arcpy.Raster(ThisRas).extent #Set environment setting for extent to do operations within extent of current image
    arcpy.env.snapRaster = r"D:\Germany\mask.tif" #Snapping of raster so cell center overlaps perfectly (Based on disturbance data)
    temp=arcpy.RasterToNumPyArray(ThisRas) #Read raster as a numpy array to do operations on
    temp = temp*1.0 #Convert to Float
    temp[temp == -1] = np.nan #Set no data to nan since Numpy stats can ignore no data if represented by nan
    cols, rows = np.shape(temp) #Get shape (cols and rows) of current image
    Rrem = rows % 30 #Get how many times a 30m window can slide over the rows
    Crem = cols % 30 #Get how many times a 30m window can slide over the cols
    newrows = rows #New variable containing rows and column which will be used to make jagged raster even
    newcols = cols
    if Rrem != 0:  
        rad=np.full((cols,(30-Rrem)),np.nan) #If the moving window is not able to move over whole image store how many rows and columns extentions are needed
        newrows = (rows+(30-Rrem))
    if Crem != 0:
        col=np.full(((30-Crem),newrows),np.nan)
        newcols = (cols+(30-Crem))
    t_r = int((newrows / 30)) #Variable for the how many movement the window will do to cover whole image
    t_c = int((newcols / 30))
    t_max=np.zeros([t_c,t_r]) #Create empty arrays size of window movments to store the stats in 30m pxls
    t_mean=np.zeros([t_c,t_r])
    t_std=np.zeros([t_c,t_r])
    temp=np.c_[ temp, rad ] #Extent image array with the amount of rows and collumns needed for window to cover whole image
    temp=np.r_[temp, col]
    for r,k in zip(range(14,newcols,30),range(0,(newcols // 30))): #The offset of rows and columns in movement, jump of 15 steps from center pixel
        for c,p in zip(range(14,newrows,30),range(0,(newrows // 30))):
            t_max[k][p]=np.nanmax(temp[r-14:r+16,c-14:c+16]) #Append each center pixel to variable
            t_mean[k][p]=np.nanmean(temp[r-14:r+16,c-14:c+16])
            t_std[k][p]=np.nanstd(temp[r-14:r+16,c-14:c+16])
                
    arcpy.conversion.PointToRaster(bb, "Dist", "D:/Germany/test.tif", "MOST_FREQUENT", "", "30" ) #Convert point occurance data to raster to be used as mask
    bb_mask = arcpy.RasterToNumPyArray("D:/Germany/test.tif", "", t_r, t_c) #Create mask of same size as new images in 30x30m
    mask = bb_mask >= 0 #Final mask to be used True= where cells are either 0 or 1 e.i. contining data
    st_max = mask*t_max #Mask out and append cell statistics where mask = True
    st_mean =mask*t_mean
    st_std =mask*t_std
    arcpy.Delete_management("D:/Germany/test.tif") #Delete temporary damage raster as it is different for each iteration
    ap1=np.where(bb_mask >=0) #Store where data is 0 or 1
    for a,b in zip(ap1[0],ap1[1]): #Loop through new variables and append cell statistics where 0,1 is present
        zstat.append([
            t_max[a,b],
            t_mean[a,b],
            t_std[a,b],
            bb_mask[a,b]])

stats=pd.DataFrame(zstat,columns=stat) # Convert to pd.dataframe using the column names
#stats.to_csv(path+"\\"+name+".csv",index=False) # Finally export as csv or xlsx in order to save it outside of python
#stats.to_excel("D:/stat_LU/Ger_stat.xlsx")
  
 
# =============================================================================
# =============================================================================
# #             source_ds = ogr.Open(point)
# # =============================================================================
#             source_layer = source_ds.GetLayer()
#     # 2) Creating the destination raster data source
# 
#         pixelWidth , pixelHeight = 30 # depending how fine you want your raster
#         x_min, x_max, y_min, y_max =  temp.GetExtent()
#         cols = int((x_max - x_min) / pixelHeight)
#         rows = int((y_max - y_min) / pixelWidth)
#         target_ds = gdal.GetDriverByName('GTiff').Create(raster_path, cols, rows, 1, gdal.GDT_Byte) 
#         target_ds.SetGeoTransform((x_min, pixelWidth, 0, y_min, 0, pixelHeight))
#         band = target_ds.GetRasterBand(1)
#         NoData_value = -1
#         band.SetNoDataValue(NoData_value)
#         band.FlushCache()
# 
#     # 4) Instead of setting a general burn_value, use optionsand set it to the attribute that contains the relevant unique value ["ATTRIBUTE=ID"]
#         gdal.RasterizeLayer(target_ds, [1], source_layer, options = ['ATTRIBUTE=b_beetle'])
#     
#     # 5) Adding a spatial reference
#         target_dsSRS = osr.SpatialReference()
#         target_dsSRS.ImportFromEPSG(3035)
#         target_ds.SetProjection(target_dsSRS.ExportToWkt())
#         return gdal.Open(raster_path).ReadAsArray()
#         np.ma.masked_where(bb = 1 or bb =0, t_max)
#         np.ma.masked_where(bb = 1 or bb =0, t_mean)
#         np.ma.masked_where(bb = 1 or bb =0, t_std)
# =============================================================================
            
            #stat.append(pd.DataFrame(t_max, columns=stat.Max), ignore_index=True)
            #stat = stat.append(pd.DataFrame(t_mean, columns=stat.Mean), ignore_index=True)
            #stat = stat.append(pd.DataFrame(t_std, columns=stat.Std), ignore_index=True)