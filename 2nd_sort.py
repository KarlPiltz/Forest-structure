# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 16:30:12 2022

@author: Karl.Piltz
"""

import arcpy
import os
#import gdal
#vector=ogr.Open(r'E:\2010\mask3.shp')
AOI=r'U:\masksw.shp' #Your area of intrest which is the location(s)/mask that you want imagery data for
rFolder=r'C:\Tempdata\2011' #input folder fro your images
oFolder=r'C:\Tempdata\2011_to_run' #output folder to where you copy images that match criteria
#desc=arcpy.Describe(AOI)
#sExt=desc.extent


#layer = vector.GetLayer()
#feature = layer.GetFeature(0)
#vectorGeometry = feature.GetGeometryRef()



listExt = [] 
with arcpy.da.SearchCursor(AOI, 'SHAPE@') as cursor: #looping through the shape extent of the mask layer
    for row in cursor:
        listExt.append(row[0].extent) #append the extent for each polygon to be tested againsd your image
        
arcpy.env.overwriteOutput = True
arcpy.env.workspace=rFolder

for ThisRas in arcpy.ListRasters(): #Loop through image folder
    rDesc = arcpy.Describe(ThisRas) #for each image describe the metadata and define the extent
    rExt = rDesc.extent 
    for sExt in listExt: #looping through list of extents from search cursor to test if image and mask is disjoint or no
        if sExt.disjoint(rExt):
            arcpy.AddMessage("Raster %s is outside" % (ThisRas))
        
        else: #copy images that are not disjoint from AOI
            arcpy.AddMessage("Raster %s overlaps" % (ThisRas)) #if not disjoint: 
            outFile = os.path.join(oFolder,ThisRas) #define output path by defining input/output directory
            arcpy.CopyRaster_management(ThisRas,outFile)