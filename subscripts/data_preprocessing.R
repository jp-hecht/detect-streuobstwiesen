## ---------------------------
##
## Script name: data_preprocessing_rgb.R
##
## Purpose of script: Script to preprocess the data and create raw data for the next script
##
## Author: Jonathan Hecht
##
## Date Created: 2021-09-11
##
## Copyright: -
##
## Email: -
##
## ---------------------------
##
## Notes: Check paths dependencies
##  
##
## ---------------------------

## set working directory 

wd <- getwd()
setwd(wd)

## ---------------------------

options(warn = -1)

## ---------------------------

## load all necessary packages

library(rgdal)
library(raster)

##----------------------------

## functions

#-


# need to create sow_mask.tif for data_split as input
input_vector <- readOGR("./data/sow/ex_raw/neu/Streuobst_aus_HLBK_GGBT_Regelbetrieb.shp")

b2 <- raster("./data/sen_inp/WASP_sen_8_cro_he_c_1_99.tif")

ra <- rasterize(input_vector,b2)
ra[is.na(ra[])] <- 0
ra <- reclassify(ra,cbind(c(1:ra@data@max),1))
writeRaster(ra,"./data/sow/neu/sow_mask.tif",overwrite=T)














