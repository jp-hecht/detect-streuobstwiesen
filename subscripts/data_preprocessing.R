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
## Notes: This script is not fully automated; therefore some manual changes need to be made 
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
library(gdalUtils)

##----------------------------

## functions

#-

# Reclassify the mask to 0 and 1 ------------------------------------------


# need to create sow_mask.tif for data_split as input
input_vector <- readOGR("./data/sow/ex_raw/neu/Streuobst_aus_HLBK_GGBT_Regelbetrieb.shp")

b2 <- raster("./data/dop40/area2/area_2.tif")

ra <- rasterize(input_vector,b2[[1]])
ra[is.na(ra[])] <- 0
ra <- reclassify(ra,cbind(c(1:ra@data@max),1))
plot(ra)

writeRaster(ra,"./data/dop40/area2/sow_mask.tif",overwrite=T)
ra <- raster("./data/dop40/area2/sow_mask.tif")



############### Test DOP
input_vector <- readOGR("./data/sow/test_mar_build.gpkg")

ras <- raster("./data/sow/test_mar_dop.tif")

ra <- rasterize(input_vector,ras)
ra[is.na(ra[])] <- 0
ra <- reclassify(ra,cbind(c(1:ra@data@max),1))
writeRaster(ra,"./data/sow/test_mar_mask.tif",overwrite=T)

# Some testing for the sentinel-2 images (not fully working!) ----------------------------------

ra <- raster("./data/raw_sen/2/SENTINEL2X_20200515-000000-000_L3A_T32UMA_C_V1-2_FRC_B2.tif")

ra_s <- stretch(ra)

s_ra <- scale(raster("./data/raw_sen/2/SENTINEL2X_20200515-000000-000_L3A_T32UMB_C_V1-2_FRC_B2.tif"))


# changes the names of the three training dop data sets for unique --------

# you have manually change the file paths and also delete some data beforehand
old_files <- "./data/dop40/area2/input224_10/mask"

new_files <- "./data/dop40/area2/input224_10/cop_m/2"

# list the png files in the folder

old_files <- list.files(old_files, pattern = "*.png", full.names = TRUE)
head(old_files)

# Create vector of new files

new_files <- paste0(new_files,1:length(old_files),".png")
head(new_files)

file.copy(from = old_files, to = new_files)

# Clear out the old files
file.remove(old_files)
























