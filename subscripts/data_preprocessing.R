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
library(gdalUtils)

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


ra <- raster("C:/Users/geoUniMarburg/Documents/detect-streuobstwiesen/data/raw_sen/2/SENTINEL2X_20200515-000000-000_L3A_T32UMA_C_V1-2_FRC_B2.tif")


ra_s <- scale(ra)


s_ra <- scale(raster("C:/Users/geoUniMarburg/Documents/detect-streuobstwiesen/data/raw_sen/2/SENTINEL2X_20200515-000000-000_L3A_T32UMB_C_V1-2_FRC_B2.tif"))


dir <- list.dirs("./data/raw_sen/",full.names = TRUE)

fil <- list.files(dir,pattern = "*.tif",full.names = TRUE)

# here some more preprocessing is needed
for(fi in fil){
   ras <- raster(fi)
   ras_s <- scale(ras)
   writeRaster(ras_s, sub(".tif",fi,replacement =  "_scaled.tif"))
}



# manually changed paths for each band
ras <- list.files("./data/raw_sen/8/",pattern ="_scaled.tif",full.names = TRUE)
ex <- extent(399960,609780,5390220,5800020) # found these values manually
tem <- raster(ex)
projection(tem) <- "+proj=utm +zone=32 +datum=WGS84 +units=m +no_defs"
writeRaster(tem,file = "./data/raw_sen/8/wasp_mosaic_8.tif",,of="GTiff",overwrite=TRUE)
mosaic_rasters(gdalfile=ras,dst_dataset="./data/raw_sen/8/wasp_mosaic_8.tif",of="GTiff")

hesse <- readOGR("./data/raw_sen/DigVGr-epsg25832-shp/LAND_LA.shp")

list_wasp <- list.files("./data/raw_sen/",recursive = TRUE, full.names = TRUE, pattern = "wasp_mosaic_")

# crop mosaic two size of Hesse
for (ras in list_wasp){
   ra<-raster(ras)
   ra_c <- crop(ra, hesse)
   writeRaster(ra_c, paste0("./data/raw_sen/hesse_",ra_c@data@names,".tif"),format = "GTiff", overwrite = TRUE)
}


