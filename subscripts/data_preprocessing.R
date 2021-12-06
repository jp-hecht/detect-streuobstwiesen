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

b2 <- raster("./data/dop40/area2/area_2.tif")

ra <- rasterize(input_vector,b2[[1]])
ra[is.na(ra[])] <- 0
ra <- reclassify(ra,cbind(c(1:ra@data@max),1))
plot(ra)

writeRaster(ra,"./data/dop40/area2/sow_mask.tif",overwrite=T)
ra <- raster("./data/dop40/area2/sow_mask.tif")
# writeRaster(ra,"./data/sow/sow_mask.tif",overwrite=T)
# ra <- raster("./data/sow/sow_mask.tif")









################################
input_vector <- readOGR("C:/Users/geoUniMarburg/Documents/detect-streuobstwiesen/data/dop40/area1/build_vergleich/build_area_test.gpkg")

b2 <- raster("C:/Users/geoUniMarburg/Documents/detect-streuobstwiesen/data/dop40/Testing/test_2.tif")

ra <- rasterize(input_vector,b2)
plot(ra)
summary(ra)

ra[is.na(ra[])] <- 0
ra <- reclassify(ra,cbind(c(1:ra@data@max),1))
plot(ra)
writeRaster(ra,"C:/Users/geoUniMarburg/Documents/detect-streuobstwiesen/data/dop40/area1/build_vergleich/build_area_test_mask.tif",overwrite=T)
ra <- raster("./data/dop40/Testing/sow_test.tif")


















############### Test DOP
input_vector <- readOGR("./data/sow/test_mar_build.gpkg")

ras <- raster("./data/sow/test_mar_dop.tif")

ra <- rasterize(input_vector,ras)
ra[is.na(ra[])] <- 0
ra <- reclassify(ra,cbind(c(1:ra@data@max),1))
writeRaster(ra,"./data/sow/test_mar_mask.tif",overwrite=T)

#################

ras <- stack("./data/dop40/area2/area2_dop40.tif")
ra_in <- ras@layers[[4]]
ra_r <- ras@layers[[1]]
ra_g <- ras@layers[[2]]
ra_b <- ras@layers[[3]]
ra_s <- stack(ra_r,ra_g,ra_g)

writeRaster(ra_s,"./data/sow/test_mar_dop.tif",overwrite=T)




ra <- raster("C:/Users/geoUniMarburg/Documents/detect-streuobstwiesen/data/raw_sen/2/SENTINEL2X_20200515-000000-000_L3A_T32UMA_C_V1-2_FRC_B2.tif")


ra_s <- stretch(ra)


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




ra_wasp <- raster("C:/Users/geoUniMarburg/Documents/detect-streuobstwiesen/data/raw_sen/hesse_wasp_mosaic_4.tif")


#ra_wasp_stre <- stretch(ra_wasp)
# die variante iust es definitv bnicht
#writeRaster(ra_wasp_stre, "C:/Users/geoUniMarburg/Documents/detect-streuobstwiesen/data/raw_sen/hesse_wasp_mosaic_4_stretch.tif")

## zum zusammenführen umbennen



old_files <- "C:/Users/geoUniMarburg/Documents/detect-streuobstwiesen/data/dop40/area2/input224_10/mask"
#m_path <- "C:/Users/geoUniMarburg/Documents/detect-streuobstwiesen/data/dop40/area2/input288/mask"

new_files <- "C:/Users/geoUniMarburg/Documents/detect-streuobstwiesen/data/dop40/area2/input224_10/cop_m/2"


# List the jpg files in the folder

old_files <- list.files(old_files, pattern = "*.png", full.names = TRUE)
head(old_files)

# Create vector of new files

new_files <- paste0(new_files,1:length(old_files),".png")
head(new_files)



file.copy(from = old_files, to = new_files)

# Clear out the old files
file.remove(old_files)
























