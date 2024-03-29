# 0.---------------------------------------------------------------------
## script name: data_split.R
##
## purpose of script: Create smaller tiles of .png from one bigger .tif;
## these tiles could be used for model creation & prediction
##
## author: Jonathan Hecht
##
## date created: 2021-09-11
##
## copyright: -
## email: -
##
## notes:
## some code parts & ideas are taken and/or modified from:
##
## @misc{tibav:49550,
##    title={Introduction to Deep Learning in R for analysis of UAV-based remote sensing data},
##    author={Knoth, Christian},
##    howpublished={OpenGeoHub foundation},
##    year={2020},
##    note={https://doi.org/10.5446/49550 /(Last accessed: 15 Sep 2021/)},
## }
##
## set working directory
wd <- getwd()
setwd(wd)

options(warn = -1)

# load all necessary packages

library(raster)
library(png)
library(greenbrown)
library(future.apply)
library(R.utils)

# 1. functions ------------------------------------------------------------

# subset the "big" .tifs to smaller .pngs
# due to the process the original input size will be changed about a small extent

subset_ds <-
   function(input_raster,
            model_input_shape,
            path,
            targetname = "",
            mask = FALSE) {
      # determine next number of quadrats in x and y direction, by simple rounding
      targetsizeX <- model_input_shape[1]
      targetsizeY <- model_input_shape[2]
      inputX <- ncol(input_raster)
      inputY <- nrow(input_raster)
      
      # determine dimensions of raster so that
      # it can be split by whole number of subsets (by shrinking it)
      while (inputX %% targetsizeX != 0) {
         inputX = inputX - 1
      }
      while (inputY %% targetsizeY != 0) {
         inputY = inputY - 1
      }
      
      # determine difference
      diffX <- ncol(input_raster) - inputX
      diffY <- nrow(input_raster) - inputY
      
      # determine new dimensions of raster and crop,
      # cutting evenly on all sides if possible
      newXmin <- floor(diffX / 2)
      newXmax <- ncol(input_raster) - ceiling(diffX / 2) - 1
      newYmin <- floor(diffY / 2)
      newYmax <- nrow(input_raster) - ceiling(diffY / 2) - 1
      rst_cropped <-
         suppressMessages(crop(
            input_raster,
            extent(input_raster, newYmin, newYmax, newXmin, newXmax)
         ))
      
      agg <-
         suppressMessages(aggregate(rst_cropped[[1]], c(targetsizeX, targetsizeY)))
      agg[]    <- suppressMessages(1:ncell(agg))
      agg_poly <- suppressMessages(rasterToPolygons(agg))
      names(agg_poly) <- "polis"
      
      if (mask) {
         future_lapply(
            seq_along(agg),
            FUN = function(i) {
               subs <- local({
                  e1  <- extent(agg_poly[agg_poly$polis == i, ])
                  
                  subs <- suppressMessages(crop(rst_cropped, e1))
                  
               })
               writePNG(as.array(subs),
                        target = paste0(path, targetname, i, ".png"))
            }
         )
      }
      else{
         future_lapply(
            seq_along(agg),
            FUN = function(i) {
               subs <- local({
                  e1  <- extent(agg_poly[agg_poly$polis == i, ])
                  
                  subs <- suppressMessages(crop(rst_cropped, e1))
                  # rescale to 0-1, for png export
                  if (mask == FALSE) {
                     subs <-
                        suppressMessages((subs - cellStats(subs, "min")) / (cellStats(subs, "max") -
                                                                               cellStats(subs, "min")))
                  }
                  
               })
               writePNG(as.array(subs),
                        target = paste0(path, targetname, i, ".png"))
            }
         )
         
         
      }
      
      rm(subs, agg, agg_poly)
      gc()
      return(rst_cropped)
   }

# removes all files whos values are all equal

remove_files <- function(df) {
   future_lapply(
      seq(1, nrow(df)),
      FUN = function(i) {
         local({
            fil = df$list_m[i]
            png = readPNG(fil)
            len = length(png)
            if (AllEqual(png)) {
               file.remove(df$list_s[i])
               file.remove(df$list_m[i])
            } else {
               
            }
            
         })
         
      }
   )
}


# function to set some necessary parameters

set_par <- function(input, path = "./data/split/", band = 3) {
   # 1 for the testing dataset
   if (input == 1) {
      size <<- c(160, 160)
      input_shape <<- c(160, 160, band)
      x <<- "input_test/"
      m_path <<- paste0(path, x, "test_m/")
      s_path <<- paste0(path, x, "test_s/")
      dir_cop_m <<- paste0(path, x, "cop_test_m")
      dir_cop_s <<- paste0(path, x, "cop_test_s")
      dir.create(dir_cop_m, recursive = T)
      dir.create(dir_cop_s, recursive = T)
      dir.create(s_path, recursive = T)
      dir.create(m_path, recursive = T)
   }
   # all other values for the Streuobstwiesen training/prediction
   else if (typeof(input) == "integer" |
            typeof(input) == "double") {
      size <<- c(input, input)
      input_shape <<- c(input, input, band)
      x = paste0("input", input, "/")
      m_path <<- paste0(path, x, "mask/")
      s_path <<- paste0(path, x,  "sen/")
      dir_cop_m <<- paste0(path, x, "cop_m")
      dir_cop_s <<- paste0(path, x, "cop_s")
      dir.create(dir_cop_m, recursive = T)
      dir.create(dir_cop_s, recursive = T)
      dir.create(m_path, recursive = T)
      dir.create(s_path, recursive = T)
   }
   else{
      print("Something went wrong")
   }
}


# 2. loop to use the created functions ------------------------------------

# paths
path = "./data/split/"

plan(multisession)


for (i in  list_shape) {
   # different input for the test dataset
   if (i == 1) {
      rasterized_vector <- stack("./data/dop40/area1/sow_mask.tif")
      input_raster <- stack("./data/dop40/area1/area_1.tif")
   } else{
      input_raster <- stack("./data/dop40/Testing/test_2.tif")
      rasterized_vector <-
         stack("./data/dop40/area1/build_vergleich/build_area_test_mask.tif")
   }
   # use the function to set some parameters
   set_par(input = i)
   
   # subsets for the mask
   target_rst <- subset_ds(
      input_raster = rasterized_vector,
      path = m_path,
      mask = TRUE,
      model_input_shape = size
   )
   # subsets for the training data
   subset_ds(
      input_raster = input_raster,
      path = s_path,
      mask = FALSE,
      model_input_shape = size
   )
   
   # write target_rst for prediction
   writeRaster(
      target_rst,
      file = paste0("./data/hes_pred/", i, ".tif"),
      overwrite = T
   )
   
   # create prediction path
   targetdir <- paste0("./data/hes_pred/", i, "/")
   
   # copy to prediction folder
   if (dir.exists(targetdir)) {
      copyDirectory(from = s_path, to = targetdir)
   } else {
      dir.create(targetdir)
      copyDirectory(from = s_path, to = targetdir)
   }
   
   # settings & randomly add some tiles
   list_s <-
      list.files(s_path, full.names = TRUE, pattern = "*.png")
   list_m <-
      list.files(m_path, full.names = TRUE, pattern = "*.png")
   
   df = data.frame(list_s, list_m)
   len <- seq(1, nrow(df))
   perc = perc
   val <- length(unique(round(perc * len)))
   
   samp <- sort(sample(len, val))
   
   subs <- df[samp,]
   
   # copy files in another folder
   for (i in subs[, 1]) {
      file.copy(from = i, to = dir_cop_s)
   }
   for (i in subs[, 2]) {
      file.copy(from = i, to = dir_cop_m)
   }
   
   # use remove files
   remove_files(df)
   
   # and add some files afterwards
   copyDirectory(from = dir_cop_s, to = s_path)
   copyDirectory(from = dir_cop_m, to = m_path)
   
}


# 3. remove the data ------------------------------------------------------
# remove all the data from the memory
rm(list = ls(all.names = TRUE))
gc()
