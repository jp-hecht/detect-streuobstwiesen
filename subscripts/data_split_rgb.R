## ---------------------------
##
## Script name: data_split_rgb.R
##
## Purpose of script: Create smaller patches of .jpeg from one big .tif;
##
## Author: Jonathan Hecht
##
## Date Created: 2021-09-11
##
## Copyright: -
## Email: -
##
## ---------------------------
##
## Notes: Check paths dependencies
##          Check if all packages are necessary
##
## ---------------------------

## set working directory 

wd <- getwd()
setwd(wd)

## ---------------------------

options(warn = -1)

## ---------------------------

## load all necessary packages

# library(rgdal)
library(raster)
library(jpeg)
library(rgeos)
library(greenbrown)
library(future.apply)
library(R.utils)

## functions ----------------

# subset the "big" tif to smaller jpegs with some crop on the corners
dl_subset_train <-
   function(input_raster,
            model_input_shape,
            path,
            targetname = "",
            img_info_only = FALSE,
            mask = FALSE) {
      #determine next number of quadrats in x and y direction, by simple rounding
      targetsizeX <- model_input_shape[1]
      targetsizeY <- model_input_shape[2]
      inputX <- ncol(input_raster)
      inputY <- nrow(input_raster)
      
      #determine dimensions of raster so that
      #it can be split by whole number of subsets (by shrinking it)
      while (inputX %% targetsizeX != 0) {
         inputX = inputX - 1
      }
      while (inputY %% targetsizeY != 0) {
         inputY = inputY - 1
      }
      
      #determine difference
      diffX <- ncol(input_raster) - inputX
      diffY <- nrow(input_raster) - inputY
      
      #determine new dimensions of raster and crop,
      #cutting evenly on all sides if possible
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
               writeJPEG(
                  as.array(subs),
                  target = paste0(path, targetname, i, ".jpg"),
                  quality = 1
               )
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
                  #rescale to 0-1, for jpeg export
                  if (mask == FALSE) {
                     subs <-
                        suppressMessages((subs - cellStats(subs, "min")) / (cellStats(subs, "max") -
                                                                               cellStats(subs, "min")))
                  }
                  
               })
               writeJPEG(
                  as.array(subs),
                  target = paste0(path, targetname, i, ".jpg"),
                  quality = 1
               )
            }
         )
         
         
      }
      
      rm(subs, agg, agg_poly)
      gc()
      return(rst_cropped)
   }

# removes all files which are equal so that mask and there corresponding jpeg are deleted
remove_files <- function(df) {
   future_lapply(
      seq(1, nrow(df)),
      FUN = function(i) {
         local({
            fil = df$list_m[i]
            jpeg = readJPEG(fil)
            len = length(jpeg)
            if (AllEqual(jpeg)) {
               file.remove(df$list_s[i])
               file.remove(df$list_m[i])
            } else {
               
            }
            
         })
         
      }
   )
}


# Read data & set some paths ----------

b2 <- raster("./data/sen_inp/WASP_sen_8_cro_he_c_1_99.tif")
b3 <- raster("./data/sen_inp/WASP_sen_3_cro_he_c_1_99.tif")
b4 <- raster("./data/sen_inp/WASP_sen_4_cro_he_c_1_99.tif")

input_raster <- stack(c(b2,b4, b3))

if (list_shape == "test") {
   rasterized_vector <- stack("./data/mask/test_mask.tif") # geht noch nicht
} else{
   rasterized_vector <- stack("./data/sow/sow_mask.tif")
}

# paths
path = "./data/split/"
mask = "mask/"
sen = "sen/"

test_s = "test_s/"
test_m = "test_m/"

# Use Functions -----------------------------------------------------------

# Probably there is a more satisfying way to handle these settings

plan(multisession)

for (i in  list_shape) {
   if (i == "input64") {
      size = c(64, 64)
      input_shape = c(64, 64, 3)
      x = "input64/"
      m_path = paste0(path, x, mask)
      s_path = paste0(path, x, sen)
      dir_cop_m = paste0(path, x, "cop_m")
      dir_cop_s = paste0(path, x, "cop_s")
      dir.create(dir_cop_m, recursive = T)
      dir.create(dir_cop_s, recursive = T)
      dir.create(s_path, recursive = T)
      dir.create(m_path, recursive = T)
   } else if (i == "input96") {
      size = c(96, 96)
      input_shape = c(96, 96, 3)
      x = "input96/"
      m_path = paste0(path, x, mask)
      s_path = paste0(path, x, sen)
      dir_cop_m = paste0(path, x, "cop_m")
      dir_cop_s = paste0(path, x, "cop_s")
      dir.create(dir_cop_m, recursive = T)
      dir.create(dir_cop_s, recursive = T)
      dir.create(s_path, recursive = T)
      dir.create(m_path, recursive = T)
   } else if (i == "input128") {
      size = c(128, 128)
      input_shape = c(128, 128, 3)
      x = "input128/"
      m_path = paste0(path, x, mask)
      s_path = paste0(path, x, sen)
      dir_cop_m = paste0(path, x, "cop_m")
      dir_cop_s = paste0(path, x, "cop_s")
      dir.create(dir_cop_m, recursive = T)
      dir.create(dir_cop_s, recursive = T)
      dir.create(s_path, recursive = T)
      dir.create(m_path, recursive = T)
   } else if (i == "input192") {
      size = c(192, 192)
      input_shape = c(192, 192, 3)
      x = "input192/"
      m_path = paste0(path, x, mask)
      s_path = paste0(path, x, sen)
      dir_cop_m = paste0(path, x, "cop_m")
      dir_cop_s = paste0(path, x, "cop_s")
      dir.create(dir_cop_m, recursive = T)
      dir.create(dir_cop_s, recursive = T)
      dir.create(s_path, recursive = T)
      dir.create(m_path, recursive = T)
   } else if (i == "input256") {
      size = c(256, 256)
      input_shape = c(256, 256, 3)
      x = "input256/"
      m_path = paste0(path, x, mask)
      s_path = paste0(path, x, sen)
      dir_cop_m = paste0(path, x, "cop_m")
      dir_cop_s = paste0(path, x, "cop_s")
      dir.create(dir_cop_m, recursive = T)
      dir.create(dir_cop_s, recursive = T)
      dir.create(s_path, recursive = T)
      dir.create(m_path, recursive = T)
   } else if (i == "input320") {
      size = c(320, 320)
      input_shape = c(320, 320, 3)
      x = "input320/"
      m_path = paste0(path, x, mask)
      s_path = paste0(path, x, sen)
      dir_cop_m = paste0(path, x, "cop_m")
      dir_cop_s = paste0(path, x, "cop_s")
      dir.create(dir_cop_m, recursive = T)
      dir.create(dir_cop_s, recursive = T)
      dir.create(s_path, recursive = T)
      dir.create(m_path, recursive = T)
   } else if (i == "input384") {
      size = c(384, 384)
      input_shape = c(384, 384, 3)
      x = "input384/"
      m_path = paste0(path, x, mask)
      s_path = paste0(path, x, sen)
      dir_cop_m = paste0(path, x, "cop_m")
      dir_cop_s = paste0(path, x, "cop_s")
      dir.create(dir_cop_m, recursive = T)
      dir.create(dir_cop_s, recursive = T)
      dir.create(s_path, recursive = T)
      dir.create(m_path, recursive = T)
   }else if (i == "test") {
      size = c(128, 128)
      input_shape = c(128, 128, 3)
      x = "input_test_256/"
      m_path = paste0(path, x, test_m)
      s_path = paste0(path, x, test_s)
      dir_cop_m = paste0(path, x, "test_m_c")
      dir_cop_s = paste0(path, x, "test_s_c")
      dir.create(dir_cop_m, recursive = T)
      dir.create(dir_cop_s, recursive = T)
      dir.create(s_path, recursive = T)
      dir.create(m_path, recursive = T)
   }else {
      print("THIS DOES NOT WORK!")
   }
   target_rst <- dl_subset_train(
      input_raster = rasterized_vector,
      path = m_path,
      mask = TRUE,
      model_input_shape = size
   )
   
   dl_subset_train(
      input_raster = input_raster,
      path = s_path,
      mask = FALSE,
      model_input_shape = size
   )
   
   
   # write targetrst for prediction
   writeRaster(
      target_rst,
      file = paste0("./hes_pred/target_rst/", i, ".tif"),
      overwrite = T
   )
   
   # create prediction path
   targetdir <- paste0("./hes_pred/", i, "/")
   
   # copy to prediction folder
   if (dir.exists(targetdir)) {
      copyDirectory(from = s_path, to = targetdir)
   } else {
      dir.create(targetdir)
      copyDirectory(from = s_path, to = targetdir)
   }
   
   # settings & random add on part
   list_s <-
      list.files(s_path, full.names = TRUE, pattern = "*.jpg")
   list_m <-
      list.files(m_path, full.names = TRUE, pattern = "*.jpg")
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
   # for old version
   # remove_files(input_shape, list_s)
   # input_shape[3] <- 1
   # remove_files(size = input_shape, list = list_m)
   
}


rm(list = ls(all.names = TRUE))
gc()
