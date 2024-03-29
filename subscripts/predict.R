# 0.---------------------------------------------------------------------
##
## script name: predict.R
##
## purpose of script: Predict the trained model for a whole dataset
##
## author: Jonathan Hecht
##
## date created: 2021-09-11
##
## copyright: -
##
## email: -
##
## notes: Some code parts & ideas are taken and/or modified from:
##
## @misc{tibav:49550,
##    title={Introduction to Deep Learning in R for analysis of UAV-based remote sensing data},
##    author={Knoth, Christian},
##    howpublished={OpenGeoHub foundation},
##    year={2020},
##    note={https://doi.org/10.5446/49550 \(Last accessed: 15 Sep 2021\)},
## }
##

# set working directory

wd <- getwd()
setwd(wd)

options(warn = -1)

#load all necessary packages

library(keras)
library(tensorflow)
library(tfdatasets)
library(gdalUtils)
library(stars)
library(raster)
library(sf)

# 1. functions ------------------------------------------------------------

# function to rebuild the small tiles to one (bigger) image
rebuild_img <-
   function(pred_subsets,
            out_path,
            target_rst,
            model_name) {
      subset_pixels_x <- ncol(pred_subsets[1, , ,])
      subset_pixels_y <- nrow(pred_subsets[1, , ,])
      tiles_rows <- nrow(target_rst) / subset_pixels_y
      tiles_cols <- ncol(target_rst) / subset_pixels_x
      
      # load target image to determine dimensions
      target_stars <- st_as_stars(target_rst, proxy = F)
      # prepare subfolder for output
      result_folder <- paste0(out_path, model_name)
      if (dir.exists(result_folder)) {
         unlink(result_folder, recursive = T)
      }
      dir.create(path = result_folder)
      
      # for each tile, create a stars from corresponding predictions,
      # assign dimensions using original/target image, and save as tif:
      for (crow in 1:tiles_rows) {
         for (ccol in 1:tiles_cols) {
            i <- (crow - 1) * tiles_cols + (ccol - 1) + 1
            
            dimx <-
               c(((ccol - 1) * subset_pixels_x + 1), (ccol * subset_pixels_x))
            dimy <-
               c(((crow - 1) * subset_pixels_y + 1), (crow * subset_pixels_y))
            cstars <- st_as_stars(t(pred_subsets[i, , , 1]))
            attr(cstars, "dimensions")[[2]]$delta = -1
            # set dimensions using original raster
            st_dimensions(cstars) <-
               st_dimensions(target_stars[, dimx[1]:dimx[2], dimy[1]:dimy[2]])[1:2]
            
            write_stars(cstars, dsn = paste0(result_folder, "/_out_", i, ".tif"))
         }
      }
      
      starstiles <-
         as.vector(list.files(result_folder, full.names = T), mode = "character")
      sf::gdal_utils(
         util = "buildvrt",
         source = starstiles,
         destination = paste0(result_folder, "/mosaic.vrt")
      )
      
      sf::gdal_utils(
         util = "warp",
         source = paste0(result_folder, "/mosaic.vrt"),
         destination = paste0(result_folder, "/mosaic.tif")
      )
   }

# function to prepare your data for prediction
prepare_ds_predict <-
   function(files = NULL,
            subsets_path = NULL,
            model_input_shape = c(256, 256),
            batch_size = batch_size,
            predict = TRUE) {
      if (predict) {
         # make sure subsets are read in in correct order
         # so that they can later be reassembled correctly
         # needs files to be named accordingly (only number)
         o <-
            order(as.numeric(tools::file_path_sans_ext(basename(
               list.files(subsets_path)
            ))))
         subset_list <-
            list.files(subsets_path, full.names = T)[o]
         
         dataset <- tensor_slices_dataset(subset_list)
         
         dataset <-
            dataset_map(dataset, function(.x)
               tf$image$decode_png(tf$io$read_file(.x)))
         
         dataset <-
            dataset_map(dataset, function(.x)
               tf$image$convert_image_dtype(.x, dtype = tf$float32))
         linerange
         dataset <- dataset_batch(dataset, batch_size)
         dataset <-  dataset_map(dataset, unname)
         
      }
      
   }


# 2. Predict the model for the tiles --------------------------------------

# load target raster
target_rst <- raster(paste0("./data/hes_pred/", osize, ".tif"))

# load the model
model <-
   load_model_tf(model_path, compile = FALSE)

# prepare data for prediction
pred_data <-
   prepare_ds_predict(
      predict = TRUE,
      subsets_path = targetdir,
      model_input_shape = size,
      batch_size = batch_size
   )

# predict for each tile
pred_subsets <- predict(object = model, x = pred_data)

# name the model
model_name <- tools::file_path_sans_ext(name_model)


# 3. Rebuild your tiles to one image --------------------------------------

rebuild_img(
   pred_subsets = pred_subsets,
   out_path = out_path ,
   target_rst = target_rst,
   model_name = name_model
)
