## ---------------------------
##
## Script name: predict_rgb.R
##
## Purpose of script:
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
## Notes:
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

library(keras)
library(tensorflow)
library(tfdatasets)
library(gdalUtils)
library(stars)

## ---------------------------

rebuild_img <-
   function(pred_subsets,
            out_path,
            target_rst,
            model_name) {
      require(raster)
      require(sf)
      require(stars)
      
      
      subset_pixels_x <- ncol(pred_subsets[1, , , ])
      subset_pixels_y <- nrow(pred_subsets[1, , , ])
      tiles_rows <- nrow(target_rst) / subset_pixels_y
      tiles_cols <- ncol(target_rst) / subset_pixels_x
      
      # load target image to determine dimensions
      target_stars <- st_as_stars(target_rst, proxy = F)
      #prepare subfolder for output
      result_folder <- paste0(out_path, model_name)
      if (dir.exists(result_folder)) {
         unlink(result_folder, recursive = T)
      }
      dir.create(path = result_folder)
      
      #for each tile, create a stars from corresponding predictions,
      #assign dimensions using original/target image, and save as tif:
      for (crow in 1:tiles_rows) {
         for (ccol in 1:tiles_cols) {
            i <- (crow - 1) * tiles_cols + (ccol - 1) + 1
            
            dimx <-
               c(((ccol - 1) * subset_pixels_x + 1), (ccol * subset_pixels_x))
            dimy <-
               c(((crow - 1) * subset_pixels_y + 1), (crow * subset_pixels_y))
            cstars <- st_as_stars(t(pred_subsets[i, , , 1]))
            attr(cstars, "dimensions")[[2]]$delta = -1
            #set dimensions using original raster
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


dl_prepare_data <-
   function(files = NULL,
            train,
            predict = FALSE,
            subsets_path = NULL,
            model_input_shape = c(256, 256),
            batch_size = batch_size) {
      if (!predict) {
         #function for random change of saturation,brightness and hue,
         #will be used as part of the augmentation
         # spectral_augmentation <- function(img) {
         #    img <- tf$image$random_brightness(img, max_delta = 0.3)
         #    img <- tf$image$random_contrast(img, lower = 0.8, upper = 1.1)
         #    img <- tf$image$random_saturation(img, lower = 0.8, upper = 1.1)
         #    # make sure we still are between 0 and 1
         #    img <- tf$clip_by_value(img, 0, 1)
         # }
         spectral_augmentation <- function(img) {
            img <- tf$image$random_brightness(img, max_delta = 0.9)
            img <-
               tf$image$random_contrast(img, lower = 0.3, upper = 1.9)
            img <-
               tf$image$random_saturation(img, lower = 0.2, upper = 1.9)
            # make sure we still are between 0 and 1
            img <- tf$clip_by_value(img, 0, 1)
         }
         
         
         #create a tf_dataset from the input data.frame
         #right now still containing only paths to images
         dataset <- tensor_slices_dataset(files)
         
         #use dataset_map to apply function on each record of the dataset
         #(each record being a list with two items: img and mask), the
         #function is list_modify, which modifies the list items
         #'img' and 'mask' by using the results of applying decode_jpg on the img and the mask
         #-> i.e. jpgs are loaded and placed where the paths to the files were (for each record in dataset)
         dataset <-
            dataset_map(dataset, function(.x)
               list_modify(
                  .x,
                  img = tf$image$decode_jpeg(tf$io$read_file(.x$img)),
                  mask = tf$image$decode_jpeg(tf$io$read_file(.x$mask))
               ))
         
         #convert to float32:
         #for each record in dataset, both its list items are modyfied
         #by the result of applying convert_image_dtype to them
         dataset <-
            dataset_map(dataset, function(.x)
               list_modify(
                  .x,
                  img = tf$image$convert_image_dtype(.x$img, dtype = tf$float32),
                  mask = tf$image$convert_image_dtype(.x$mask, dtype = tf$float32)
               ))
         
         #resize:
         #for each record in dataset, both its list items are modified
         #by the results of applying resize to them
         dataset <-
            dataset_map(dataset, function(.x)
               list_modify(
                  .x,
                  img = tf$image$resize(.x$img, size = shape(
                     model_input_shape[1], model_input_shape[2]
                  )),
                  mask = tf$image$resize(.x$mask, size = shape(
                     model_input_shape[1], model_input_shape[2]
                  ))
               ))
         
         
         # data augmentation performed on training set only
         if (train) {
            #augmentation 1: flip left right, including random change of
            #saturation, brightness and contrast
            
            #for each record in dataset, only the img item is modified by the result
            #of applying spectral_augmentation to it
            augmentation <-
               dataset_map(dataset, function(.x)
                  list_modify(.x, img = spectral_augmentation(.x$img)))
            
            #...as opposed to this, flipping is applied to img and mask of each record
            augmentation <-
               dataset_map(augmentation, function(.x)
                  list_modify(
                     .x,
                     img = tf$image$flip_left_right(.x$img),
                     mask = tf$image$flip_left_right(.x$mask)
                  ))
            
            dataset_augmented <-
               dataset_concatenate(dataset, augmentation)
            
            #augmentation 2: flip up down,
            #including random change of saturation, brightness and contrast
            augmentation <-
               dataset_map(dataset, function(.x)
                  list_modify(.x, img = spectral_augmentation(.x$img)))
            
            augmentation <-
               dataset_map(augmentation, function(.x)
                  list_modify(
                     .x,
                     img = tf$image$flip_up_down(.x$img),
                     mask = tf$image$flip_up_down(.x$mask)
                  ))
            
            dataset_augmented <-
               dataset_concatenate(dataset_augmented, augmentation)
            
            #augmentation 3: flip left right AND up down,
            #including random change of saturation, brightness and contrast
            
            augmentation <-
               dataset_map(dataset, function(.x)
                  list_modify(.x, img = spectral_augmentation(.x$img)))
            
            augmentation <-
               dataset_map(augmentation, function(.x)
                  list_modify(
                     .x,
                     img = tf$image$flip_left_right(.x$img),
                     mask = tf$image$flip_left_right(.x$mask)
                  ))
            
            augmentation <-
               dataset_map(augmentation, function(.x)
                  list_modify(
                     .x,
                     img = tf$image$flip_up_down(.x$img),
                     mask = tf$image$flip_up_down(.x$mask)
                  ))
            
            dataset_augmented <-
               dataset_concatenate(dataset_augmented, augmentation)
            
         }
         
         # shuffling on training set only
         if (train) {
            dataset <-
               dataset_shuffle(dataset_augmented, buffer_size = batch_size * 256)
         }
         
         # train in batches; batch size might need to be adapted depending on
         # available memory
         dataset <- dataset_batch(dataset, batch_size)
         
         # output needs to be unnamed
         dataset <-  dataset_map(dataset, unname)
         
      } else{
         #make sure subsets are read in in correct order
         #so that they can later be reassembled correctly
         #needs files to be named accordingly (only number)
         o <-
            order(as.numeric(tools::file_path_sans_ext(basename(
               list.files(subsets_path)
            ))))
         subset_list <- list.files(subsets_path, full.names = T)[o]
         
         dataset <- tensor_slices_dataset(subset_list)
         
         dataset <-
            dataset_map(dataset, function(.x)
               tf$image$decode_jpeg(tf$io$read_file(.x)))
         
         dataset <-
            dataset_map(dataset, function(.x)
               tf$image$convert_image_dtype(.x, dtype = tf$float32))
         
         dataset <-
            dataset_map(dataset, function(.x)
               tf$image$resize(.x, size = shape(
                  model_input_shape[1], model_input_shape[2]
               )))
         
         dataset <- dataset_batch(dataset, batch_size)
         dataset <-  dataset_map(dataset, unname)
         
      }
      
   }


# Predict -----------------------------------------------------------------

i <- gsub('.$', '', input)

target_rst <- raster(paste0("./hes_pred/target_rst/", i, ".tif"))

model <-
   load_model_tf(model_path, compile=FALSE)

# just for predictions                 
#custom_objects = list("mcc" = mcc, "dice_coef" = dice_coef))


pred_data <-
   dl_prepare_data(
      train = FALSE,
      predict = TRUE,
      subsets_path = targetdir,
      model_input_shape = size,
      batch_size = batch_size
   )

pred_subsets <- predict(object = model, x = pred_data)

model_name <- tools::file_path_sans_ext(name_model)


rebuild_img(
   pred_subsets = pred_subsets,
   out_path = out_path ,
   target_rst = target_rst,
   model_name = model_name
)
# also with future_apply would be nice
