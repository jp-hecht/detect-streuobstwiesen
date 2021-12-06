# 0.---------------------------------------------------------------------
##
## Script name: main_cnn_model.R
##
## Purpose of script: Script to train a U-Net and also to partly evaluate the model
##
## Author: Jonathan Hecht
##
## Date Created: 2021-09-11
##
## Copyright: -
## Email: -
##
## Notes: Some code parts & ideas are taken and/or modified from:
##
## @misc{tibav:49550,
##    title={Introduction to Deep Learning in R for analysis of UAV-based remote sensing data},
##    author={Knoth, Christian},
##    howpublished={OpenGeoHub foundation},
##    year={2020},
##    note={https://doi.org/10.5446/49550 /(Last accessed: 15 Sep 2021/)},
## }
##

# set working directory

wd <- getwd()
setwd(wd)

options(warn = -1)

# load all necessary packages
library(tfdatasets)
library(purrr)
library(rsample)
library(keras)
library(tensorflow)
library(reticulate)
library(tfruns)
library(png)
library(magick)

# 1. functions ------------------------------------------------------------

# function to prepare your data from the data_split.R script
prepare_ds <-
   function(files = NULL,
            train,
            predict = FALSE,
            subsets_path = NULL,
            model_input_shape = c(256, 256),
            batch_size = batch_size,
            visual = FALSE) {
      if (!predict) {
         # function for random change of saturation,brightness and hue,
         # will be used as part of the augmentation
         
         spectral_augmentation <- function(img) {
            img <- tf$image$random_brightness(img, max_delta = FLAGS$bright_d)
            img <-
               tf$image$random_contrast(img,
                                        lower = FLAGS$contrast_lo,
                                        upper = FLAGS$contrast_hi)
            img <-
               tf$image$random_saturation(img,
                                          lower = FLAGS$sat_lo,
                                          upper = FLAGS$sat_hi)
            # make sure we still are between 0 and 1
            img <- tf$clip_by_value(img, 0, 1)
         }
         
         
         # create a tf_dataset from the input data.frame
         # right now still containing only paths to images
         dataset <- tensor_slices_dataset(files)
         
         # use dataset_map to apply function on each record of the dataset
         # (each record being a list with two items: img and mask), the
         # function is list_modify, which modifies the list items
         # 'img' and 'mask' by using the results of applying decode_png on the img and the mask
         # -> i.e. pngs are loaded and placed where the paths to the files were (for each record in dataset)
         dataset <-
            dataset_map(dataset, function(.x)
               list_modify(
                  .x,
                  img = tf$image$decode_png(tf$io$read_file(.x$img)),
                  mask = tf$image$decode_png(tf$io$read_file(.x$mask))
               ))
         
         # convert to float32:
         # for each record in dataset, both its list items are modified
         # by the result of applying convert_image_dtype to them
         dataset <-
            dataset_map(dataset, function(.x)
               list_modify(
                  .x,
                  img = tf$image$convert_image_dtype(.x$img, dtype = tf$float32),
                  mask = tf$image$convert_image_dtype(.x$mask, dtype = tf$float32)
               ))
         
         
         # data augmentation performed on training set only
         if (train) {
            # augmentation 1: flip left right, including random change of
            # saturation, brightness and contrast
            
            # for each record in dataset, only the img item is modified by the result
            # of applying spectral_augmentation to it
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
               dataset_concatenate(augmentation, dataset)
            
            # augmentation 2: flip up down,
            # including random change of saturation, brightness and contrast
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
               dataset_concatenate(augmentation, dataset_augmented)
            
            # augmentation 3: flip left right AND up down,
            # including random change of saturation, brightness and contrast
            
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
               dataset_concatenate(augmentation, dataset_augmented)
            
         }
         
         # shuffling on training set only
         if (!visual) {
            if (train) {
               dataset <-
                  dataset_shuffle(dataset_augmented, buffer_size = batch_size * 256)
            }
            
            # train in batches; batch size might need to be adapted depending on
            # available memory
            dataset <- dataset_batch(dataset, batch_size)
         }
         if (visual) {
            dataset <- dataset_augmented
         }
         
         # output needs to be unnamed
         dataset <-  dataset_map(dataset, unname)
         
      } else{
         # make sure subsets are read in in correct order
         # so that they can later be reassembled correctly
         # needs files to be named accordingly (only number)
         o <-
            order(as.numeric(tools::file_path_sans_ext(basename(
               list.files(subsets_path)
            ))))
         subset_list <- list.files(subsets_path, full.names = T)[o]
         
         dataset <- tensor_slices_dataset(subset_list)
         
         dataset <-
            dataset_map(dataset, function(.x)
               tf$image$decode_png(tf$io$read_file(.x)))
         
         dataset <-
            dataset_map(dataset, function(.x)
               tf$image$convert_image_dtype(.x, dtype = tf$float32))
         
         dataset <- dataset_batch(dataset, batch_size)
         dataset <-  dataset_map(dataset, unname)
         
      }
      
   }


get_unet <- function(input_shape = c(128, 128, 3),
                     num_classes = 1) {
   inputs <- layer_input(shape = input_shape)
   # 128
   
   down1 <- inputs %>%
      layer_conv_2d(filters = 64,
                    kernel_size = c(3, 3),
                    padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 64,
                    kernel_size = c(3, 3),
                    padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu")
   down1_pool <- down1 %>%
      layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
   # 64
   
   down2 <- down1_pool %>%
      layer_conv_2d(filters = 128,
                    kernel_size = c(3, 3),
                    padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 128,
                    kernel_size = c(3, 3),
                    padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu")
   down2_pool <- down2 %>%
      layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
   # 32
   
   down3 <- down2_pool %>%
      layer_conv_2d(filters = 256,
                    kernel_size = c(3, 3),
                    padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 256,
                    kernel_size = c(3, 3),
                    padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu")
   down3_pool <- down3 %>%
      layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
   # 16
   
   down4 <- down3_pool %>%
      layer_conv_2d(filters = 512,
                    kernel_size = c(3, 3),
                    padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 512,
                    kernel_size = c(3, 3),
                    padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu")
   down4_pool <- down4 %>%
      layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
   #    # 8
   
   center <- down4_pool %>%
      layer_conv_2d(filters = 1024,
                    kernel_size = c(3, 3),
                    padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 1024,
                    kernel_size = c(3, 3),
                    padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu")
   # center
   
   up4 <- center %>%
      layer_upsampling_2d(size = c(2, 2)) %>%
      {
         layer_concatenate(inputs = list(down4, .), axis = 3)
      } %>%
      layer_conv_2d(filters = 512,
                    kernel_size = c(3, 3),
                    padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 512,
                    kernel_size = c(3, 3),
                    padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 512,
                    kernel_size = c(3, 3),
                    padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu")
   # 16
   
   up3 <- up4 %>%
      layer_upsampling_2d(size = c(2, 2)) %>%
      {
         layer_concatenate(inputs = list(down3, .), axis = 3)
      } %>%
      layer_conv_2d(filters = 256,
                    kernel_size = c(3, 3),
                    padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 256,
                    kernel_size = c(3, 3),
                    padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 256,
                    kernel_size = c(3, 3),
                    padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu")
   # 32
   
   up2 <- up3 %>%
      layer_upsampling_2d(size = c(2, 2)) %>%
      {
         layer_concatenate(inputs = list(down2, .), axis = 3)
      } %>%
      layer_conv_2d(filters = 128,
                    kernel_size = c(3, 3),
                    padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 128,
                    kernel_size = c(3, 3),
                    padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 128,
                    kernel_size = c(3, 3),
                    padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu")
   #    # 64
   
   up1 <- up2 %>%
      layer_upsampling_2d(size = c(2, 2)) %>%
      {
         layer_concatenate(inputs = list(down1, .), axis = 3)
      } %>%
      layer_conv_2d(filters = 64,
                    kernel_size = c(3, 3),
                    padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 64,
                    kernel_size = c(3, 3),
                    padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 64,
                    kernel_size = c(3, 3),
                    padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu")
   # 128
   
   classify <- layer_conv_2d(
      up1,
      filters = num_classes,
      kernel_size = c(1, 1),
      activation = "sigmoid"
   )
   
   
   model <- keras_model(inputs = inputs,
                        outputs = classify)
   
   return(model)
}


set_par <- function(input, path = "./data/split/", band = 3) {
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
   else if (typeof(input) == "integer" | typeof(input) == "double") {
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


# 2. some settings and paths ----------------------------------------------

# basic flags which could be modified by the control_script.R
FLAGS <- flags(
   flag_integer("epoch", 15, "Quantity of trained epochs"),
   flag_numeric("prop1", 0.8, "Proportion training/validation data"),
   flag_numeric("lr", 0.001, "Learning rate"),
   flag_integer("input", 1, "Sets the input shape and size"),
   flag_integer("batch_size", 8, "Changes the batch size"),
   flag_numeric(
      "factor_lr",
      0.1,
      "Setting for callback_reduce_lr_on_plateau (How much to reduce learning rate?"
   ),
   flag_numeric(
      "bright_d",
      0.3,
      "Change brightness in spectral augmention; float, must be non-negative; default 0.3"
   ),
   flag_numeric(
      "contrast_lo",
      0.85,
      "Change of the lower bound of the contrast level"
   ),
   flag_numeric(
      "contrast_hi",
      1.25,
      "Change of the upper bound of the contrast level"
   ),
   flag_numeric("sat_lo", 0.75, "Change of the saturation; lower bound"),
   flag_numeric("sat_hi", 1.25, "Change of the saturation; upper bound")
)

# set paths
path = "./data/split/"

set_par(input = FLAGS$input)

# set variables
batch_size = FLAGS$batch_size


# 3. data preparation -----------------------------------------------------

# create dataset with path to mask and data
files <- data.frame(
   img = list.files(s_path, full.names = TRUE, pattern = "*.png"),
   mask = list.files(m_path, full.names = TRUE, pattern = "*.png")
)

# split the data into training and validation dataset
set.seed(7)
data <- initial_split(files, prop = FLAGS$prop1, strata = NULL)

# prepare data for training
training_dataset <-
   prepare_ds(
      training(data),
      train = TRUE,
      predict = FALSE,
      model_input_shape = size,
      batch_size = batch_size
   )

# also prepare validation data
validation_dataset <-
   prepare_ds(
      testing(data),
      train = FALSE,
      predict = FALSE,
      model_input_shape = size ,
      batch_size = batch_size
   )


# 4. model building and training ------------------------------------------

model <- get_unet(input_shape = input_shape)

# Matthew correlation coefficient coefficient

mcc <- function(y_true, y_pred) {
   y_true_f <- k_flatten(y_true)
   y_pred_f <- k_flatten(y_pred)
   
   tp <- k_sum(y_true_f * y_pred_f)
   tn <- k_sum((1 - y_true_f) * (1 - y_pred_f))
   fp <- k_sum((1 - y_true_f) * y_pred_f)
   fn <- k_sum(y_true_f * (1 - y_pred_f))
   
   up <- tp * tn - fp * fn
   down <- (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
   
   mcc = up / k_sqrt(down + k_epsilon())
   result = k_mean(mcc)
   
   return (result)
}

# just tested
# mcc_loss <- function(y_true,y_pred){
#    mcc_loss <- 1- mcc(y_true,y_pred)
#    return(mcc_loss)
# }

dice_coef <- function(y_true, y_pred, smooth = 1.0) {
   y_true_f <- k_flatten(y_true)
   y_pred_f <- k_flatten(y_pred)
   intersection <- k_sum(y_true_f * y_pred_f)
   result <- (2 * intersection + smooth) /
      (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
   return(result)
}

# just tested
# bce_dice_loss <- function(y_true, y_pred) {
#    result <- loss_binary_crossentropy(y_true, y_pred) +
#       (1 - dice_coef(y_true, y_pred))
#    return(result)
# }


dice_loss <- function(y_true, y_pred) {
   result <- (1 - dice_coef(y_true, y_pred))
   return(result)
}

# compiling of the model
model %>% compile(
   optimizer = optimizer_adam(learning_rate = FLAGS$lr),
   loss = dice_loss,
   metrics = c(
      custom_metric("dice_coef", dice_coef),
      custom_metric("mcc", mcc)
   )
)

# create a path to save the model later
st <- format(Sys.time(), "%Y_%m_%d_%H_%M")
path <- paste("./data/model/sow_unet_model_", st, sep = "")

# train model
model %>% fit(
   training_dataset,
   validation_data = validation_dataset,
   epochs = FLAGS$epoch,
   verbose = 1,
   callbacks = c(
      callback_tensorboard(tb_path),
      callback_early_stopping(
         monitor = "val_loss",
         patience = 3
      ),
      callback_reduce_lr_on_plateau(
         monitor = "val_loss",
         factor = FLAGS$factor_lr,
         patience = 2
      )
   )
)

model %>% save_model_tf(filepath = path)


# 5. first visual impression and comparison -------------------------------

t_s_path <- "./data/hes_pred/160/"
t_m_path <- "./data/split/input160/mask/"


testing <- data.frame(
   img = list.files(t_s_path, full.names = TRUE, pattern = "*.png"),
   mask = list.files(t_m_path, full.names = TRUE, pattern = "*.png")
)

testing_dataset <-
   prepare_ds(
      testing,
      train = FALSE,
      predict = FALSE,
      model_input_shape = size ,
      batch_size = batch_size
   )

sample <-
   floor(runif(
      n = 5,
      min = 1,
      max = nrow(testing)
   ))

for (i in sample) {
   png_path <- testing
   png_path <- png_path[i,]
   
   img <- magick::image_read(png_path[, 1])
   mask <- magick::image_read(png_path[, 2])
   pred <-
      magick::image_read(as.raster(predict(object = model, testing_dataset)[i, , ,]))
   
   out <- magick::image_append(c(
      image_annotate(
         mask,
         "Mask",
         size = 10,
         color = "black",
         boxcolor = "white"
      ),
      image_annotate(
         img,
         "Original Image",
         size = 10,
         color = "black",
         boxcolor = "white"
      ),
      image_annotate(
         pred,
         "Prediction",
         size = 10,
         color = "black",
         boxcolor = "white"
      )
   ))
   
   plot(out)
   
}


# 6. evaluation of the model with test dataset -------------------------------
ev <- model$evaluate(testing_dataset)
