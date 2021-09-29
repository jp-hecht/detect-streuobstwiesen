## ---------------------------
##
## Script name: simple_cnn.R
##
## Purpose of script: Simplified version of the whole script
##
## Author: Jonathan Hecht
##
## Date Created: 2021-09-29
##
## Copyright: -
##
## Email: -
##
## ---------------------------
##
## Notes: 
## 
## ---------------------------
##
## set working directory
wd <- getwd()
setwd(wd)
##
## ---------------------------
##
options(warn = -1)
##
## ---------------------------
## 
## load all necessary packages
library(tfdatasets)
library(tensorflow)
library(keras)
library(magick)
library(purrr)
##
## ---------------------------
## functions

prepare_ds <-
   function(files = NULL,
            train,
            predict = FALSE,
            subsets_path = NULL,
            img_size = c(256, 256),
            batch_size = batch_size,
            visual =FALSE) {
      if (!predict) {
         # function for random change of saturation,brightness and hue,
         # will be used as part of the augmentation
         
         spectral_augmentation <- function(img) {
            img <- tf$image$random_brightness(img, max_delta = 0.1)
            img <-
               tf$image$random_contrast(img, lower = 0.9, upper = 1.1)
            img <-
               tf$image$random_saturation(img, lower = 0.9, upper = 1.1)
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
               dataset_concatenate(augmentation,dataset)
            
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
               dataset_concatenate(augmentation,dataset_augmented)
            
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
               dataset_concatenate(augmentation,dataset_augmented)
            
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
         if(visual){
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

get_unet_128 <- function(input_shape = c(128, 128, 3),
                         num_classes = 1) {
   
   inputs <- layer_input(shape = input_shape)
   # 128
   
   down1 <- inputs %>%
      layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu")
   down1_pool <- down1 %>%
      layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
   # 64
   
   down2 <- down1_pool %>%
      layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu")
   down2_pool <- down2 %>%
      layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
   # 32
   
   down3 <- down2_pool %>%
      layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu")
   down3_pool <- down3 %>%
      layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
   # 16
   
   down4 <- down3_pool %>%
      layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu")
   down4_pool <- down4 %>%
      layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
   #    # 8
   
   center <- down4_pool %>%
      layer_conv_2d(filters = 1024, kernel_size = c(3, 3), padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 1024, kernel_size = c(3, 3), padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu")
   # center
   
   up4 <- center %>%
      layer_upsampling_2d(size = c(2, 2)) %>%
      {layer_concatenate(inputs = list(down4, .), axis = 3)} %>%
      layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu")
   # 16
   
   up3 <- up4 %>%
      layer_upsampling_2d(size = c(2, 2)) %>%
      {layer_concatenate(inputs = list(down3, .), axis = 3)} %>%
      layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu")
   # 32
   
   up2 <- up3 %>%
      layer_upsampling_2d(size = c(2, 2)) %>%
      {layer_concatenate(inputs = list(down2, .), axis = 3)} %>%
      layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu")
   #    # 64
   
   up1 <- up2 %>%
      layer_upsampling_2d(size = c(2, 2)) %>%
      {layer_concatenate(inputs = list(down1, .), axis = 3)} %>%
      layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu") %>%
      layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
      layer_batch_normalization() %>%
      layer_activation("relu")
   # 128
   
   classify <- layer_conv_2d(up1,
                             filters = num_classes,
                             kernel_size = c(1, 1),
                             activation = "sigmoid")
   
   
   model <- keras_model(
      inputs = inputs,
      outputs = classify
   )
   
   return(model)
}
## ---------------------------

# path to the two data sets
m_path <- "C:/Users/geoUniMarburg/Documents/detect-streuobstwiesen/data/split/input_test/test_m"
s_path <- "C:/Users/geoUniMarburg/Documents/detect-streuobstwiesen/data/split/input_test/test_s"

# create data frame with two columns and all listed files
files <- data.frame(
   img = list.files(s_path, full.names = TRUE, pattern = "*.png"),
   mask = list.files(m_path, full.names = TRUE, pattern = "*.png")
)


set.seed(7)

# proportion of training/validation/testing data
t_sample <- 0.8
v_sample <- 0.9

# create samples
s_size <- sample(rep(1:3, diff(floor(nrow(files) *c(0,t_sample,v_sample,1)))))
training <- files[s_size==1,]
validation <- files[s_size==2,]
testing <- files[s_size==3,]


# settings
img_size <- c(128,128)

model_shape <- c(128,128,3)

batch_size <- 4

epochs <- 10

lr <- 0.01

# prepare data for training
training_dataset <-
   prepare_ds(
      training,
      train = TRUE,
      predict = FALSE,
      img_size = img_size,
      batch_size = batch_size
   )

# also prepare validation data
validation_dataset <-
   prepare_ds(
      validation,
      train = FALSE,
      predict = FALSE,
      img_size = img_size,
      batch_size = batch_size
   )

# also prepare testing data
testing_dataset <-
   prepare_ds(
      testing,
      train = FALSE,
      predict = FALSE,
      img_size = img_size,
      batch_size = batch_size
   )


# Unet --------------------------------------------------------------------

# formula for dice coefficient
dice_coef <- function(y_true, y_pred, smooth = 1.0) {
   y_true_f <- k_flatten(y_true)
   y_pred_f <- k_flatten(y_pred)
   intersection <- k_sum(y_true_f * y_pred_f)
   result <- (2 * intersection + smooth) /
      (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
   return(result)
}

# and dice loss
bce_dice_loss <- function(y_true, y_pred) {
    result <- loss_binary_crossentropy(y_true, y_pred) +
       (1 - dice_coef(y_true, y_pred))
    return(result)
 }


unet_model <- get_unet_128()
 
unet_model %>% compile(
   optimizer = optimizer_rmsprop(learning_rate = lr),
   loss = bce_dice_loss,
   metrics = custom_metric("dice_coef", dice_coef)
)


# fit/train the model
unet_model %>% fit(
   training_dataset,
   validation_data =validation_dataset,
   epochs = epochs,
   verbose = 1
)


# path = 
# one version to save just for prediction
# model %>% save_model_tf(filepath = path)

# get sample of data from testing data
t_sample <-
   floor(runif(n = 5, min = 1, max = 12))  

# simple comparision of mask, image and prediction
for (i in t_sample) {
   png_path <- testing
   png_path <- png_path[i, ]
   
   img <- image_read(png_path[, 1])
   mask <- image_read(png_path[, 2])
   pred <-
      image_read(as.raster(predict(object = unet_model, testing_dataset)[i, , , ]))
   
   out <- image_append(c(
      image_annotate(mask,"Mask", size = 10, color = "black", boxcolor = "white"),
      image_annotate(img,"Original Image", size = 10, color = "black", boxcolor = "white"),
      image_annotate(pred,"Prediction", size = 10, color = "black", boxcolor = "white")
   ))
   
   plot(out)
   
}












