## ---------------------------
##
## Script name: data_preprocessing_rgb.R
##
## Purpose of script: Script to generate a model for the data created by data_split_rgb.R
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
## Notes: Some code parts & ideas are taken and/or modified from:
##
## @misc{tibav:49550,
##    title={Introduction to Deep Learning in R for analysis of UAV-based remote sensing data},
##    author={Knoth, Christian},
##    howpublished={OpenGeoHub foundation},
##    year={2020},
##    note={https://doi.org/10.5446/49550 \(Last accessed: 15 Sep 2021\)},
## }
##
## ---------------------------

## set working directory 

wd <- getwd()
setwd(wd)

## ---------------------------

options(warn = -1)

## ---------------------------

## load all necessary packages

library(tfdatasets)
library(purrr)
library(rsample)
library(stars)
library(keras)
library(tensorflow)
library(raster)
library(reticulate)
library(sf)
library(rgdal)
library(caret)
library(rgeos)
library(tfruns)
library(jpeg)
library(tfaddons)
library(magick)
library(dplyr)

##----------------------------

## functions

prepare_ds <-
   function(files = NULL,
            train,
            predict = FALSE,
            subsets_path = NULL,
            model_input_shape = c(256, 256),
            batch_size = batch_size,
            visual =FALSE) {
      if (!predict) {
         # function for random change of saturation,brightness and hue,
         # will be used as part of the augmentation
         
         spectral_augmentation <- function(img) {
            img <- tf$image$random_brightness(img, max_delta = FLAGS$bright_d)
            img <-
               tf$image$random_contrast(img, lower = FLAGS$contrast_lo, upper = FLAGS$contrast_hi)
            img <-
               tf$image$random_saturation(img, lower = FLAGS$sat_lo, upper = FLAGS$sat_hi)
            # make sure we still are between 0 and 1
            img <- tf$clip_by_value(img, 0, 1)
         }
         
         
         # create a tf_dataset from the input data.frame
         # right now still containing only paths to images
         dataset <- tensor_slices_dataset(files)
         
         # use dataset_map to apply function on each record of the dataset
         # (each record being a list with two items: img and mask), the
         # function is list_modify, which modifies the list items
         # 'img' and 'mask' by using the results of applying decode_jpg on the img and the mask
         # -> i.e. jpgs are loaded and placed where the paths to the files were (for each record in dataset)
         dataset <-
            dataset_map(dataset, function(.x)
               list_modify(
                  .x,
                  img = tf$image$decode_jpeg(tf$io$read_file(.x$img)),
                  mask = tf$image$decode_jpeg(tf$io$read_file(.x$mask))
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
               tf$image$decode_jpeg(tf$io$read_file(.x)))
         
         dataset <-
            dataset_map(dataset, function(.x)
               tf$image$convert_image_dtype(.x, dtype = tf$float32))
         
         dataset <- dataset_batch(dataset, batch_size)
         dataset <-  dataset_map(dataset, unname)
         
      }
      
   }


# flags and settings -------------------------------------------------------------------

# flags for different training runs
FLAGS <- flags(
   flag_integer("epoch", 25, "Quantity of trained epochs"),
   flag_numeric("prop1", 0.9, "Proportion training/test/validation data"),
   flag_numeric("prop2", 0.95, "Proportion training/test/validation data"),
   
   flag_numeric("lr", 0.0001, "Learning rate"),
   flag_string("input", "input128", "Sets the input shape and size"),
   flag_integer("batch_size", 8, "Changes the batch size"),
   flag_numeric(
      "factor_lr",
      0.1,
      "Setting for callback_reduce_lr_on_plateau (How much to reduce learning rate?"
   ),
   flag_string(
      "block_freeze",
      "block1_pool",
      "Way to freeze specific layers of the vgg16 from block1_pool to block5_pool;
               should also be possible with each layer for e.g. block1_conv1 and input1 should be no freeze"
   ),
   flag_numeric("bright_d",0.3,"Change brightness in spectral augmention; float, must be non-negative; default 0.3"),
   flag_numeric("contrast_lo",0.8, "Change of the lower bound of the contrast level"),
   flag_numeric("contrast_hi",1.1, "Change of the upper bound of the contrast level"),
   flag_numeric("sat_lo",0.8,"Change of the saturation; lower bound"),
   flag_numeric("sat_hi",1.1,"Change of the saturation; upper bound")
)



# for Sigmoidfocalloss
# flag_numeric("alpha", 0.7, "Alpha Values for Focal Loss"),
# flag_numeric("gamma", 2, "Gamma Values for Focal Loss"),

# set paths
path = "./data/split/"
mask = "mask/"
sen = "sen/"
pred = "test_pred/"

test_s = "test_s/"
test_m = "test_m/"

# probably there is a more satisfying way to handle these settings
if (FLAGS$input == "input96") {
   size = c(96, 96)
   input_shape = c(96, 96, 3)
   x = "input96/"
   m_path = paste0(path, x, mask)
   s_path = paste0(path, x, sen)
   p_path = paste0(path, x, pred)
} else if (FLAGS$input == "input128") {
   size = c(128, 128)
   input_shape = c(128, 128, 3)
   x = "input128/"
   m_path = paste0(path, x, mask)
   s_path = paste0(path, x, sen)
   p_path = paste0(path, x, pred)
} else if (FLAGS$input == "input192") {
   size = c(192, 192)
   input_shape = c(192, 192, 3)
   x = "input192/"
   m_path = paste0(path, x, mask)
   s_path = paste0(path, x, sen)
   p_path = paste0(path, x, pred)
} else if (FLAGS$input == "input256") {
   size = c(256, 256)
   input_shape = c(256, 256, 3)
   x = "input256/"
   m_path = paste0(path, x, mask)
   s_path = paste0(path, x, sen)
   p_path = paste0(path, x, pred)
} else if (FLAGS$input == "input320") {
   size = c(320, 320)
   input_shape = c(320, 320, 3)
   x = "input320/"
   m_path = paste0(path, x, mask)
   s_path = paste0(path, x, sen)
   p_path = paste0(path, x, pred)
   
} else if (FLAGS$input == "input384") {
   size = c(384, 384)
   input_shape = c(384, 384, 3)
   x = "input384/"
   m_path = paste0(path, x, mask)
   s_path = paste0(path, x, sen)
   p_path = paste0(path, x, pred)
} else if (FLAGS$input == "test") {
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
} else {
   print("THIS DOES NOT WORK!")
}



# get data & make some settings -------------------------------------------

# set variables
batch_size = FLAGS$batch_size

# create dataset with path to mask and data
files <- data.frame(
   img = list.files(s_path, full.names = TRUE, pattern = "*.jpg"),
   mask = list.files(m_path, full.names = TRUE, pattern = "*.jpg")
)

# split the data into training and validation dataset
# files <- initial_split(files, prop = FLAGS$prop) -> originally
set.seed(7)
ss <- sample(rep(1:3, diff(floor(nrow(files) *c(0,FLAGS$prop1,FLAGS$prop2,1)))))
training <- files[ss==1,]
validation <- files[ss==2,]
testing <- files[ss==3,]


# prepare data for training
training_dataset <-
   prepare_ds(
      training,
      train = TRUE,
      predict = FALSE,
      model_input_shape = size,
      batch_size = batch_size
   )

# also prepare validation data
validation_dataset <-
   prepare_ds(
      validation,
      train = FALSE,
      predict = FALSE,
      model_input_shape = size ,
      batch_size = batch_size
   )


# building a U-Net -------------------------------------------------------------------

# model <- unet::unet(input_shape = input_shape)
# different/easier method with package, but not possible with a pretrained network

#load pretrained vgg16 and use part of it as contracting path (feature extraction)
vgg16_feat_extr <-
   application_vgg16(weights = "imagenet",
                     include_top = FALSE,
                     input_shape = input_shape)

# freeze weights
freeze_weights(vgg16_feat_extr, to = FLAGS$block_freeze)

# just use the first 15 layers of vgg16
unet_tensor <- vgg16_feat_extr$layers[[15]]$output

# add the second part of 'U' for segemntation

# "bottom curve" of U-net
unet_tensor <-
   layer_conv_2d(
      unet_tensor,
      filters = 1024,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
   )
unet_tensor <-
   layer_conv_2d(
      unet_tensor,
      filters = 1024,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
   )

# upsampling block 1
unet_tensor <-
   layer_conv_2d_transpose(
      unet_tensor,
      filters = 512,
      kernel_size = 2,
      strides = 2,
      padding = "same"
   )
unet_tensor <-
   layer_concatenate(list(vgg16_feat_extr$layers[[14]]$output, unet_tensor))
unet_tensor <-
   layer_conv_2d(
      unet_tensor,
      filters = 512,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
   )
unet_tensor <-
   layer_conv_2d(
      unet_tensor,
      filters = 512,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
   )

# upsampling block 2
unet_tensor <-
   layer_conv_2d_transpose(
      unet_tensor,
      filters = 256,
      kernel_size = 2,
      strides = 2,
      padding = "same"
   )
unet_tensor <-
   layer_concatenate(list(vgg16_feat_extr$layers[[10]]$output, unet_tensor))
unet_tensor <-
   layer_conv_2d(
      unet_tensor,
      filters = 256,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
   )
unet_tensor <-
   layer_conv_2d(
      unet_tensor,
      filters = 256,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
   )

# upsampling block 3
unet_tensor <-
   layer_conv_2d_transpose(
      unet_tensor,
      filters = 128,
      kernel_size = 2,
      strides = 2,
      padding = "same"
   )
unet_tensor <-
   layer_concatenate(list(vgg16_feat_extr$layers[[6]]$output, unet_tensor))
unet_tensor <-
   layer_conv_2d(
      unet_tensor,
      filters = 128,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
   )
unet_tensor <-
   layer_conv_2d(
      unet_tensor,
      filters = 128,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
   )

# upsampling block 4
unet_tensor <-
   layer_conv_2d_transpose(
      unet_tensor,
      filters = 64,
      kernel_size = 2,
      strides = 2,
      padding = "same"
   )
unet_tensor <-
   layer_concatenate(list(vgg16_feat_extr$layers[[3]]$output, unet_tensor))
unet_tensor <-
   layer_conv_2d(
      unet_tensor,
      filters = 64,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
   )
unet_tensor <-
   layer_conv_2d(
      unet_tensor,
      filters = 64,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
   )

# final output
unet_tensor <-
   layer_conv_2d(
      unet_tensor,
      filters = 1,
      kernel_size = 1,
      activation = "sigmoid"
   )

# create model from tensors
model <-
   keras_model(inputs = vgg16_feat_extr$input, outputs = unet_tensor)

# compile & fit model -----------------------------------------------------------------

# some self implemented metrics & losses;

# Matthew correlation coefficient/phi coefficient
mcc <- custom_metric("mcc",function(y_true, y_pred) {
   y_true_f <- k_flatten(y_true)
   y_pred_f <- k_flatten(y_pred)
   
   tp <- k_sum(y_true_f * y_pred_f)
   tn <- k_sum((1 - y_true_f) * (1 - y_pred_f))
   fp <- k_sum((1 - y_true_f) * y_pred_f) * 100
   fn <- k_sum(y_true_f * (1 - y_pred_f)) / 100
   
   up <- tp * tn - fp * fn
   down <- k_sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
   
   mcc = up / down
   result = k_mean(mcc)
   
   return (result)
})


# Dice coefficient/F1 score
dice_coef <- custom_metric("dice_coef",function(y_true, y_pred, smooth = 1.0) {
   y_true_f <- k_flatten(y_true)
   y_pred_f <- k_flatten(y_pred)
   intersection <- k_sum(y_true_f * y_pred_f)
   result <- (2 * intersection + smooth) /
      (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
   return(result)
})


# test of buildung your own loss function e.g focal tversky loss (paper) --> could also be just focal loss

focal_tversky_loss <- custom_metric("focal_tversky_loss",function(y_true,y_pred,smooth = 1){
   y_true_pos <- k_flatten(y_true)
   y_pred_pos <- k_flatten(y_pred)
   true_pos <- k_sum(y_true_pos * y_pred_pos)
   false_neg <- k_sum(y_true_pos*(1-y_pred_pos))
   false_pos <- k_sum((1-y_true_pos)*y_pred_pos)
   alpha = 0.7
   result <- (true_pos + smooth)/(true_pos + alpha*false_neg+ (1-alpha)*false_pos + smooth)
   gamma <- 0.75
   result <- k_pow((1-result),gamma)
   #  result <- 1 - result # should not be necessary anymore
   return(result)
})

dice_loss <- custom_metric("dice_loss", function(y_true, y_pred){
   loss <- 1- dice_coef(y_true,y_pred)
   return(loss)
})



# compiling of the model
model %>% compile(
   optimizer = optimizer_adam(lr = FLAGS$lr),
   loss = "binary_crossentropy",
   metrics =  c("accuracy",mcc,dice_coef)
)
# loss_sigmoid_focal_crossentropy(alpha = FLAGS$alpha,gamma = FLAGS$gamma)

st <- format(Sys.time(), "%Y_%m_%d_%H_%M")
tb_path <- paste0("data/board_logs/",st)

# tensorboard(tb_path)
# sadly does not work for some reason -> package? installation? something else?
# callback_tensorboard("board_logs")

# train model
model %>% fit(
   training_dataset,
   validation_data =validation_dataset,
   epochs = FLAGS$epoch,
   verbose = 1,
   callbacks = c(
      callback_tensorboard(tb_path),
      callback_early_stopping(monitor = "val_loss", patience = 3),
      callback_reduce_lr_on_plateau(
         monitor = "val_loss",
         factor = FLAGS$factor_lr,
         patience = 2
      )
   )
)

# tensorboard(action="stop")

path <- paste("./data/model/sow_unet_model_",st, sep = "")

model %>% save_model_tf(filepath = path)
# save model just for prediction without custom metrics and later compiling 

# Take a look -------------------------------------------------------------
# In the future there will be a own script for visualization 
# currently in work

sample <-
   floor(runif(n = 5, min = 1, max = 17))                      

testing_dataset <-
   prepare_ds(
      testing,
      train = FALSE,
      predict = FALSE,
      model_input_shape = size ,
      batch_size = batch_size
   )

# warum auch immer geht das mit predict gleich false sehr gut --> eigentlich ja trie
# nur die Frage ob das richtig so ist

for (i in sample) {
   jpeg_path <- testing
   jpeg_path <- jpeg_path[i, ]
   
   img <- magick::image_read(jpeg_path[, 1])
   mask <- magick::image_read(jpeg_path[, 2])
   pred <-
      magick::image_read(as.raster(predict(object = model, testing_dataset)[i, , , ]))
   
   out <- magick::image_append(c(
      image_annotate(mask,"Mask", size = 10, color = "black", boxcolor = "white"),
      image_annotate(img,"Original Image", size = 10, color = "black", boxcolor = "white"),
      image_annotate(pred,"Prediction", size = 10, color = "black", boxcolor = "white")
   ))
   
   plot(out)

}

# function should only work with pretrained on imagenet bc of imagenet_preprocess_input
plot_layer_activations <-
   function(img_path,
            model,
            activations_layers,
            channels) {
      model_input_size <-
         c(model$input_shape[[2]], model$input_shape[[3]])
      
      #preprocess image for the model
      img <-
         image_load(img_path, target_size =  model_input_size) %>%
         image_to_array() %>%
         array_reshape(dim = c(1, model_input_size[1], model_input_size[2], 3)) %>%
         imagenet_preprocess_input()
      
      layer_outputs <-
         lapply(model$layers[activations_layers], function(layer)
            layer$output)
      activation_model <-
         keras_model(inputs = model$input, outputs = layer_outputs)
      activations <- predict(activation_model, img)
      if (!is.list(activations)) {
         activations <- list(activations)
      }
      
      #function for plotting one channel of a layer, adopted from: Chollet (2018): "Deep learning with R"
      plot_channel <- function(channel, layer_name, channel_name) {
         rotate <- function(x)
            t(apply(x, 2, rev))
         image(
            rotate(channel),
            axes = FALSE,
            asp = 1,
            col = terrain.colors(12),
            main = paste("layer:", layer_name, "channel:", channel_name)
         )
      }
      
      for (i in 1:length(activations)) {
         layer_activation <- activations[[i]]
         layer_name <- model$layers[[activations_layers[i]]]$name
         n_features <- dim(layer_activation)[[4]]
         for (c in channels) {
            channel_image <- layer_activation[1, , , c]
            plot_channel(channel_image, layer_name, c)
            
         }
      }
      
   }

# #visualize layers 3 and 10, channels 1 to 20
par(mfrow = c(3, 4),
    mar = c(1, 1, 1, 1),
    cex = 0.5)
plot_layer_activations(
   img_path = jpeg_path[, 1],
   model = model ,
   activations_layers = c(2,3,5,6,8,9,10,12,13,14,16,17,20,21,24,25,27,28,29,32,33),
   channels = 1:4
)

# Take a 2nd look ---------------------------------------------------------

par(mfrow = c(1, 2),
    cex = 0.5)

# # without augmention
dataset <- tensor_slices_dataset(testing)

dataset <-
   dataset_map(dataset, function(.x)
      list_modify(.x,img = tf$image$decode_jpeg(tf$io$read_file(.x$img)),
                  mask = tf$image$decode_jpeg(tf$io$read_file(.x$mask))))
dataset <-
   dataset_map(dataset, function(.x)
      list_modify(
         .x,
         img = tf$image$convert_image_dtype(.x$img, dtype = tf$float32),
         mask = tf$image$convert_image_dtype(.x$mask, dtype = tf$float32)
      ))

dataset <-
   dataset_map(dataset, function(.x)
      list_modify(.x,img = tf$clip_by_value(.x$img, 0, 1)))


example1 <- dataset %>% as_iterator() %>% iter_next()
example1$img %>% as.array() %>% as.raster() %>% plot()



# with augmention
training_dataset <-
   prepare_ds(
      testing,
      train = TRUE,
      predict = FALSE,
      model_input_shape = size,
      visual = TRUE
   )


example2 <-training_dataset  %>% as_iterator() %>% iter_next()
example2[[1]] %>% as.array() %>% as.raster() %>% plot()





