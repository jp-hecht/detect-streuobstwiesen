## ---------------------------
##
## Script name: visualization.R
##
## Purpose of script: Visualization
##
## Author: Jonathan Hecht
##
## Date Created: 2021-09-16
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

## set working directory 

wd <- getwd()
setwd(wd)

## ---------------------------

options(warn = -1)

## ---------------------------

## load all necessary packages

library(magick)

## ---------------------------

## functions


# Images ------------------------------------------------------------------

par(mfrow = c(3, 4),
    mar = c(1, 1, 1, 1),
    cex = 0.1)

s_path <- "./data/split/input256/sen/"
m_path <- "./data/split/input256/mask/"

files <- data.frame(
   img = list.files(s_path, full.names = TRUE, pattern = "*.jpg"),
   mask = list.files(m_path, full.names = TRUE, pattern = "*.jpg")
)

set.seed(16)
sample <-
   floor(runif(n = 6, min = 1, max = 259)) 



for (i in sample) {
   jpeg_path <- files
   jpeg_path <- jpeg_path[i, ]
   
   img <- magick::image_read(jpeg_path[, 1])
   mask <- magick::image_read(jpeg_path[, 2])
   out <- magick::image_append(c(
      image_annotate(mask,"Mask", size = 15, color = "black", boxcolor = "white"),
      image_annotate(img,"Original Image", size = 15, color = "black", boxcolor = "white")))
   
   plot(out)
   # image_write(out, path = paste0("./test/",i,".png"), format = "png")
}

s_path <- "./data/split/input384/sen/"
m_path <- "./data/split/input384/mask/"

files <- data.frame(
   img = list.files(s_path, full.names = TRUE, pattern = "*.jpg"),
   mask = list.files(m_path, full.names = TRUE, pattern = "*.jpg")
)

set.seed(30)
sample <-
   floor(runif(n = 6, min = 1, max = 159)) 



for (i in sample) {
   jpeg_path <- files
   jpeg_path <- jpeg_path[i, ]
   
   img <- magick::image_read(jpeg_path[, 1])
   mask <- magick::image_read(jpeg_path[, 2])
   out <- magick::image_append(c(
      image_annotate(mask,"Mask", size = 15, color = "black", boxcolor = "white"),
      image_annotate(img,"Original Image", size = 15, color = "black", boxcolor = "white")))
   
   plot(out)
   # image_write(out, path = paste0("./test/",i,".png"), format = "png")
}



### grad-cam from chollet
library(keras)
library(tensorflow)
library(tfdatasets)

model_path<- paste0("./data/model/", "sow_unet_model_2021_11_11_23_20")

model <-
   load_model_tf(model_path, compile = FALSE)

img_path <- "./data/dop40/Testing/input224/sen/493.png"


img <- image_load(img_path, target_size = c(224, 224)) %>%          
image_to_array() %>%                                             
array_reshape(dim = c(1, 224, 224, 3))

# imagenet_preprocess_input()    -> vlt muss ich mein bild auch noch preprocessen und nicht nur laden! bspw data augmentation



preds <- model %>% predict(img)
preds

which.max(preds[1,,,])


sow <- model$output[,,, 1]                             

last_conv_layer <- model %>% get_layer("conv2d_22")                     

grads <- tf$GradientTape(sow, last_conv_layer$output)[[1]] 




