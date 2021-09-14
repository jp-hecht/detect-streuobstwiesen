## ---------------------------
##
## Script name: control_script_rgb.R
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
# set wd to folder detect-streuobstwiesen
wd <- getwd()
setwd(wd)

## ---------------------------

options(warn = -1)

## ---------------------------

## load all necessary packages

library(tfruns)
library(raster)

## ---------------------------

# Script to generate input for the model ----------------------------------

# shapes
list_shape = c("test")
# percentage of black/zero mask to be added; maybe also include false negativ values ( from all values/ whole hesse so also 10% could be quite all lot comparing to just XY true positive)
perc = 0.000
source("subscripts/data_split_rgb.R")


# Script to run & evaluate the model  -------------------------------------

# possible  (Hyper-)parameters to set --> Flags
batch_size = c(6,8)
lr = c(0.01, 0.001)
prop1 =  0.8
prop2 =  0.9
epoch = c(25)
sample = 0.001
factor_lr = c(0.1, 0.3,0.5)
block_freeze = c("input1","block1_pool")
#, "block1_pool", "block3_pool"
# should be better to use just the first few convolutional bases of vgg16
# smarten und realistischen input wählen --> bei acht batch mit 256 stürtzt er schon ab

bright_d = c(0.2, 0.5)
contrast_lo = c(0.5, 0.8)
contrast_hi = c(1.2, 1.4)
sat_lo = c(0.5, 0.8)
sat_hi = c(1.2, 1.4)

# for Sigmoidfocalloss
# gamma = c(1, 3, 4)
# alpha =  c(0.1, 0.7)

# currently it is better to just set one input shape <--> conflicts with tuning_run
input = c("test")

tuning_run(
   file = "subscripts/main_cnn_model_rgb.R",
   runs_dir = paste0("data/runs_rgb", "/", input),
   flags = list(
      epoch = epoch,
      prop1 = prop1,
      prop2 = prop2,
      lr = lr,
      batch_size = batch_size,
      input = input,
      factor_lr = factor_lr,
      block_freeze = block_freeze,
      bright_d = bright_d,
      contrast_lo = contrast_lo,
      contrast_hi = contrast_hi,
      sat_lo = sat_lo,
      sat_hi = sat_hi
   ),
   sample = sample,
   confirm = TRUE
)


# List runs for comparing the results -----------------------------------------

# need other paths
# ls96 <- ls_runs()
# ls128 <- ls_runs(runs_dir="C:/Users/geoUniMarburg/Desktop/Jonathan/NN/runs/RGB/input128")
# ls192 <- ls_runs(runs_dir="C:/Users/geoUniMarburg/Desktop/Jonathan/NN/runs/RGB/input192")
# ls256 <- ls_runs(runs_dir="C:/Users/geoUniMarburg/Desktop/Jonathan/NN/runs/RGB/input256")

# ls320 <- ls_runs(runs_dir="C:/Users/geoUniMarburg/Desktop/Jonathan/NN/runs/RGB/input320")
# test <- ls_runs(runs_dir="C:/Users/geoUniMarburg/Desktop/Jonathan/NN/runs/small_test")

# Predict  ----------------------------------------------------------------

name_model <- "sow_unet_model_2021_09_14_10_31"
model_path <- paste0("./data/model/", name_model)

b4 <- raster("./data/sen_inp/WASP_sen_4_cro_he_c_1_99.tif")
b3 <- raster("./data/sen_inp/WASP_sen_3_cro_he_c_1_99.tif")
b8 <- raster("./data/sen_inp/WASP_sen_8_cro_he_c_1_99.tif")

input_rst <- stack(c(b8,b4, b3))

size <- c(192, 192)

input <- "input192/"

targetdir <- paste0("./data/hes_pred/", input)


batch_size <- 6

out_path <- "./data/hes_pred/"

source("subscripts/predict_rgb.R")











