## ---------------------------
##
## Script name: control_script_rgb.R
##
## Purpose of script: Script to run parts of the workflow and therefore the
## possibility to set parameters for different testing runs and hyperparameters.
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
## Notes: - tuning_run takes extremely long to start
##        - not sure if source() works as it should -> more testing
##
## ---------------------------
##
## set working directory
setwd(wd)
##
## ---------------------------
##
options(warn = -1)
##
## ---------------------------
##
## load all necessary packages
library(tfruns)
library(raster)
##

# script to generate inputs for the model and later predictions ----------------

# setting for different input shapes e.g. input128 or input256
list_shape = c("input384")

# percentage of black/zero mask to be added; maybe also include false negative
# values ( from all values/ whole hesse so also 10% could be quite all lot
# comparing to just XY true positive)
perc = 0.000

source("subscripts/data_split_rgb.R")

# script to run & evaluate the model  ------------------------------------------

# possible  (Hyper-)parameters to set --> Flags

# batch size
batch_size = c(6, 8)
# learning rate
lr = c(0.01, 0.001)
# proportation training/testing/validation data
prop1 =  0.8
prop2 =  0.9
# number of epochs
epoch = c(25)
# randomly sampled set of parameters
sample = 0.001
# factor to reduce the lr when loss does not improve anymore
factor_lr = c(0.1, 0.3, 0.5)
# convolutional blocks to freeze the pretrained model
block_freeze = c("input1", "block1_pool")
# settings for the spectral augmentation
bright_d = c(0.2, 0.5)
contrast_lo = c(0.5, 0.8)
contrast_hi = c(1.2, 1.4)
sat_lo = c(0.5, 0.8)
sat_hi = c(1.2, 1.4)

# for Sigmoidfocalloss
# gamma = c(1, 3, 4)
# alpha =  c(0.1, 0.7)

# currently it is better to just set one input shape <--> conflicts with tuning_run
input = c("input96")

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


# comparing the results in each folder -----------------------------------------
# e.g.
# ls96 <- ls_runs()

# path maybe not right -> testing
# test <- ls_runs(runs_dir="./data/run")

# predict  ---------------------------------------------------------------------

# manually set the name of the model folder to predict
name_model <- "sow_unet_model_2021_09_16_09_25"

model_path <- paste0("./data/model/", name_model)

# should not be necessary -> testing
# b4 <- raster("./data/sen_inp/WASP_sen_4_cro_he_c_1_99.tif")
# b3 <- raster("./data/sen_inp/WASP_sen_3_cro_he_c_1_99.tif")
# b8 <- raster("./data/sen_inp/WASP_sen_8_cro_he_c_1_99.tif")
# input_rst <- stack(c(b8,b4, b3))

# maybe it would be easier to set these parameters in the predict script or
# at least parts of it-> testing

size <- c(96, 96)

input <- "input96/"

targetdir <- paste0("./data/hes_pred/", input)

batch_size <- 8

out_path <- "./data/hes_pred/"

source("subscripts/predict_rgb.R")
