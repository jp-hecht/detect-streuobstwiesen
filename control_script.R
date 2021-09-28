## ---------------------------
##
## Script name: control_script.R
##
## Purpose of script: Script to run parts of the workflow and therefore the
## possibility to set parameters for different testing runs.
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
## - tuning_run takes extremely long to start (on HDD or GPU?)
## - for prediction you have to set some parameters manually
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
library(tfruns)
library(raster)
##

# script to generate inputs for the model and later predictions ----------------

# setting for different input shapes e.g. c(128, 192....) and for test 1 with fixed shape values
list_shape = c(256)

# percentage of black/zero mask to be added; maybe also include false negative
# values; mask are taken from whole Hesse, so also 10% could be quite a lot 

perc = 0.000
# time:  ~15:10 - 16:10; for 1 (test) 
# time: ~16:19 -17:02 for 256
# source("subscripts/data_split.R")

# script to run & evaluate the model  ------------------------------------------

# possible  (Hyper-)parameters to set --> Flags

# batch size
batch_size = c(6, 8)
# learning rate
lr = c(0.0001)
# proportion training/testing/validation data
prop1 =  0.8
prop2 =  0.9
# number of epochs
epoch = c(10)
# randomly sampled set of parameters
sample = 0.001
# factor to reduce the lr when loss does not improve anymore
factor_lr = c(0.1, 0.3, 0.5)
# convolutional blocks to freeze the pretrained model
# block_freeze = c("block1_pool")
# settings for the spectral augmentation
bright_d = c(0.1, 0.3)
contrast_lo = c(0.9, 0.8)
contrast_hi = c(1.2, 1.4)
sat_lo = c(0.9, 0.8)
sat_hi = c(1.2, 1.4)

# for Sigmoidfocalloss
# gamma = c(1, 3, 4)
# alpha =  c(0.1, 0.7)

# currently it is better to just set one input shape <--> conflicts with tuning_run
input = c(1) # test: 1 and everything else 128 192...

tuning_run(
   file = "subscripts/main_cnn_model.R",
   runs_dir = paste0("data/runs", "/", input),
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

# 17:05 - 17:55 with 10 epochs on the testing set

# comparing the results in each folder -----------------------------------------
# e.g.
# ls96 <- ls_runs()

# predict  ---------------------------------------------------------------------

# manually set the name of the model folder to predict
name_model <- "sow_unet_model_2021_09_22_08_55"


model_path <- paste0("./data/model/", name_model)

# maybe it would be easier to set these parameters in the predict script or
# at least parts of it-> testing

osize <- 128
batch_size <- 8
size <- c(osize, osize)

input <- paste0("input",osize,"/")

targetdir <- paste0("./data/hes_pred/", input)

out_path <- "./data/hes_pred/"

source("subscripts/predict.R")
