# 0. ----------------------------------------------------------------------
##
## script name: control_script.R
##
## purpose of script: script to run parts of the workflow and therefore the
## possibility to set parameters for different testing runs.
##
## author: Jonathan Hecht
##
## date created: 2021-09-11
##
## copyright: -
##
## email: -
##
## notes:
## - tuning_run takes extremely long to start
## - for prediction you have to set some parameters manually
##
## set working directory
wd <- getwd()
setwd(wd)
##
##
options(warn = -1)
##
## load all necessary packages
library(tfruns)
library(raster)


# 1. settings to run the data_split.R script ------------------------------

# setting for different input shapes e.g. c(128, 192....) and for test 1 with fixed shape values
list_shape = c(160)

# percentage of background masks to be added; maybe also include false negative
# values; mask are taken from the whole dataset, so also 10% could be quite a lot

perc = 0.00

source("subscripts/data_split.R")

# 2. settings to run the main_cnn_model.R script --------------------------
# possible flag values
# batch size
batch_size = c(8)
# learning rate
lr = c(1e-02, 1e-03, 1e-04)
# proportion training/validation data
prop1 =  0.8
# number of epochs
epoch = c(10)
# randomly sampled set of parameters to run
sample = 0.02
# factor to reduce the lr when loss does not improve anymore
factor_lr = c(0.5, 0.1)

# settings for the spectral augmentation
bright_d = c(0.1, 0.2)
contrast_lo = c(1, 0.9)
contrast_hi = c(1.2, 1.3)
sat_lo = c(0.9, 1)
sat_hi = c(1.2, 1.3)

# input size of the model e.g. 128,256 and 1 for testing
input = c(1)

tuning_run(
   file = "subscripts/main_cnn_model.R",
   runs_dir = paste0("data/runs", "/", input),
   flags = list(
      epoch = epoch,
      prop1 = prop1,
      lr = lr,
      batch_size = batch_size,
      input = input,
      factor_lr = factor_lr,
      bright_d = bright_d,
      contrast_lo = contrast_lo,
      contrast_hi = contrast_hi,
      sat_lo = sat_lo,
      sat_hi = sat_hi
   ),
   sample = sample,
   confirm = FALSE
)

# compare the results of different runs
ls224 <- ls_runs(runs_dir = "./data/runs/224/old/")

# 3. Settings to run the predict.R script ---------------------------------

# manually set the name of the model folder
name_model <- "sow_unet_model_2021_11_17_14_22"

# some settings
model_path <- paste0("./data/model/", name_model)
osize <- 160
batch_size <- 8
size <- c(osize, osize)

input <- paste0(osize, "/")

targetdir <- paste0("./data/hes_pred/", input)

out_path <- "./data/hes_pred/"

# source the script
source("subscripts/predict.R")
