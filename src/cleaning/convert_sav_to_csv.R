#' convert_sav_to_csv.R
#' Author William van Doorn
#'
#' This R-script was generated to convert the sepsis.SAV database to sepsis.CSV database
#' This is easier to read in Python, and thus we used a simple R-script to convert
#' Last edited: 23/03/2019

# load library
library(haven)

# set working directory
setwd('/home/wptmdoorn/2018_Sepsis/data/raw')

# print all files
print(list.files())

# read data and print dimensions and columns
data <- read_sav("sepsis.sav")
print(dim(data))
print(names(data))

# save it as sepsis.csv seperated by a comma
write.table(data, file='sepsis.csv', sep=',')