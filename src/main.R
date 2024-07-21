Sys.setenv(TZ='GMT')

# library(lubridate)
# library(dplyr)
library(ggplot2)
# library(caret)
# library(forecast)


# Load required libraries
# library(forecast)
# library(tidyverse)
# library(caret)

# Read in the data
data <- read.csv("../dataset/data_without_velocity.csv", 
                 colClasses = c(MedidasCCVStatusId = "integer", 
                                SensorCCVId = "integer", 
                                VehicleTypeId = "integer",
                                Timestamp = "character",
                                Fluxo = "integer",
                                Velocidade = "NULL",
                                EstadodeTrafego = "NULL"))

# Convert the Timestamp column to a datetime object
data$Timestamp <- as.POSIXct(data$Timestamp, format="%Y-%m-%d %H:%M:%S")


model <- plot(data$Timestamp, data$Fluxo,type="l", main="Dataset 1 without velocity", xlab="Timestamp", ylab="Fluxo", sub = "Aggregation Time - 5 minutes")
print(model)


data2 <- read.csv("../dataset/data_with_velocity.csv", 
                 colClasses = c(MedidasCCVStatusId = "integer", 
                                SensorCCVId = "integer", 
                                VehicleTypeId = "integer",
                                Timestamp = "character",
                                Fluxo = "integer",
                                Velocidade = "integer",
                                EstadodeTrafego = "NULL"))

data2$Timestamp <- as.POSIXct(data2$Timestamp, format="%Y-%m-%d %H:%M:%S")

model <- plot(data2$Timestamp, data2$Fluxo,type="l", main="Dataset 2 with velocity", xlab="Timestamp",ylab="Fluxo", sub = "Aggregation Time - 1 minute")
print(model)

model <- plot(data2$Timestamp, data2$Velocidade,type="l", main="Dataset 2 with velocity", xlab="Timestamp", ylab="Velocidade", sub = "Aggregation Time - 1 minute")
print(model)


plot(data2[, 4:6], main = "Correlation plot for dataset 2")
print(model)

# # Create the relationship model.
# model <- lm(Fluxo~Timestamp+VehicleTypeId, data = data)


# # Show the model.
# print(model)

# # Get the Intercept and coefficients as vector elements.
# cat("# # # # The Coefficient Values # # # ","\n")

# a <- coef(model)[1]
# print(a)

# Xdisp <- coef(model)[2]
# Xhp <- coef(model)[3]
# Xwt <- coef(model)[4]

# print(Xdisp)
# print(Xhp)
# print(Xwt)

# # Plot the flux of vehicles over time
# # Get unique vehicle types
# vehicle_types <- unique(data$VehicleTypeId)

# # Loop over vehicle types and create separate plots
# plots <- lapply(vehicle_types, function(vt) {
#   # Filter data for current vehicle type
#   data_subset <- subset(data, VehicleTypeId == vt & Timestamp >= start_date & Timestamp < end_date)
  

#   # Create plot for current vehicle type
#   plot <- ggplot(data_subset, aes(x=Timestamp, y=Fluxo)) +
#     geom_line() +
#     scale_x_datetime(date_breaks = "1 day", date_labels = "%d %b") +
#     xlab("Time") + ylab("Fluxo") +
#     ggtitle(paste("Traffic Flux for Vehicle Type", vt))

#   return(plot)
# })

# # Print plots

# for (p in plots) {
#   print(p)
# }


