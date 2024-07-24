#!/usr/bin/env Rscript

#Run like this:
#Rscript --vanilla tests/IVIMmodels/unit_tests/analyze.r test_output_priors.csv test_duration_priors.csv
args = commandArgs(trailingOnly=TRUE)
output_name = "test_output.csv"
duration_name = "test_duration.csv"
runPrediction = FALSE
if (length(args)>=1) {
   output_name = args[1]
}
if (length(args)>=2) {
    duration_name = args[2]
}
print(output_name)
print(duration_name)

library(plyr)  # order matters, we need dplyr to come later for grouping
library(dplyr)
library(tidyverse)
library(data.table)


plot_ivim <- function(data, fileExtension) {
    f_plot <- ggplot(data, aes(x=Algorithm)) + geom_boxplot(aes(y=f_fitted)) + geom_boxplot(color="red", aes(y=f)) + facet_grid(SNR ~ Region) + scale_x_discrete(guide = guide_axis(angle = 90)) + ylim(0, 1) + ggtitle("Perfusion fraction grid") + ylab("Perfusion fraction")
    print(f_plot)
    ggsave(paste("f", fileExtension, sep=""), plot=f_plot, width = 50, height = 50, units = "cm")
    D_plot <- ggplot(data, aes(x=Algorithm)) + geom_boxplot(aes(y=D_fitted)) + geom_boxplot(color="red", aes(y=D)) + facet_grid(SNR ~ Region) + scale_x_discrete(guide = guide_axis(angle = 90)) + ylim(0, 0.005) + ggtitle("Diffusion grid") + ylab("Diffusion")
    print(D_plot)
    ggsave(paste("D", fileExtension, sep=""), plot=D_plot, width = 50, height = 50, units = "cm")
    Dp_plot <- ggplot(data, aes(x=Algorithm)) + geom_boxplot(aes(y=Dp_fitted)) + geom_boxplot(color="red", aes(y=Dp)) + facet_grid(SNR ~ Region) + scale_x_discrete(guide = guide_axis(angle = 90)) + ylim(0, 0.25) + ggtitle("Perfusion grid") + ylab("Perfusion")
    print(Dp_plot)
    ggsave(paste("Dp", fileExtension, sep=""), plot=Dp_plot, width = 50, height = 50, units = "cm")
}

data <- read.csv(output_name)
data <- data %>% mutate_if(is.character, as.factor)
plot_ivim(data, ".pdf")

data_restricted <- data[data$Region %in% c("Liver", "spleen", "Right kydney cortex", "right kidney medulla"),]
plot_ivim(data_restricted, "_limited.pdf")

data_duration <- read.csv(duration_name)
data_duration <- data_duration %>% mutate_if(is.character, as.factor)
data_duration$ms <- data_duration$Duration..us./data_duration$Count/1000
duration_plot <- ggplot(data_duration, aes(x=Algorithm, y=ms)) + geom_boxplot() + scale_x_discrete(guide = guide_axis(angle = 90)) + ggtitle("Fit Duration") + ylab("Time (ms)")
ggsave("durations.pdf", plot=duration_plot, width = 20, height = 20, units = "cm")


if (runPrediction) {
    # Then this widens it so we can lm()
    data_wide <- data %>% pivot_wider(names_from=Algorithm, values_from=c(f_fitted, Dp_fitted, D_fitted), id_cols=c(Region, SNR, index, f, D, Dp))
    # linear fit for f
    f_model <- lm(f ~ f_fitted_IAR_LU_biexp + f_fitted_IAR_LU_modified_mix + f_fitted_IAR_LU_modified_topopro + f_fitted_IAR_LU_segmented_2step + f_fitted_IAR_LU_segmented_3step + f_fitted_IAR_LU_subtracted, data=data_wide)
    #f_model <- lm(f ~ f_fitted_ETP_SRI_LinearFitting + f_fitted_IAR_LU_biexp + f_fitted_IAR_LU_modified_mix + f_fitted_IAR_LU_modified_topopro + f_fitted_IAR_LU_segmented_2step + f_fitted_IAR_LU_segmented_3step + f_fitted_IAR_LU_subtracted, data=data_new_wide)
    D_model <- lm(D ~ D_fitted_ETP_SRI_LinearFitting + D_fitted_IAR_LU_biexp + D_fitted_IAR_LU_modified_topopro + D_fitted_IAR_LU_segmented_2step + D_fitted_IAR_LU_segmented_3step + D_fitted_IAR_LU_subtracted, data=data_wide)
    # D_model <- lm(D ~ D_fitted_ETP_SRI_LinearFitting + D_fitted_IAR_LU_biexp + D_fitted_IAR_LU_modified_mix + D_fitted_IAR_LU_modified_topopro + D_fitted_IAR_LU_segmented_2step + D_fitted_IAR_LU_segmented_3step + D_fitted_IAR_LU_subtracted, data=data_new_wide)
    Dp_model <- lm(Dp ~ Dp_fitted_IAR_LU_biexp + Dp_fitted_IAR_LU_modified_mix + Dp_fitted_IAR_LU_modified_topopro + Dp_fitted_IAR_LU_segmented_2step + Dp_fitted_IAR_LU_segmented_3step + Dp_fitted_IAR_LU_subtracted, data=data_wide)
    # Dp_model <- lm(Dp ~ Dp_fitted_ETP_SRI_LinearFitting + Dp_fitted_IAR_LU_biexp + Dp_fitted_IAR_LU_modified_mix + Dp_fitted_IAR_LU_modified_topopro + Dp_fitted_IAR_LU_segmented_2step + Dp_fitted_IAR_LU_segmented_3step + Dp_fitted_IAR_LU_subtracted, data=data_new_wide)
    # predict new data from existing model
    predict(object = f_model, newdata = data_wide)
}


ivim_decay <- function(f, D, Dp, bvalues) {
    return(f*exp(-Dp*bvalues) + (1-f)*exp(-D*bvalues))
}
#TODO: read bvalues from file
bvalues <- c(0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 250.0, 350.0, 400.0, 550.0, 700.0, 850.0, 1000.0)

generate_curves <- function(data, bvalues, appended_string) {
    curves <-apply(data[,c(paste("f", appended_string, sep=""), paste("D", appended_string, sep=""), paste("Dp", appended_string, sep=""))], 1, function(x) ivim_decay(x[1], x[2], x[3], bvalues))
    curves <- transpose(data.frame(curves))
    colnames(curves) <- paste("b", bvalues, sep="_")
    data_curves <- cbind(data, curves)
    data_curves <- pivot_longer(data_curves, starts_with("b_"), names_to="bvalue", values_to=paste("signal", appended_string, sep=""), names_prefix="b_", names_transform=list(bvalue = as.numeric))
    return(data_curves)
}
curves_restricted_fitted <- generate_curves(data_restricted, bvalues, "_fitted")
curves_restricted <- generate_curves(data_restricted, bvalues, "")

data_curves_restricted <- cbind(curves_restricted, signal_fitted=curves_restricted_fitted$signal_fitted)
curve_plot <- ggplot(data_curves_restricted, aes(x=bvalue))  + facet_grid(Region ~ SNR) + geom_line(alpha=0.2, aes(y=signal_fitted, group=interaction(Algorithm, index), color=Algorithm)) + geom_line(aes(y=signal)) + ylim(0, 1) + ylab("Signal (a.u.)")
print(curve_plot)
ggsave("curve_plot.pdf", plot=curve_plot, width = 30, height = 30, units = "cm")


data_points_restricted <- data_restricted[data_restricted$index==0,]
data_points_restricted <- pivot_longer(data_points_restricted, starts_with("bval_"), names_to="bvalue", values_to="fitted_data", names_prefix="bval_", names_transform=list(bvalue = as.numeric))
# data_curves_restricted_idx0 <- data_curves_restricted[data_curves_restricted$index==0,]
data_points_restricted <- cbind(data_curves_restricted[data_curves_restricted$index==0,], fitted_data=data_points_restricted$fitted_data)
# fitted_curves <- ggplot(data_points_restricted, aes(x=bvalue))  + facet_grid(Region ~ SNR) + geom_line(alpha=0.5, aes(y=signal_fitted, group=Algorithm, color=Algorithm)) + geom_point(aes(y=fitted_data, group=Algorithm, color=Algorithm)) + ylim(0, 1) + ylab("Signal (a.u.)")
fitted_curves <- ggplot(data_points_restricted, aes(x=bvalue))  + facet_grid(Region ~ SNR) + geom_line(alpha=0.5, aes(y=signal_fitted, group=Algorithm, color=Algorithm)) + geom_point(aes(y=fitted_data, shape="Data")) + ylim(0, 1) + ylab("Signal (a.u.)") + guides(shape = guide_legend("Fitted Data"))
print(fitted_curves)
ggsave("fitted_curves.pdf", plot=fitted_curves, width = 30, height = 30, units = "cm")
