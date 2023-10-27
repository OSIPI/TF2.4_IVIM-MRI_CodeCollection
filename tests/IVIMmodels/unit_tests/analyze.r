library(plyr)  # order matters, we need dplyr to come later for grouping
library(dplyr)
library(tidyverse)


plot_ivim <- function(data, fileExtension) {
    f_plot <- ggplot(data, aes(x=Algorithm)) + geom_boxplot(aes(y=f_fitted)) + geom_boxplot(color="red", aes(y=f)) + facet_grid(SNR ~ Region) + scale_x_discrete(guide = guide_axis(angle = 90)) + ylim(0, 1) + ggtitle("Perfusion fraction grid") + ylab("Perfusion fraction")
    print(f_plot)
    ggsave(paste("f", fileExtension, sep=""), plot=f_plot, width = 50, height = 50, units = "cm")
    D_plot <- ggplot(data, aes(x=Algorithm)) + geom_boxplot(aes(y=D_fitted)) + geom_boxplot(color="red", aes(y=D)) + facet_grid(SNR ~ Region) + scale_x_discrete(guide = guide_axis(angle = 90)) + ggtitle("Diffusion grid") + ylab("Diffusion")
    print(D_plot)
    ggsave(paste("D", fileExtension, sep=""), plot=D_plot, width = 50, height = 50, units = "cm")
    Dp_plot <- ggplot(data, aes(x=Algorithm)) + geom_boxplot(aes(y=Dp_fitted)) + geom_boxplot(color="red", aes(y=Dp)) + facet_grid(SNR ~ Region) + scale_x_discrete(guide = guide_axis(angle = 90)) + ylim(0, 0.25) + ggtitle("Perfusion grid") + ylab("Perfusion")
    print(Dp_plot)
    ggsave(paste("Dp", fileExtension, sep=""), plot=Dp_plot, width = 50, height = 50, units = "cm")
}

data <- read.csv("test_output.csv")
data <- data %>% mutate_if(is.character, as.factor)
plot_ivim(data, ".pdf")

data_restricted <- data[data$Region %in% c("Liver", "spleen", "Right kydney cortex", "right kidney medulla"),]
plot_ivim(data_restricted, "_limited.pdf")

data_duration <- read.csv("test_duration.csv")
data_duration <- data_duration %>% mutate_if(is.character, as.factor)
data_duration$ms <- data_duration$Duration..us./data_duration$Count/1000
ggplot(data_duration, aes(x=Algorithm, y=ms)) + geom_boxplot() + scale_x_discrete(guide = guide_axis(angle = 90)) + ggtitle("Fit Duration") + ylab("Time (ms)")
ggsave("durations.pdf", width = 20, height = 20, units = "cm")


# some kind of black voodoo magic to number unique groups
# data_new <- data %>% group_by(Algorithm, Region, SNR) %>% mutate(index = row_number()) %>% ungroup()
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
