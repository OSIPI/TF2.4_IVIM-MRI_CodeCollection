library(tidyverse)
library(plyr)

plot_ivim <- function(data, fileExtension) {
    ggplot(data, aes(x=Algorithm)) + geom_boxplot(aes(y=f_fitted)) + geom_boxplot(color="red", aes(y=f)) + facet_grid(SNR ~ Region) + scale_x_discrete(guide = guide_axis(angle = 90)) + ylim(0, 1) + ggtitle("Perfusion fraction grid") + ylab("Perfusion fraction")
    ggsave(paste("f", fileExtension, sep=""), width = 50, height = 50, units = "cm")
    ggplot(data, aes(x=Algorithm)) + geom_boxplot(aes(y=D_fitted)) + geom_boxplot(color="red", aes(y=D)) + facet_grid(SNR ~ Region) + scale_x_discrete(guide = guide_axis(angle = 90)) + ggtitle("Diffusion grid") + ylab("Diffusion")
    ggsave(paste("D", fileExtension, sep=""), width = 50, height = 50, units = "cm")
    ggplot(data, aes(x=Algorithm)) + geom_boxplot(aes(y=Dp_fitted)) + geom_boxplot(color="red", aes(y=Dp)) + facet_grid(SNR ~ Region) + scale_x_discrete(guide = guide_axis(angle = 90)) + ylim(0, 0.25) + ggtitle("Perfusion grid") + ylab("Perfusion")
    ggsave(paste("Dp", fileExtension, sep=""), width = 50, height = 50, units = "cm")
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

