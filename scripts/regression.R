#!/usr/bin/env Rscript
library(ggplot2)
library(dplyr)
args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 0 || length(args) > 1) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
}

file <- args[1]
output <- gsub(".csv", ".pdf", basename(file))

pdf(output)

data <- read.table(file, head = TRUE, sep = ",")
y <- data$count
x <- data$rank
# Perform power regression
m <- lm(log10(y) ~ log10(x))
# plot(x, y, log = "xy")
# abline(m, col = "blue")
# legend("bottomleft", legend=paste("R^2 is", format(summary(m)$r.squared, digits = 3)))
# Do same thing below by using ggplot2
p <- ggplot(data, aes(x = rank, y = count)) + geom_point() + theme(text = element_text(size = 20)) +
    geom_smooth(method = "lm", se = FALSE) +
    geom_label(data = data,
               aes(x = Inf, y = Inf,
                   label = paste("R2 =", round(summary(m)$r.squared, digits = 3), sep = " ")),
               hjust = 1, vjust = 1, size=8)
# Log base 10 scale + log ticks (on left and bottom side)
p + scale_y_log10() +
    scale_x_log10() +
    annotation_logticks(sides = "lb")
