#!/usr/bin/env Rscript
library("ggplot2")
library("ggpubr")

avrora <- read.table("table/avrora.csv", head = TRUE, sep = ",")
batik <- read.table("table/batik.csv", head = TRUE, sep = ",")
fop <- read.table("table/fop.csv", head = TRUE, sep = ",")
jython <- read.table("table/jython.csv", head = TRUE, sep = ",")

regression <- function(data) {
    y <- data$count
    x <- data$rank
    m <- lm(log10(y) ~ log10(x))
    p <- ggplot(data, aes(x = rank, y = count)) + geom_point() +
        geom_smooth(method = "lm", se = FALSE) +
        geom_label(data = data,
                   aes(x = Inf, y = Inf,
                       label = paste("R2 =", round(summary(m)$r.squared, digits = 3), sep = " ")),
                   hjust = 1, vjust = 1, size=7) +
        scale_y_log10() +
        scale_x_log10() +
        annotation_logticks(sides = "lb")
                                        # Log base 10 scale + log ticks (on left and bottom side)
    return(p)
}

p_avrora <- regression(avrora)
p_batik <- regression(batik)
p_fop <- regression(fop)
p_jython <- regression(jython)

pdf("output/dacapo.pdf")

ggarrange(p_avrora, p_batik, p_fop, p_jython,
          ncol = 2, nrow = 2,
          labels = c("avrora", "batik", "fop", "jython"),
          font.label=list(color="black", size=11))
