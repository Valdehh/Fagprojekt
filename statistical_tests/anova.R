library(reticulate)
np <- import("numpy")

##################################################################################################
# SEMI-SUPERVISED
##################################################################################################

semi10 <- np$load("statistical_tests/seeds/10semi_bbbc/test_ELBOs.npz")
semi30 <- np$load("statistical_tests/seeds/30semi_bbbc/test_ELBOs.npz")
semi42 <- np$load("statistical_tests/seeds/42semi_bbbc/test_ELBOs.npz")

data1 <- semi10$f[["test_ELBOs"]]
data2 <- semi30$f[["test_ELBOs"]]
data3 <- (semi42$f[["test_ELBOs"]])[0:100]  # same size as the other two (doesnt matter though)


seeds <- factor(c(rep(10, 100), rep(30, 100), rep(42, 100)))

df <- data.frame(ELBO = c(data1, data2, data3), seed = seeds)


res.aov <- aov(ELBO ~ seed, data = df)
summary(res.aov)

##################################################################################################
# Plotting the histograms: (they look fairly normal)
##################################################################################################

library(ggplot2)

df1 <- data.frame(ELBO = data1)
ggplot(df1, aes(x = ELBO)) + 
    geom_histogram(aes(y =after_stat(density))) + stat_function(fun = dnorm, args = list(mean = mean(df1$ELBO), sd = sd(df1$ELBO))) + ggtitle("seed = 10")
ggsave("semi1.png", width = 10, height = 10, units = "cm") 

df2 <- data.frame(ELBO = data2)
ggplot(df2, aes(x = ELBO)) + 
    geom_histogram(aes(y =after_stat(density))) + stat_function(fun = dnorm, args = list(mean = mean(df2$ELBO), sd = sd(df2$ELBO))) + ggtitle("seed = 30")
ggsave("semi2.png", width = 10, height = 10, units = "cm")


df3 <- data.frame(ELBO = data3)
ggplot(df3, aes(x = ELBO)) + 
    geom_histogram(aes(y =after_stat(density))) + stat_function(fun = dnorm, args = list(mean = mean(df3$ELBO), sd = sd(df3$ELBO))) + ggtitle("seed = 42")
ggsave("semi3.png", width = 10, height = 10, units = "cm") 


##################################################################################################
# means
mean(data1)
mean(data2)
mean(data3)

##################################################################################################
# VANILLA-VAE
##################################################################################################

van10 <- np$load("statistical_tests/seeds/10van/test_ELBOs.npz")
van30 <- np$load("statistical_tests/seeds/30van/test_ELBOs.npz")
van42 <- np$load("statistical_tests/seeds/42van/test_ELBOs.npz")

data4 <- van10$f[["test_ELBOs"]]
data5 <- van30$f[["test_ELBOs"]]
data6 <- (van42$f[["test_ELBOs"]])[0:100]

seeds <- factor(c(rep(10, 100), rep(30, 100), rep(42, 100)))

df <- data.frame(ELBO = c(data4, data5, data6), seed = seeds)

res.aov2 <- aov(ELBO ~ seed, data = df)
summary(res.aov2)

mean(data4)
mean(data5)
mean(data6)

##################################################################################################
# Plotting the histograms: (they look fairly normal)
##################################################################################################

df4 <- data.frame(ELBO = data4)
ggplot(df4, aes(x = ELBO)) + 
    geom_histogram(aes(y =after_stat(density))) + stat_function(fun = dnorm, args = list(mean = mean(df4$ELBO), sd = sd(df4$ELBO))) + ggtitle("seed = 10")
ggsave("van1.png", width = 10, height = 10, units = "cm") 

df5 <- data.frame(ELBO = data5)
ggplot(df5, aes(x = ELBO)) + 
    geom_histogram(aes(y =after_stat(density))) + stat_function(fun = dnorm, args = list(mean = mean(df5$ELBO), sd = sd(df5$ELBO))) + ggtitle("seed = 30")
ggsave("van2.png", width = 10, height = 10, units = "cm")


df6 <- data.frame(ELBO = data6)
ggplot(df6, aes(x = ELBO)) + 
    geom_histogram(aes(y =after_stat(density))) + stat_function(fun = dnorm, args = list(mean = mean(df6$ELBO), sd = sd(df6$ELBO))) + ggtitle("seed = 42")
ggsave("van3.png", width = 10, height = 10, units = "cm") 


##################################################################################################