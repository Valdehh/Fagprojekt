# For appendix A (grid search)

library(reticulate)
np <- import("numpy")

npz1 <- np$load("statistical_tests/final_grid/100/test_ELBOs.npz")
npz2 <- np$load("statistical_tests/final_grid/200/test_ELBOs.npz")
npz3 <- np$load("statistical_tests/final_grid/300/test_ELBOs.npz")
npz4 <- np$load("statistical_tests/final_grid/400/test_ELBOs.npz")
npz5 <- np$load("statistical_tests/final_grid/500/test_ELBOs.npz")

data1 <- npz1$f[["test_ELBOs"]]
data2 <- npz2$f[["test_ELBOs"]]
data3 <- npz3$f[["test_ELBOs"]]
data4 <- npz4$f[["test_ELBOs"]]
data5 <- npz5$f[["test_ELBOs"]]

t.test(data1, data2)
t.test(data1, data3)
t.test(data1, data4)
t.test(data1, data5)
t.test(data2, data3)

t.test(data2, data4)
t.test(data2, data5)
t.test(data3, data4)
t.test(data3, data5)
t.test(data4, data5)




