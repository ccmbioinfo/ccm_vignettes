# Random number generation

# stats packages that comes with base R installation has many of the most commonly
# used distributions, additional distributions can be found in external R packages
# that you can download from CRAN or github

# so we all get the same result
set.seed(42)

# generate a four populations from a normal distribution
x<-rnorm(n=1000, mean=0, sd=1)
y<-rnorm(n=1000, mean=4, sd=1)
z<-rnorm(n=1000, mean=0, sd=4)
w<-rnorm(n=1000, mean=4, sd=4)

#create a 2x2 histogram to look at the populations

par(mfrow=c(2,2))
hist(x)
hist(y)
hist(z)
hist(w)

# t test

# compare populations
t.test(x, z)
t.test(x, y)

# one sample test
t.test(x, mu=0)

# parierd t.test
t.test(x, z, paired=T)

# one sided test
t.test(x, y, alternative = "less")
t.test(x, y, alternative = "greater")

# wilcox test

# two sample
wilcox.test(x, y)

# one sample
wilcox.test(x, mu=0)


# ks test
# two sample
ks.test(x, y)

# compare to a known distribution
ks.test(x, "pnorm", mean=0, sd=1)


# fisher's exact test
#create a matrix

milk_and_tea<-matrix(c(30, 10, 10, 30),
                     nrow = 2,
                     dimnames = list(Guess = c("Milk", "Tea"),
                                     Truth = c("Milk", "Tea")))

milk_and_tea

# perform the test
fisher.test(milk_and_tea)
fisher.test(milk_and_tea, or=9)
fisher.test(milk_and_tea, alternative="less")

# chisq test
m <- matrix(c(100, 50, 200, 100, 300, 150), nrow=2)
m
chisq.test(m)

m <- matrix(c(100, 500, 200, 100, 300, 150), nrow=2)
m
chisq.test(m)

# effect sizes

#cohen's d
#install.packages("effsize")

library(effsize)

cohen.d(x, y)
cohen.d(x, z)


# cliff's delta
cliff.delta(x, y)
cliff.delta(x, z)


# one way anova and kruskall-wallis

# anova
# need to create a "long format" data frame

dists<-data.frame(values=c(x,y,z), 
                  group=c(rep("X", 1000), rep("Y", 1000), rep("Z", 1000)))

oneway.test(values~group, dists)

x2<-rnorm(1000, mean=0, sd=12)
dists2<-data.frame(values=c(x,x2,z), 
                  group=c(rep("X", 1000), rep("X2", 1000), rep("Z", 1000)))

oneway.test(values~group, dists2)

# krustal wallis

kruskal.test(values~group, dists)
kruskal.test(values~group, dists2)

# calculating values across rows and columns of a dataframe or matrix
# create a large matrix

big_mat<-matrix(rnorm(n=80000), ncol=16)
row_means<-rowMeans(big_mat)
col_means<-colMeans(big_mat)

# install.packages("matrixStats")
# a list of calculations covered in matrixStats package https://cran.rstudio.com/web/packages/matrixStats/vignettes/matrixStats-methods.html
library(matrixStats)
row_medians<-rowMedians(big_mat)


# introduction to lapply, sapply, apply

# apply row and column wise
row_sds<-apply(big_mat, 1, sd)
col_sds<-apply(big_mat, 2, sd)

# lapply on a list
mylist<-list(a=x, b=y, c=z, d=w)
lapply(mylist, sd)

sapply(mylist, median)
#same as above
unlist(lapply(mylist, median))

# performing statistical tests programatically

# loop

# this loop is rather rudimentary it contains hard coded values based on the data
# can write a more robust loop

#create a vector of NAs but same length as what we want
pvals_loop<-rep(NA, nrow(big_mat))

for(i in 1:nrow(big_mat)){
  test<-t.test(big_mat[i, 1:8], big_mat[i, 9:16])
  pval<-test$p.value
  pvals_loop[i]<-pval
}


# apply
# we need to create a small function that takes 8 values and compares the first 4 to 
# the last 4 and returns the p value

test_one_by_one<-function(row, test, groups=list(1:8, 9:16), param="p.value"){
  results<-test(row[groups[[1]]], row[groups[[2]]])
  pval<-results[[param]]
  return(pval)
}

pvals_apply<-apply(big_mat, 1, test_one_by_one, test=t.test)

# Are they identical?
mean(pvals_apply==pvals_loop)

# multiple testing corrections

# the fraction of "significant" results
mean(pvals_loop<0.05)


#since the pvals are identical I will be using the loop one
# bonferroni

bonferroni_adjusted<-p.adjust(pvals_loop, method = "bonferroni")
mean(bonferroni_adjusted<0.05)

# fdr

fdr_adjusted<-p.adjust(pvals_loop, method = "fdr")
mean(fdr_adjusted<0.05)