# PredictCollision

In the src folder, there is a python script that tests all the algorithms for ACC, CACC and PLOEG separately and saves the results into "results" folder

In the data folder, you can find all the data. ACC CACC and PLOEG separately and CACC and PLOEG together.

I couldn't test the algorithms with CACC and PLOEG together because the variable "Controller" is not a numeric value (it is categorical: "CACC" or "PLOEG"). 
Not sure if I should remove it, mixing both results, change it to numbers (1 and 2) or keep them separated.
From ML perspective, I would say the best approach would be to have one algorithm per controller. 
  - Removing "Controller" variable from dataset makes the algorithm think all are the same. We would lose a very important information there.
  - Changing to 1-2 values. We could mantain somehow controller information, but it is not a good practice. Some algorithms can be used for categorical variables, but not all of them. The ones that can't will assume that they are numbers (so, 1>2)
 