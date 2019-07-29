# Analyze Website Landing Page A/B Test Results
Author: Libin Guo

The Goal are
- use A/B hypothesis testing in python to help company find out whether the new website landing page has higher users’ convert rate than old page using Pandas and Matplotlib. 
- perform regression analysis using two method -- Using loop to similate 10,000 experiment using numpy.random.choice() function and another method is to use StatsModel package to run logistic regression model.
- merge another dataset and measure effect of country factor using sklearn and interpret results with practical reasoning

- Introduction
- Part I - Probability
- Part II - A/B Test
- Part III - Regression

Finding:
 1. 
The p-value associated with ab_page is 0.190. The null in c-e part is that there is no difference between the treatment and control group. Alternative hypotheses is that there is difference between between the treatment and control group

Part II assumes the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, compared to question c-e,they have different explainory varibale or factor for the result.

2. Country Factors:
I create Logit regression model for conveted and country, and us CA and old page as baseline
```For every unit for new_page decrease, convert will be 1.5% more likely to happen, holding all other varible constant.
For every unit for UK increases, convert is 5.2% more to happen, holding all other varible constant.
For every unit for US increases, convert is 4.2% more to happen, holding all other varible constant.
```
