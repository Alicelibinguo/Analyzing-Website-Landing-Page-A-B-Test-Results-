
# coding: utf-8

# ## Analyze A/B Test Results
# 
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists. 
# 
# For this project, I will be working to understand the results of an A/B test run by an e-commerce website.  My goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[177]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`. 
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[178]:


df=pd.read_csv('ab_data.csv')
df.head()


# b. Use the below cell to find the number of rows in the dataset.

# In[179]:


total_users = float(df.shape[0])
total_users 


# c. The number of unique users in the dataset.

# In[180]:


df['user_id'].nunique()


# d. The proportion of users converted.

# In[181]:


users_converted=float(df.query('converted == 1')['user_id'].nunique())
p1 = (users_converted/total_users)
print("The proportion of users converted is {0:.2%}".format(p1))


# e. The number of times the `new_page` and `treatment` don't line up.

# In[182]:


df.query('(group == "treatment" and landing_page != "new_page") or (group != "treatment" and landing_page == "new_page")')['user_id'].count()


# f. Do any of the rows have missing values?

# In[183]:


df.isnull().values.any()
#The result shows there is no missing values


# `2.` For the rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to provide how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[184]:


df2 = df.drop(df.query('(group == "treatment" and landing_page != "new_page") or (group != "treatment" and landing_page == "new_page") or (group == "control" and landing_page != "old_page") or (group != "control" and landing_page == "old_page")').index)


# In[185]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[186]:


df2['user_id'].nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[187]:


df2[df2.duplicated(['user_id'], keep=False)]['user_id']
#User_id 773192 is repeated


# c. What is the row information for the repeat **user_id**? 

# In[188]:


df2[df2['user_id'] == 773192]
#Two rows has dfferent timestamp and other columns are the same 


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[189]:


df2 = df2.drop(df2[(df2.user_id == 773192) & (df2['timestamp'] == '2017-01-09 05:37:58.781806')].index)
df2[df2['user_id'] == 773192]


# `4.` Use **df2** in the below cells to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[190]:


converted_users2 = float(df2.query('converted == 1')['user_id'].nunique())
p2 = converted_users2/float(df2.shape[0])
print("The probability of an individual converting regardless of the page they receive is {0:.2%}".format(p2))


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[191]:


converted_controlusers2 = float(df2.query('converted == 1 and group == "control"')['user_id'].nunique())
control_users2 =float(df2.query('group == "control"')['user_id'].nunique())
cp2 = converted_controlusers2 /control_users2
print(" Given that an individual was in the control group, the probability they converted is {0:.2%}".format(cp2))


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[192]:


converted_controlusers2 = float(df2.query('converted == 1 and group == "treatment"')['user_id'].nunique())
treat_users2 =float(df2.query('group == "treatment"')['user_id'].nunique())
tp2 = converted_controlusers2 /treat_users2
print(" Given that an individual was in the treatment group, the probability they converted is {0:.2%}".format(tp2))


# d. What is the probability that an individual received the new page?

# In[193]:


new_page_users2 = float(df2.query('landing_page == "new_page"')['user_id'].nunique())
Newpage_p2 = new_page_users2/float(df2.shape[0])
print("The probability that an individual received the new page is {0:.2%}".format(Newpage_p2))


# In[194]:


new_c2 = float(df2.query('converted == 1 and  landing_page == "new_page"')['user_id'].nunique())
new_users2 =float(df2.query('landing_page == "new_page"')['user_id'].nunique())

print(" Given that an individual was in new landing page, the probability they converted is {0:.2%}".format(new_c2 /new_users2))


# e. Use the results in the previous two portions of this question to suggest if you think there is evidence that one page leads to more conversions?  Write your response below.

# The probability of an individual converting regardless of the page they receive is 11.96%,
# Given that an individual was in the control group, the probability they converted is 12.04%
# Given that an individual was in the treatment group, the probability they converted is 11.88%.
# The probablity users converted in both control and treatment group are pretty similar to each other and  probability of an individual converting regardless of the page they receive. therefore, there is no evidence that ne page leads to more conversions.

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, I could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do i stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do I run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# Null hypothese is H0: p_new - p_old <= 0  
# Alternative hypothese is H1: p_new - p_old > 0 

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - I am going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure I am on the right track.<br><br>

# Here we are looking at a null where there is no difference in conversion based on the page, which means the conversions for each page are the same.

# a. What is the **convert rate** for $p_{new}$ under the null? 

# In[240]:


# Compute converted success rate, which equals to the converted success rate regardless of page
p_new = round(float(df2.query('converted == 1')['user_id'].nunique())/float(df2['user_id'].nunique()),4)

# Display converted success rate
p_new


# b. What is the **convert rate** for $p_{old}$ under the null? <br><br>

# In[241]:


# Compute old converted success rate, which equals to the converted success rate regardless of page
p_old = round(float(df2.query('converted == 1')['user_id'].nunique())/float(df2['user_id'].nunique()),4)

# Display old converted success rate
p_old


# c. What is $n_{new}$?

# In[214]:


#Compute the number of unique users who has new page using df2 dataframe
N_new = df2.query('landing_page == "new_page"')['user_id'].nunique()

#display the number of unique users who has new page
N_new 


# d. What is $n_{old}$?

# In[215]:


#Compute the number of unique users who has old page  using df2 dataframe
N_old = df2.query('landing_page == "old_page"')['user_id'].nunique() 
#display the number of unique users who has new page
N_old 


# e. Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[309]:


#Simulate  n_new  transactions with a convert rate of  p_new  under the null
new_page_converted = np.random.choice([0,1],N_new, p=(p_new,1-p_new))

#Display new_page_converted
new_page_converted 


# f. Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[310]:


#Simulate  n_old  transactions with a convert rate of  p_old  under the null
old_page_converted = np.random.choice([0,1],N_old, p=(p_old,1-p_old))

#Display old_page_converted
old_page_converted


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[218]:


#Compute the probablity of new page converted rate
new_page_converted.mean()


# In[219]:


#Compute the probablity of old page converted rate
old_page_converted.mean()


# In[220]:


#Find  pnewpnew  -  poldpold  for your simulated values from part (e) and (f).
new_page_converted.mean() - old_page_converted.mean()


# h. Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts **a. through g.** above.  Store all 10,000 values in **p_diffs**.

# In[320]:


#Import timeit package
import timeit
start = timeit.default_timer()

# Create sampling distribution for difference in completion rates
# with boostrapping
p_diffs = []
size = df.shape[0]
for _ in range(10000):
    samp = df2.sample(size, replace = True)
    new_page_converted = np.random.choice([0,1],N_new, p=(p_new,1-p_new))
    old_page_converted = np.random.choice([0,1],N_old, p=(p_old,1-p_old))
    p_diffs.append(new_page_converted.mean() - old_page_converted.mean())
    
#Compute python running time.
stop = timeit.default_timer()
print stop - start 


# In[321]:


p_diffs = np.array(p_diffs)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[322]:


plt.hist(p_diffs)


# In[352]:


# Create number of users with all new_page users and all new_page users
convert_new = df2.query('converted == 1 and landing_page == "new_page"')['user_id'].nunique()
convert_old = df2.query('converted == 1 and landing_page == "old_page"')['user_id'].nunique()

# Compute actual converted rate
actual_cvt_new = float(convert_new)/ float(n_new) 
actual_cvt_old = float(convert_old)/ float(n_old)


# In[353]:


# Compute observed difference in converted rate 
obs_diff = actual_cvt_new - actual_cvt_old

# Display observed difference in converted rate 
obs_diff


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[354]:


# create distribution under the null hypothesis
null_vals = np.random.normal(0, p_diffs.std(), p_diffs.size)


# In[355]:


#Plot Null distribution
plt.hist(null_vals)
#Plot vertical line for observed statistic
plt.axvline(x=obs_diff,color ='red')


# In[375]:


#Compute proportion of the p_diffs are greater than the actual difference observed in ab_data.csv
(null_vals > obs_diff).mean()


# k. In words, explain what you just computed in part **j.**.  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# Type I error rate of 5%, and Pold > Alpha, we fail to reject the null.
# Therefore, the data show, with a type I error rate of 0.05, that the old page has higher probablity of convert rate than new page.

# P-Value: The probability of observing our statistic or a more extreme statistic from the null hypothesis.

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[376]:


#Import regression model library
import statsmodels.api as sm


# In[377]:


convert_old,convert_new,n_old,n_new


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](http://knowledgetack.com/python/statsmodels/proportions_ztest/) is a helpful link on using the built in.

# In[378]:


z_score, p_value = sm.stats.proportions_ztest(np.array([convert_new,convert_old]),np.array([n_new,n_old]), alternative = 'larger')


# In[380]:


z_score, p_value
# it's a one tail test so a z-score past 1.96 will be significant.


# In[381]:


from scipy.stats import norm

norm.cdf(z_score)
# 0.094941687240975514 # Tells us how significant our z-score is


# In[382]:


norm.ppf(1-(0.05/2))
# 1.959963984540054 # Tells us what our critical value at 95% confidence is


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# Since the z-score of 1.3109241984234394 does not exceed the critical value of 1.959963984540054, we fail to reject the null hypothesis that old page users has a better or equal converted rate than old page users. 
# Therefore, the converted rate for new page and old page have no difference. This result is the same as parts J. and K. result. 

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.<br><br>
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# Logistic Regression, due to the fact that response variable is categorical variable. 
# logistic regression is multiple regression but with an outcome variable that is a categorical variable and predictor variables that are continuous 

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives.  However, you first need to create a colun for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[67]:


#create a colun for the intercept
df2['intercept'] = 1


# In[68]:


#create a dummy variable column for which page each user received
df2= df2.join(pd.get_dummies(df2['landing_page']))


# In[69]:


df2['ab_page'] = pd.get_dummies(df['group']) ['treatment']
df2.head()


# c. Use **statsmodels** to import your regression model.  Instantiate the model, and fit the model using the two columns you created in part **b.** to predict whether or not an individual converts.

# In[70]:


#Create Logit regression model for conveted variable and  ab_page, and us control as baseline
lo = sm.Logit(df2['converted'], df2[['intercept','ab_page']])


# In[71]:


result = lo.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[72]:


print result.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in the **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in the **Part II**?

# The p-value associated with ab_page is 0.190.
# The null in c-e part is  that there is no difference between the treatment and control group.
# Alternative hypotheses is that there is difference between between the treatment and control group
# 
# Part II assumes the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, compared to question c-e,they have different explainory varibale or factor for the result. 

# f. Now, I am considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into   regression model.  Are there any disadvantages to adding additional terms into   regression model?

# Other factor can be the time(timestamp	variable). We can check if the converted rate depends on certain time of the day or certain day when user browerse the website.
# For timestamp	variable, we can further convert time as categorical variable which includes "Morning, afternoon, and evening", or "weekday and weekend".
# Disadavantage for adding additional terms into regression model is that it will make interpretate the model more complex and also, if new terms are dependable variable with the exisiting explanatory term, we need to add higher order term to help predict the result better.

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. You will need to read in the **countries.csv** dataset and merge together your datasets on the approporiate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy varaibles.** Provide the statistical output as well as a written response to answer this question.

# In[73]:


c = pd.read_csv('countries.csv')
c.head()


# In[74]:


#Join ab dataset with country dataset
df3 = df2.merge(c, on ='user_id', how='left')
df3.head()


# In[75]:


c['country'].unique()


# In[76]:


df3[['CA','UK','US']] = pd.get_dummies(df3['country'])
df3 = df3.drop(df3['CA'])


# In[85]:


#Create intercept variable
df3['intercept'] = 1

#Create Logit regression model for conveted and country, and us CA and old page as baseline
logit3 = sm.Logit(df3['converted'], df3[['intercept','new_page','UK','US']])
result = logit3.fit()
result.summary()


# In[283]:


1/np.exp(-0.0150),np.exp(0.0506),np.exp(0.0408)


# Interpreting Result:
# 
# For every unit for new_page decrease, convert will be 1.5% more likely to happen, holding all other varible constant.
# 
# For every unit for UK increases, convert is 5.2% more to happen, holding all other varible constant.
# 
# For every unit for US increases, convert is 4.2% more to happen, holding all other varible constant.

# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.
# What the p-value for each country? what is the p-critical, does the coefficient calculated for each country is significant?

# In[98]:


#Create a new intereacton variable between new page and country US and UK
df3['UK_new_page'] = df3['new_page']* df3['UK']
df3['US_new_page'] = df3['new_page']* df3['US']


# In[99]:


#Create logistic regression for the intereaction variable between new page and country using dummy variable
logit4 = sm.Logit(df3['converted'], df3[['intercept','new_page','UK_new_page','US_new_page','UK','US']])
result4 = logit4.fit()
result4.summary()


# In[300]:


#exponentiated the CV to inteprete the result
np.exp(result4.params)


# Interpreting Result:
# 
# From the above Logit Regression Results, we can see the coefficient of intereaction variable "UK_new_page" and "US_new_page" are different from the coefficient of new_page itself. 
# 
# Also,only intercept's p-value is less than 0.05, which is statistically significant enough for converted rate. Other varable in the summary are not statistically significant. 
# Additionally, Z-score for all X variables are not large enough to be significant for predicting converted rate. 
# 
# Therefore, the country a user lives is not significant on the converted rate considering the page the user land in. 
# 
# For every unit for new_page decreases, convert will be 7.0% more likely to happen, holding all other varible constant.
# 
# Convert is 1.08 times more likely to happen for UK and new page users than CA and new page users, holding all other varible constant.
# 
# Convert is 1.04 times more likely to happen for US and new page users than CA and new page users, holding all other varible constant.
# 
# Convert is 1.18 % more likely to happen for the users in UK than CA, holding all other varible constant.
# 
# Convert is 1.76 % more likely to happen for the users in US than CA, holding all other varible constant.
# 

# In[324]:


#Import sklearn model to split, test and score data,and fit data model 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split


# In[346]:


#Define X and Y variable 
x = df3[['new_page','UK_new_page','US_new_page','UK','US']]
y = df3['converted']
        
#Split data into train and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)


# In[347]:


lm = LinearRegression()


# In[348]:


lm.fit(X_train,y_train) # fit the train data


# In[351]:


print(lm.score(X_test,y_test))


# The score reuslt is very low, which mean the page and country dataset are not a good fit to predit converted rate .
