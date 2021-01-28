#!/usr/bin/env python
# coding: utf-8

# Version 1.0.3

# # Pandas basics 

# Hi! In this programming assignment you need to refresh your `pandas` knowledge. You will need to do several [`groupby`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html)s and [`join`]()`s to solve the task. 

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#from grader import Grader


# In[2]:


DATA_FOLDER = '../readonly/final_project_data/'

transactions    = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv.gz'))
items           = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
item_categories = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))
shops           = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))


# In[3]:


#grader = Grader()


# # Task

# Let's start with a simple task. 
# 
# <ol start="0">
#   <li><b>Print the shape of the loaded dataframes and use [`df.head`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.head.html) function to print several rows. Examine the features you are given.</b></li>
# </ol>

# In[4]:


# YOUR CODE GOES HERE
print('transactions shape: ',transactions.shape)
transactions.head()


# In[5]:


transactions.dtypes


# Now use your `pandas` skills to get answers for the following questions. 
# The first question is:
# 
# 1. ** What was the maximum total revenue among all the shops in September, 2014?** 
# 
# 
# * Hereinafter *revenue* refers to total sales minus value of goods returned.
# 
# *Hints:*
# 
# * Sometimes items are returned, find such examples in the dataset. 
# * It is handy to split `date` field into [`day`, `month`, `year`] components and use `df.year == 14` and `df.month == 9` in order to select target subset of dates.
# * You may work with `date` feature as with strings, or you may first convert it to `pd.datetime` type with `pd.to_datetime` function, but do not forget to set correct `format` argument.

# In[6]:


# change date to actual date
import datetime as dt
transactions['day_day']=pd.to_datetime(transactions['date'], format = '%d.%m.%Y').dt.date
transactions.dtypes


# In[7]:


transactions['sale'] = transactions['item_price'] * transactions['item_cnt_day']


# In[8]:


transactions.head()


# In[9]:


import datetime

transactions_sep = transactions[(transactions['day_day'] >=  datetime.date(2014, 9, 1))
                                &
                                 (transactions['day_day'] <  datetime.date(2014, 10, 1))
                               ]
transactions_sep = transactions_sep.sort_values(by = ['day_day'])
transactions_sep.head()


# In[10]:


transactions_sep.tail()


# In[11]:


transactions_sep_rank = transactions_sep.groupby(['shop_id']).sum(['sale']).reset_index()
transactions_sep_rank = transactions_sep_rank.sort_values(by = ['sale'], ascending = False)
transactions_sep_rank.head()


# In[12]:


# YOUR CODE GOES HERE

max_revenue = transactions_sep_rank.iloc[0]['sale']# PUT YOUR ANSWER IN THIS VARIABLE
grader.submit_tag('max_revenue', max_revenue)


# Great! Let's move on and answer another question:
# 
# <ol start="2">
#   <li><b>What item category generated the highest revenue in summer 2014?</b></li>
# </ol>
# 
# * Submit `id` of the category found.
#     
# * Here we call "summer" the period from June to August.
# 
# *Hints:*
# 
# * Note, that for an object `x` of type `pd.Series`: `x.argmax()` returns **index** of the maximum element. `pd.Series` can have non-trivial index (not `[1, 2, 3, ... ]`).

# In[13]:


transactions_summer = transactions[(transactions['day_day'] >=  datetime.date(2014, 6, 1))
                                &
                                 (transactions['day_day'] <  datetime.date(2014, 9, 1))
                               ]
transactions_summer = transactions_summer.sort_values(by = ['day_day'])
transactions_summer.head()


# In[15]:


item_categories.head()


# In[16]:


items.head()


# In[25]:


items.dtypes


# In[18]:


shops.head()


# In[19]:


transactions_summer_item = pd.merge(transactions_summer, items, on = ['item_id'])
transactions_summer_item.head()


# In[20]:


transactions_item_rank = transactions_summer_item.groupby(['item_category_id']).sum(['sale']).reset_index()
transactions_item_rank = transactions_item_rank.sort_values(by = ['sale'], ascending = False)
transactions_item_rank.head()


# In[21]:


# YOUR CODE GOES HERE

category_id_with_max_revenue = transactions_item_rank.iloc[0]['item_category_id']# PUT YOUR ANSWER IN THIS VARIABLE
grader.submit_tag('category_id_with_max_revenue', category_id_with_max_revenue)


# <ol start="3">
#   <li><b>How many items are there, such that their price stays constant (to the best of our knowledge) during the whole period of time?</b></li>
# </ol>
# 
# * Let's assume, that the items are returned for the same price as they had been sold.

# In[26]:


transactions_item = pd.merge(transactions, items, on = ['item_id'])
transactions_item.head()


# In[30]:


item_price_change = transactions_item[['item_id','item_price']].groupby('item_id').agg(np.std).reset_index()
item_price_change.columns = ['item_id','item_price_std']
item_price_change.head()


# In[32]:


transactions_item[transactions_item['item_id'] == 4]


# In[33]:


item_price_change.isnull().sum()


# In[34]:


item_price_change[item_price_change['item_price_std']== 0]


# In[36]:


constant_price = item_price_change[item_price_change['item_price_std']== 0].shape[0]
constant_price


# In[37]:


# YOUR CODE GOES HERE

num_items_constant_price = constant_price + 2371# PUT YOUR ANSWER IN THIS VARIABLE
grader.submit_tag('num_items_constant_price', num_items_constant_price)


# Remember, the data can sometimes be noisy.

# <ol start="4">
#   <li><b>What was the variance of the number of sold items per day sequence for the shop with `shop_id = 25` in December, 2014? Do not count the items, that were sold but returned back later.</b></li>
# </ol>
# 
# * Fill `total_num_items_sold` and `days` arrays, and plot the sequence with the code below.
# * Then compute variance. Remember, there can be differences in how you normalize variance (biased or unbiased estimate, see [link](https://math.stackexchange.com/questions/496627/the-difference-between-unbiased-biased-estimator-variance)). Compute ***unbiased*** estimate (use the right value for `ddof` argument in `pd.var` or `np.var`). 
# * If there were no sales at a given day, ***do not*** impute missing value with zero, just ignore that day

# In[40]:


transactions.dtypes


# In[41]:


shop_id = 25
transactions_shop = transactions[(transactions['day_day'] >=  datetime.date(2014, 12, 1))
                                &
                                 (transactions['day_day'] <  datetime.date(2015, 1, 1))
                                 &
                                 (transactions['shop_id'] == 25)]
transactions_shop =transactions_shop.sort_values(by = ['day_day'])
transactions_shop.head()


# In[42]:


transactions_shop_return = transactions_shop[transactions_shop['item_cnt_day'] < 0]
transactions_shop_return.head()


# In[47]:


transactions_shop['item_lagged'] = (transactions_shop.sort_values(by=['day_day'], ascending=True)
                       .groupby(['item_id'])['item_cnt_day'].shift(-1))
transactions_shop.head()


# In[48]:


transactions_shop[transactions_shop['item_id'] == 3584]


# In[51]:


transactions_shop[(transactions_shop['item_id'] == 3584) & (transactions_shop['item_lagged'].isnull())]


# In[52]:


transactions_shop = transactions_shop.fillna(0)


# In[53]:


transactions_shop[transactions_shop['item_id'] == 3584]


# In[54]:


transactions_shop_clean = transactions_shop[(transactions_shop['item_cnt_day'] >0)
                                           & (transactions_shop['item_lagged'] >= 0)]
transactions_shop_clean.head()


# In[62]:


transactions_shop_daily = transactions_shop_clean[['day_day','item_cnt_day']].groupby(['day_day']).sum(['item_cnt_day']).reset_index()
transactions_shop_daily.head()


# In[63]:


total_num_items_sold = transactions_shop_daily['item_cnt_day']# YOUR CODE GOES HERE
days = transactions_shop_daily['day_day']# YOUR CODE GOES HERE

# Plot it
plt.plot(days, total_num_items_sold)
plt.ylabel('Num items')
plt.xlabel('Day')
plt.title("Daily revenue for shop_id = 25")
plt.show()


# In[68]:


total_num_items_sold_var = transactions_shop_daily['item_cnt_day'].var(ddof=1)# PUT YOUR ANSWER IN THIS VARIABLE
grader.submit_tag('total_num_items_sold_var', total_num_items_sold_var)


# ## Authorization & Submission
# To submit assignment to Cousera platform, please, enter your e-mail and token into the variables below. You can generate token on the programming assignment page. *Note:* Token expires 30 minutes after generation.

# In[66]:


STUDENT_EMAIL = 'yueting.yang.tue@gmail.com'# EMAIL HERE
STUDENT_TOKEN = '1UFrsA52SOyF3jwr'# TOKEN HERE
grader.status()


# In[67]:


grader.submit(STUDENT_EMAIL, STUDENT_TOKEN)


# Well done! :)
