#!/usr/bin/env python
# coding: utf-8

# In[1]:


def missing(df) : 
    missing_number = df.isnull().sum().sort_values(ascending = False)
    missing_percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending = False)
    missing_values = pd.concat([missing_number, missing_percent], axis = 1, keys = ['Missing_number', 'Missing_percent'])
    return missing_values 

def categorize(df) :
    Quantitive_features = df.select_dtypes([np.number]).columns.tolist()
    Categorical_features = df.select_dtypes(exclude = [np.number]).columns.tolist()
    Discrete_features = [col for col in Quantitive_features if len(df[col].unique()) < 10]
    Continuous_features = [col for col in Quantitive_features if col not in Discrete_features]
    print("Quantitive feautres : {} \nDiscrete features : {} \nContinous features : {} \nCategorical features : {}\n"
     .format(Quantitive_features, Discrete_features, Continuous_features, Categorical_features))
    print("Number of quantitive feautres : {} \nNumber of discrete features : {} \nNumber of continous features : {} \nNumber of categorical features : {}"
     .format(len(Quantitive_features), len(Discrete_features), len(Continuous_features), len(Categorical_features)))
    return Quantitive_features, Categorical_features, Discrete_features, Continuous_features
    
def unique(df) : 
    tb1 = pd.DataFrame({'Columns' : df.columns, 'Number_of_Unique' : df.nunique().values.tolist(),
                       'Sample1' : df.sample(1).values.tolist()[0], 'Sample2' : df.sample(1).values.tolist()[0], 
                       'Sample3' : df.sample(1).values.tolist()[0],
                       'Sample4' : df.sample(1).values.tolist()[0], 'Sample5' : df.sample(1).values.tolist()[0]})
    return tb1
    
def data_glimpse(df) :  
      # Dataset preview 
    print("1. Dataset Preview \n")
    display(df.head())
    print("-------------------------------------------------------------------------------\n")
    
    # Columns imformation
    print("2. Column Imformation \n")
    print("Dataset have {} rows and {} columns".format(df.shape[0], df.shape[1]))
    print("\n") 
    print("Dataset Column name : {}".format(df.columns.values))
    print("\n")
    categorize(df)
    print("-------------------------------------------------------------------------------\n")
    
    # Basic imformation table 
    print("3. Missing data table : \n")
    display(missing(df))
    print("-------------------------------------------------------------------------------\n")
    
    print("4. Number of unique value by column : \n")
    display(unique(df))
    print("-------------------------------------------------------------------------------\n")
    
    print("5. Describe table : \n")
    display(df.describe())
    print("-------------------------------------------------------------------------------\n")
    
    print(df.info())
    print("-------------------------------------------------------------------------------\n")


# In[2]:


# Data Analysis
import warnings 
warnings.filterwarnings('ignore')
    
import pandas as pd
import numpy as np
import os 
import missingno as msno
    
# Data View
pd.options.display.max_columns = 200

# Import Basic Visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
    


# In[3]:


df_raw= pd.read_csv("C:\\Users\\jason\\OneDrive\\Desktop\\NCU\\8550_Data Preparation Methods\\Week 4\\marketing_campaign.csv")


# In[4]:


data_glimpse(df_raw)


# In[5]:


#Deal ing with NA's

df_raw[df_raw['Income'].isnull() == True]


# In[6]:


#check if 0 income exist in the dataset.  If not then NA's may be 0
is_0 = len(df_raw[df_raw['Income']==0])
print("Number of 0 income in data: {}".format(is_0))


# In[7]:


df_raw['Income'].fillna(0, inplace = True)
df_raw.isnull().sum()


# In[8]:


#change data type for Income and Dt_customer
#Income as 'int64'
#Dt_customer: as 'datetime'

df_raw.Income = df_raw.Income.astype('int64')
df_raw.Dt_Customer = pd.to_datetime(df_raw.Dt_Customer)

df_raw.dtypes


# In[9]:


# Dealing with Datetime Value for Year_Birth

df=df_raw.copy()

df['Year_Old'] = df.apply(lambda x : 2014 - x.Year_Birth, axis = 1) #Data collected in 2014
df.drop(columns = 'Year_Birth', inplace = True) # Drop column 'Year_Birth'
df.head()


# In[10]:


# Dealing with Datetime Value for Dt_Customer

df['Year_Customer'] = 2014 - df.Dt_Customer.dt.year # Because data has been written in 2014. 
df.drop(columns = 'Dt_Customer', inplace = True)
df.head()


# In[11]:


#drop some variables
df = df.drop(columns = ['ID', 'Z_CostContact', 'Z_Revenue']) # We don't need column ID and feature 'Z_CostContac', 'Z_Revenue' becuase they have only one variable. 
df.head()


# In[12]:


new_col = [fea for fea in df.columns if fea != 'Response'] # reorder columns to make correlation more prettier
new_col.append('Response')

df = df[new_col]
df.head()


# In[13]:


#Check Data Type
df_raw.dtypes


# In[14]:


df_raw.head() #preview first five rows


# In[15]:


df_raw.describe() #get summary statistics about the dataset


# In[16]:


df_raw.info() #get additional detail from data i.e., count of non-null

