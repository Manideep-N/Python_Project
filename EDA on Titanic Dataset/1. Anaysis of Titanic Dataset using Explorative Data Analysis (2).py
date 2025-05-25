#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read the csv File
data = pd.read_csv("titanic.csv")
print(data.shape)

data.head(5)


# In[4]:


data.shape


# # data cleaning

# In[5]:


data.isnull().sum()


# In[6]:


# Checking the Missing Values percentage

print(data.isnull().sum())


# In[7]:


#checking wheather to remove the whole columne or fill the null places

(327/418)*100

#78% missing values are present so dropping the whole cabin columne


# In[8]:


data = data.drop(["Cabin"],axis = 1)
data.head()
#cabin columne is dropped


# In[9]:


#checking age columne

(86/418)*100  

#it's less that 30% so impute with either mean on median values based on outliers


# In[10]:


#checking for outliers
sns.boxplot(x=data["Age"])


# In[11]:


#there are less ouliers, so checking for mean and median values
mean=data["Age"].mean()
print(mean)

median=data["Age"].median()
print(median)


# In[12]:


#there is small difference between mean and median , so we can impute mean values for null values
data["Age"]=data["Age"].fillna(data["Age"].mean())


# In[5]:


#checking Fare columne

(1/418)*100  

#it's less that 30% so impute with either mean on median values based on outliers


# In[14]:


#checking for outliers
sns.boxplot(x=data["Fare"])


# In[15]:


#there are more ouliers,substitue median values or
#check mean and median values, if there is huge difference between then take median
#else impute mean

median=data["Fare"].median()
print(median)

mean=data["Fare"].mean()
print(median)


# In[16]:


data["Fare"]=data["Fare"].fillna(data["Fare"].median())


# In[17]:


#checking for null values
print(data.isnull().sum())


# In[18]:


#drop the unwanted columns as Cabin has most null values and other columns which are not utlized for investigation
data=data.drop(['Ticket','Name','Fare','PassengerId'],axis=1)


# In[19]:


data


# # UNIVARIENT ANALYSIS

# In[20]:


plt.figure(figsize=(50,7))
sns.countplot(data=data,x='Age')


# In[21]:


sns.countplot(data=data,x='Sex')

#male traveller are more


# In[22]:


sns.countplot(data=data,x='Pclass')


# In[23]:


#checking people onboard port
sns.countplot(data=data,x="Embarked",color='aqua')


# In[24]:


#analysing people embarkment and survived 0->no 1->yes
sns.countplot(data=data,x='Embarked',hue='Survived')


# In[25]:


#analysing relation between embarkment and pclass
sns.countplot(data=data,x='Embarked',hue='Pclass')


# In[26]:


#analysing relation between survived and pclass
sns.countplot(data=data,x='Pclass',hue='Survived')


# In[27]:


#analysis of sex and embarked
sns.countplot(data=data,x='Embarked',hue='Sex')


# # Bivariate Analysis

# In[28]:


#analysis of age and embarked
sns.barplot(data=data,x='Embarked',y='Age')


# In[29]:


#analysis of sex,embarked,age
sns.barplot(data=data,x='Embarked',y='Age',hue='Sex')


# In[30]:


#analysis of embarked,pclass and sex
sns.barplot(data=data,x='Embarked',y='Pclass',hue='Sex')


# In[31]:


#analysis of age,embarked and survived
sns.barplot(data=data,x='Embarked',y='Age',hue='Survived')


# In[35]:


#analysis of embarked,pclass and survived
sns.barplot(data=data,x='Embarked',y='Pclass',hue='Survived')


# From the above graph we can conclude that people embarked from "Q" has more number of survivers

# # Multivariate analysis

# In[38]:


sns.heatmap(data.corr(),annot=True,fmt='0.01f')


# In[ ]:




