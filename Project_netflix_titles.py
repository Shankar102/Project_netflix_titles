#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
print('Done')


# https://www.kaggle.com/code/ezzaldin6/eda-of-netflix-contents/notebook

# # Data Investigation

# In[2]:


df=pd.read_csv('data/netflix_titles_nov_2019.csv')
df.head(10)


# In[3]:


df.columns


# In[4]:


df.isnull().sum()


# # Data Cleaning
# 

# Drop id column
# Drop dublicate shows
# create a new column shows the number of cast in each row
# we have 10 missing rows in rating column, replace them by the mode
# for the missing rows in added_date column, replace them by January 1,{release_year}
# I think we can not replace missing rows in column country by other countries, but we can use genre to
# identify this country ex: replace missing rows by japan for Anime
# convert the date_added column from object type to datetime

# In[5]:


dup=df.duplicated(['director','cast','country','date_added'])
df[dup]


# In[6]:


df=df.drop_duplicates(['director','cast','country','date_added'])
df


# In[7]:


df=df.drop('director',axis=1)
df


# In[8]:


df['cast']=df['cast'].replace(np.nan,'Unknown')
def cast_counter(cast):
    if cast=='Unknown':
        return 0
    else:
        lst=cast.split(', ')
        length=len(lst)
        return length
df['number_of_cast']=df['cast'].apply(cast_counter)
df['cast']=df['cast'].replace('Unknown',np.nan)
df['cast']


# In[9]:


df=df.reset_index()
df


# In[10]:


df['rating']=df['rating'].fillna(df['rating'].mode()[0])
df['rating']


# In[11]:


df['date_added']=df['date_added'].fillna('January 1, {}'.format(str(df['release_year'].mode()[0])))
df['date_added']


# In[12]:


df.isnull().sum()


# In[13]:


for i,j in zip(df['country'].values,df.index):
    if i==np.nan:
        if ('Anime' in df.loc[j,'listed_in']) or ('anime' in df.loc[j,'listed_in']):
                df.loc[j,'country']='Japan'
        else:
            continue
    else:
        continue
df


# In[14]:


import re
months={
    'January':1,
    'February':2,
    'March':3,
    'April':4,
    'May':5,
    'June':6,
    'July':7,
    'August':8,
    'September':9,
    'October':10,
    'November':11,
    'December':12
}

    


# In[15]:


date_lst=[]

for i in df['date_added'].values:
        
        str1=re.findall('([a-zA-Z]+)\s[0-9]+\,\s[0-9]+',i)
        str2=re.findall('[a-zA-Z]+\s([0-9]+)\,\s[0-9]+',i)
        str3=re.findall('[a-zA-Z]+\s[0-9]+\,\s([0-9]+)',i)
        date='{}-{}-{}'.format(str3[0],months[str1[0]],str2[0])
        date_lst.append(date)


# In[16]:


df['date_added_cleaned']=date_lst


# In[17]:


df=df.drop('date_added',axis=1)
df['date_added_cleaned']=df['date_added_cleaned'].astype('datetime64[ns]')


# # Exploratory Data Analysis

# Exploratory Data Analysis
# now, it is time to answer some questions.
# 
# Understand every category in rating column(Google it
# Understanding what content is available in different countries.
# Is Netflix has increasingly focusing on TV rather than movies in recent years.
# The most observed rating categories in TV-shows and Movies
# Identifying similar content by matching text-based features
# How many content its release year differ from its year added
# let's now google the categories and explore them
# 
# TV-MA:This program is specifically designed to be viewed by adults and therefore may be unsuitable for children under 17.
# TV-14:This program contains some material that many parents would find unsuitable for children under 14 years of age.
# TV-PG:This program contains material that parents may find unsuitable for younger children.
# R:Under 17 requires accompanying parent or adult guardian,Parents are urged to learn more about the film before taking their young children with them.
# PG-13:Some material may be inappropriate for children under 13. Parents are urged to be cautious. Some material may be inappropriate for pre-teenagers.
# NR or UR:If a film has not been submitted for a rating or is an uncut version of a film that was submitted
# PG:Some material may not be suitable for children,May contain some material parents might not like for their young children.
# TV-Y7:This program is designed for children age 7 and above.
# TV-G:This program is suitable for all ages.
# TV-Y:Programs rated TV-Y are designed to be appropriate for children of all ages. The thematic elements portrayed in programs with this rating are specifically designed for a very young audience, including children ages 2-6.
# TV-Y7-FV:is recommended for ages 7 and older, with the unique advisory that the program contains fantasy violence.
# G:All ages admitted. Nothing that would offend parents for viewing by children.
# NC-17:No One 17 and Under Admitted. Clearly adult. Children are not admitted.
# here we discover that UR and NR is the same rating(unrated,Not rated)
# Uncut/extended versions of films that are labeled "Unrated" also contain warnings saying that the uncut version of the film contains content that differs from the theatrical release and might not be suitable for minors.
# so we have the fix this.

# In[ ]:





# In[19]:


df['rating']


# In[20]:


plt.figure(figsize=(8,5))
df['rating'].value_counts(normalize=True).plot.bar()
plt.xlabel('Rating')
plt.ylabel('value_counts')
plt.title('Rating Categories')


# In[21]:


plt.figure(figsize=(10,5))
sns.countplot(x='rating',hue='type',data=df)
plt.title('Differnce between rating & type')
plt.xlabel(' rating & type')


# In[22]:


df['country'].value_counts()


# In[52]:


top_country=df[(df['country']=='United States')|(df['country']=='India')|(df['country']=='United Kingdom')|
               (df['country']=='Japan')|(df['country']=='Canada')]
plt.figure(figsize=(8,4))
sns.countplot(x='country',hue='type',data=top_country)
plt.xlabel('Country')


# In[58]:


for i in top_country['country'].unique():
    print(i)
    print(top_country[top_country['country']==i]['rating'].value_counts(normalize=True) * 100)


# Data Cleaning
# Drop id column
# Drop dublicate shows
# create a new column shows the number of cast in each row
# we have 10 missing rows in rating column, replace them by the mode
# for the missing rows in added_date column, replace them by January 1,{release_year}
# I think we can not replace missing rows in column country by other countries, but we can use genre to
# identify this country ex: replace missing rows by japan for Anime
# convert the date_added column from object type to datetime

# In[61]:


df['date_added']=df['date_added_cleaned'].dt.year
df['date_added']


# In[62]:


df['type'].value_counts(normalize=True)


# In[72]:


df.groupby('date_added')['type'].value_counts(normalize=True)*100


# In[81]:


dup=df.duplicated(['title'])
df[dup]['title']


# In[90]:


for i in df[dup]['title'].values:
    print(df[df['title']==i][['country','type','title','date_added']])
    print('_'*50)


# In[97]:


plt.figure(figsize=(8,5))
df['date_added'].value_counts().plot.bar()
plt.xlabel('data_added')
plt.ylabel('value_counts')


# In[107]:


counts=0
for i,j in zip(df['release_year'].values,df['date_added'].values):
    if i!=j:
        counts+=1
print(str(counts))    


# In[ ]:





# In[ ]:




