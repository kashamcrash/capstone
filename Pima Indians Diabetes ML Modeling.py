#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Agenda
# Predict the possibility of diabetes based on multiple diagnostic measurements 
# Dataset has been sourced from NIDDK (National Institute of Diabetes and Digestive and Kidney Diseases)

# Credentials - kasham1991@gmail.com | Karan Sharma


# In[2]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Goal 1.0: Descriptive analysis
# 1. Understand the different variables and values
# 2. Identify missing values to suggest treatment
# 3. Counting the various dtypes 


# In[4]:


# Loading the dataset
Diabetes = pd.read_csv('C:\\Datasets\\Diabetes.csv')


# In[5]:


# Lets look at the top five values
# 8 feature variables and 1 predicted value
Diabetes.head()


# In[6]:


# Lets see the data frame in detail
# There are two dtypes
Diabetes.info()


# In[7]:


# Now that we know the dtypes, lets get into statistics
# The describe function deals only with numerical values
Diabetes.describe()


# In[8]:


# Lets use the transpose function to get a better view
# Moving the rows data to the column and columns data to the rows
# There are multiple 0 values; missing values
# Glucose, blood pressure, skin thickness, insulin and BMI have missing values
# Pregancy can be ignored as 0 indicates no children
Diabetes.describe().T


# In[9]:


# Replacing 0s with NaN, it will be easier to count them 
# Plus, 0s have to be replaced with a suitable value anyway, let it be NaN for now
Diabetes[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = Diabetes[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0, np.NaN)


# In[10]:


# Checking total NaN values
Diabetes.isnull()
Diabetes.isnull().sum()


# In[11]:


# Visualizing the null values
# ggplot2 is a data visualization package from the statistical programming language R
import matplotlib.pyplot as plt
from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[12]:


# Understanding the distribution of null values with individual histograms
# You can choose the figure size as per your screen
Diabetes.hist(figsize = (13,13))


# In[13]:


# Treating the missing values accordingly with mean and median
# Mean is suitable for data that is centrally placed
# Median is suitable for data that has outliers and is subject to skewness
Diabetes['Glucose'].fillna(Diabetes['Glucose'].mean(), inplace = True)
Diabetes['BloodPressure'].fillna(Diabetes['BloodPressure'].mean(), inplace = True)
Diabetes['SkinThickness'].fillna(Diabetes['SkinThickness'].median(), inplace = True)
Diabetes['Insulin'].fillna(Diabetes['Insulin'].median(), inplace = True)
Diabetes['BMI'].fillna(Diabetes['BMI'].median(), inplace = True)


# In[14]:


# Plotting after NaN value treatment
# We can clearly see the change in the histograms
Diabetes.hist(figsize = (13,13))


# In[15]:


# Goal 2.0: Data exploration
# 1. Checking the balance of the data
# 2. Creating scatter charts between variables
# 3. Correlation analysis


# In[16]:


# Seaborn has a count plot method that shows the counts of observations 
import seaborn as sns
sns.set()


# In[17]:


# Looking at the shape of the data
# 768 rows and 9 colums; dimensions
# Out of the 768 persons, 500 are labeled as 0 (non-diabetic) and 268 as 1 (diabetic)
Diabetes.shape
Diabetes.groupby('Outcome').size()


# In[18]:


# Creating count plot with title using seaborn
# Number of non-diabetics is twice the number of diabetic patients
sns.countplot(x = Diabetes.Outcome)
plt.title("Count Plot for Outcome")


# In[19]:


# Creating a pairplot on the basis of outcome
# Pairplot comprises of a histogram and a scatter plot
# It will showcase the distribution of a single variable/relationship b/w two variables
# this will show how much one variable is affected by another
sns.pairplot(Diabetes, hue = 'Outcome')
plt.title("Pairplot of Variables by Outcome")


# In[20]:


# Creating a correlation matrix
a = Diabetes.corr(method = 'pearson')
a


# In[21]:


# Creating a heatmap for the same
# Age & pregnancies, BMI & skin thickness, insulin an glucose have moderate positive linear relationship
sns.heatmap(a, annot= True ,cmap ='RdYlGn')
plt.figure(figsize=(12, 12))


# In[22]:


# Goal 3.0: ML modeling
# 1. Standardization and Model Building
# 2. Model Building using KNeighboursClassifier with cross validation


# In[23]:


# Since we are using KNN, it is better to standardize the data
# Standardization involves shifting the distribution of each data point to a mean of 0 and an SD of 1
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x =  pd.DataFrame(sc_x.fit_transform(Diabetes.drop(["Outcome"], axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])
x.head()


# In[24]:


# Creating the predictor variable
y = Diabetes.Outcome
y


# In[25]:


# Splitting the data into train/test with stratify
# Our data is highly imbalanced, stratify = y will handle the samples accordingly
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 1, stratify = y)


# In[26]:


# Fitting the KNeighborsClassifier algorithm with K Flold Cross Validation
# A CV of 5 or 10 is generally preferred 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors = 10)
knn_model = knn.fit(x_train, y_train)
Accuracy_Scores_10_fold = cross_val_score(knn_model, x, y, cv = 10, scoring = 'accuracy')
print('Accuracy score over 10 validation datasets are: ', Accuracy_Scores_10_fold)
print('Average Accuracy score over 10 validation datasets is: ', round(Accuracy_Scores_10_fold.mean(), 2))


# In[27]:


# Goal 4.0: Classification report 
# Confusion Matrix
# AUC-ROC Curve


# In[28]:


from sklearn.metrics import confusion_matrix
y_pred = knn.predict(x_test)
confusion_matrix(y_test, y_pred)
pd.crosstab(y_test, y_pred, colnames = ['Predicted'], margins = True)


# In[29]:


# Classification report
# 0.74 precision is considered good
# recall > 0.5 is considered good
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[30]:


# Creating and Plotting ROC-AUC
# Receiver Operating Characteristic Curve
from sklearn.metrics import roc_curve
y_pred_roc = knn.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_roc)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label = 'KNN')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN (n_neighbors = 10) ROC Curve')


# In[31]:


# AUC - Area under the curve
# This is in the form of a score
# Higher the AUC, better the model
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_roc)


# In[32]:


# ML Modeling with Logistic Regression
from sklearn.linear_model import LogisticRegression
s = LogisticRegression()
s.fit(x_train, y_train)


# In[33]:


print(s.score(x_train, y_train))
print(s.score(x_test, y_test))


# In[34]:


# Using ANN Modeling
import tensorflow as tf


# In[35]:


tf.__version__


# In[36]:


# Architecting ANN

model3 = tf.keras.Sequential([
    tf.keras.layers.Dense(units = 10, activation = 'relu', input_dim = 8),
    tf.keras.layers.Dense(units = 8, activation = 'relu' ),
    tf.keras.layers.Dense(units = 6, activation = 'relu' ),
    tf.keras.layers.Dense(units = 1, activation = 'sigmoid')
    
])


# In[37]:


# Compile ANN

model3.compile(loss= tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics = ['accuracy']
             )


# In[40]:


# Fitting the model

model3.fit(x_train,
         y_train,
         epochs = 10,
         validation_data= (x_test, y_test))


# In[39]:


# Thank you :) 

