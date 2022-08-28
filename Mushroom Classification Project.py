#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ### Reading the csv file of the dataset

# In[2]:


#import the dataset
df = pd.read_csv('mushrooms.csv')
df.head()


# # EDA

# In[3]:


df.info()


# In[4]:


df.describe()


# ####  The column "veil-type" has only 1 unique value - that is "p", all 8124 mushroom instances have the same veil-type and it is not contributing to the data so we remove it.

# In[5]:


df.drop(["veil-type"],axis=1, inplace=True)
df.head()


# In[ ]:





# ### Finding the shape of the dataset

# In[6]:


print('Shape of the Dataset:', df.shape)


# ### Visualizing the count of edible and poisonous mushrooms

# In[7]:


df['class'].value_counts()


# In[8]:


# Checking the unique values
df["class"].unique()


# In[9]:


count = df['class'].value_counts()
plt.figure(figsize=(8,7))
sns.barplot(count.index, count.values, alpha=0.8, palette="prism")
plt.ylabel('Count', fontsize=15)
plt.xlabel('Class', fontsize=15)
plt.title('Number of poisonous/edible mushrooms')
plt.show()


# #### As we can see. The dataset is balanced.

# ### Converting categorical data to numerical

# #### The data is categorical so we’ll use LabelEncoder and One Hot Encoding to convert it to numerical. 

# In[ ]:


# Label Encoding


# In[17]:


from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])
df.head()


# #### We can see how it has converted some of the features to values of 0 or 1. More importantly, our labels (the _class_ column) are now 0=e, and 1=p.

# In[ ]:


# One-Hot Encoding


# In[ ]:


df1 = pd.get_dummies(df)
df1.head()


# In[ ]:





# ### Seperating labels from features.

# #### X will now contain our features, and y our labels (0 for edible and 1 for poisonous/unknown) 

# In[18]:


X = df.iloc[:,1:22]
Y = df.iloc[:, 0]


# In[19]:


X.head()


# In[20]:


Y.head()


# In[21]:


X.describe()


# In[22]:


df.corr()


# In[24]:


plt.figure(figsize=(14,12))
sns.heatmap(df.corr(), linewidths=.1, cmap="Purples", annot=True, annot_kws={"size": 7})
plt.show()


# ### Standardizing the values 

# In[25]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(X)
X


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.4,random_state=10)


# # Classification Models

# ## 1. Logistic Regression

# In[27]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[28]:


# Model Building
mushroom_lr=LogisticRegression()
mushroom_lr.fit(X_train,Y_train)


# In[29]:


y_prob = mushroom_lr.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
mushroom_lr.score(X_test, y_pred)


# ### Model Accuracy

# In[30]:


# Confusion Matrix for the model accuracy
confusion_matrix = confusion_matrix(Y_test,y_pred)
confusion_matrix


# In[31]:


# classification Report
print(classification_report(Y_test,y_pred))


# In[32]:


# The model accuracy is calculated by (a+d)/(a+b+c+d)
(1631+1468)/(1631+68+83+1468)


# In[33]:


# ROC Curve plotting and finding AUC value
fpr,tpr,thresholds=roc_curve(Y,mushroom_lr.predict_proba(X)[:,1])
plt.plot(fpr,tpr,color='red')
auc=roc_auc_score(Y_test,y_pred)

plt.plot(fpr,tpr,color='red',label= 'AUC = %0.2f' %auc)
plt.plot([0,1],[0,1],'k--')
plt.legend(loc = 'lower right')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.show()

print('Accuracy is:',auc)


# ## 2. Decision Tree

# In[34]:


from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report,confusion_matrix,f1_score
from sklearn.tree import DecisionTreeRegressor


# In[35]:


# fitting on train data
model = DecisionTreeClassifier(criterion='entropy',max_depth = 5) 
model.fit(X_train,Y_train)


# In[36]:


fig = plt.figure(figsize=(18,9))
tree.plot_tree(model)
plt.show()


# In[37]:


# predict on test data
preds = model.predict(X_test) 
pd.Series(preds).value_counts()


# In[38]:


pd.crosstab(Y_test,preds)


# In[39]:


# Accuracy
np.mean(preds==Y_test)*100


# In[40]:


# Model Evaluation
from sklearn.metrics import classification_report,confusion_matrix,f1_score,accuracy_score
confusion_matrix = confusion_matrix(Y_test,preds)
confusion_matrix


# In[41]:


# Classification Report
print(classification_report(Y_test,preds))


# In[42]:


# ROC Curve plotting and finding AUC value
fpr,tpr,thresholds=roc_curve(Y,model.predict_proba(X)[:,1])
plt.plot(fpr,tpr,color='red')
auc=roc_auc_score(Y_test,preds)

plt.plot(fpr,tpr,color='red',label= 'AUC = %0.2f' %auc)
plt.plot([0,1],[0,1],'k--')
plt.legend(loc = 'lower right')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.show()

print('Accuracy is:',auc)


# ## 3. Random Forest

# In[43]:


# Necessary Libraries
from sklearn.ensemble import RandomForestClassifier


# In[44]:


# Fitting on train data
RF = RandomForestClassifier(n_estimators=100,max_features=3)
RF.fit(X_train,Y_train)


# In[45]:


# Predict on test data
y_predict = RF.predict(X_test)
np.mean(y_predict==Y_test)*100 


# In[46]:


# Model Evaluation
from sklearn.metrics import classification_report,confusion_matrix,f1_score,accuracy_score
confusion_matrix = confusion_matrix(Y_test,y_predict)
confusion_matrix


# In[47]:


# Classification Report
print(classification_report(Y_test,y_predict))


# In[48]:


# ROC Curve plotting and finding AUC value
fpr,tpr,thresholds=roc_curve(Y,RF.predict_proba(X)[:,1])
plt.plot(fpr,tpr,color='red')
auc=roc_auc_score(Y_test,y_predict)

plt.plot(fpr,tpr,color='red',label= 'AUC = %0.2f' %auc)
plt.plot([0,1],[0,1],'k--')
plt.legend(loc = 'lower right')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.show()

print('Accuracy is:',auc)


# ## 4. Support Vector Machines(SVM)

# In[49]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


# In[50]:


svm = SVC(kernel ='linear',C= 0.1, gamma = 50)
svm.fit(X_train , Y_train)
y_pred = svm.predict(X_test)
acc = accuracy_score(Y_test, y_pred) * 100
print("Accuracy =", acc)


# In[51]:


from sklearn.metrics import classification_report,confusion_matrix,f1_score,accuracy_score
confusion_matrix = confusion_matrix(Y_test, y_pred)
confusion_matrix


# In[52]:


clr = classification_report(Y_test,y_pred)
print(clr)


# In[53]:


# ROC Curve plotting and finding AUC value
plt.plot(fpr,tpr,color='red')
auc=roc_auc_score(Y_test,y_pred)

plt.plot(fpr,tpr,color='red',label= 'AUC = %0.2f' %auc)
plt.plot([0,1],[0,1],'k--')
plt.legend(loc = 'lower right')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.show()

print('Accuracy is:',auc)


# ## 5. Gaussian Naive Bayes

# In[54]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB


# In[55]:


# Gaussian Naive Bayes
Gmodel = GaussianNB()
train_pred_gau = Gmodel.fit(X_train,Y_train).predict(X_test)


# In[56]:


#accuracy score on train data using gaussian NB
train_acc_gau=np.mean(Y_test==train_pred_gau)
train_acc_gau


# In[57]:


#confusion matrix for gaussian model 
from  sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(Y_test,train_pred_gau)


# In[58]:


# classification report for gaussian model 
CLF = classification_report(Y_test,train_pred_gau)
print(CLF)


# In[59]:


# ROC Curve plotting and finding AUC value
plt.plot(fpr,tpr,color='red')
auc=roc_auc_score(Y_test,train_pred_gau)

plt.plot(fpr,tpr,color='red',label= 'AUC = %0.2f' %auc)
plt.plot([0,1],[0,1],'k--')
plt.legend(loc = 'lower right')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.show()

print('Accuracy is:',auc)


# In[60]:


# Creating a Performance report for all ML Models
from sklearn.metrics import classification_report, confusion_matrix, precision_score, accuracy_score, recall_score, f1_score


classifier_pred = {'Logistic Regression ':y_pred, 'Decision Tree Classifier':preds,
                   'Random Forest Classifier':y_predict, 'Support Vector Classifier':y_pred, 'Gaussian Naive Bayes':train_pred_gau,}

report = dict()

for key, value in classifier_pred.items():
    # calculating scores 
    accuracy = accuracy_score(Y_test, value)
    precision = precision_score(Y_test, value)
    recall = recall_score(Y_test, value)
    f1 = f1_score(Y_test, value)
    # entering scores in report
    report[key] = [accuracy, precision, recall, f1]

# report dataframe
report_df = pd.DataFrame(data=report, index=['Accuracy', 'Precision', 'Recall', 'F1 Score']).T
report_df.index.name = 'ML Model'
report_df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




