
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("Credit_Risk_Validate_data(1).csv")
data1 = data.drop(["Gender"], axis = 1)


# In[3]:


data1.head()


# In[4]:


X1 = data1.iloc[:, :-1].values
y1 = data1.iloc[:, 11].values


# In[5]:


from sklearn.preprocessing import LabelEncoder , Imputer


# In[6]:


label = LabelEncoder()


# In[7]:


X1[:, 1] = label.fit_transform(X1[:, 1])
X1[:, 3] = label.fit_transform(X1[:, 3])
X1[:, 0] = label.fit_transform(X1[:, 0])
X1[:, 10] = label.fit_transform(X1[:, 10])


# In[8]:


X1


# In[9]:


y1 = label.fit_transform(y1)


# In[10]:


y1


# In[11]:


data1.head()


# In[12]:


# Checking the Nan values
print(data1[data1['Married'].isnull()].index.tolist())
print(data1[data1['Dependents'].isnull()].index.tolist())
print(data1[data1['Education'].isnull()].index.tolist())
print(data1[data1['Self_Employed'].isnull()].index.tolist())
print(data1[data1['ApplicantIncome'].isnull()].index.tolist())
print(data1[data1['CoapplicantIncome'].isnull()].index.tolist())
print(data1[data1['LoanAmount'].isnull()].index.tolist())
print(data1[data1['Loan_Amount_Term'].isnull()].index.tolist())
print(data1[data1['Credit_History'].isnull()].index.tolist())
print(data1[data1['Property_Area'].isnull()].index.tolist())


# In[13]:


from collections import Counter


# In[14]:


Counter(data1['Dependents'])


# In[15]:


pd.crosstab(data1['Married'], data1['Dependents'].isnull())


# In[16]:


pd.crosstab(data1['Dependents'], data1['Married'])


# In[17]:


print(data1[data1['Dependents'].isnull()].index.tolist())


# In[18]:


# For bachelor , lets fill the dependents as 0
bachelor_ = data1[(data1["Married"]=='No') & (data1['Dependents'].isnull())].index.tolist()


# In[19]:


bachelor_


# In[20]:


data1['Dependents'].iloc[bachelor_]='0'


# In[21]:


Counter(data1["Dependents"])


# In[22]:


#Let's fill the remaining ones
married_d = data1[(data1["Married"]=="Yes") & (data1["Dependents"].isnull())].index.tolist()
married_d


# In[23]:


data1["Dependents"].iloc[married_d]="2"


# In[24]:


Counter(data1["Dependents"])


# In[25]:


data1.isnull().sum()


# In[26]:


Counter(data1["Self_Employed"])


# In[27]:


pd.crosstab(data1["Self_Employed"], data1["Married"])


# In[28]:


# Let's manage those ppl who are married and have missing data about their employment, and fill them as not self employed
#
Single_not_self_empl = data1[(data1["Married"]=="Yes")& (data1["Self_Employed"].isnull())].index.tolist()


# In[29]:


data1["Self_Employed"].iloc[Single_not_self_empl]="No"


# In[30]:


Counter(data1["Self_Employed"])


# In[31]:


se_emp = data1[(data1["Married"]=="No") & (data1["Self_Employed"].isnull())].index.tolist()


# In[32]:


data1["Self_Employed"].iloc[se_emp]="Yes"


# In[33]:


Counter(data1["Self_Employed"])


# In[34]:


data1.isnull().sum()


# In[35]:


pd.crosstab(data1["Self_Employed"], data1["Credit_History"])


# In[36]:


# Let's find ppl who have missing credit history and are not self employed and then fill that with 1

workingclass_ = data1[(data1["Self_Employed"]=="No") & (data1["Credit_History"].isnull())].index.tolist()


# In[37]:


data1["Credit_History"].iloc[workingclass_]=1.0


# In[38]:


Counter(data1["Credit_History"])


# In[39]:


bclass_ = data1[(data1["Self_Employed"]=="Yes")&(data1['Credit_History'].isnull())].index.tolist()


# In[40]:


data1["Credit_History"].iloc[bclass_]=0.0


# In[41]:


Counter(data1["Credit_History"])


# In[42]:


data1.isnull().sum()


# In[43]:


from sklearn.preprocessing import Imputer


# In[44]:


imputer  = Imputer(missing_values="NaN", strategy = "mean")


# In[45]:


data1.groupby(data1['Loan_Amount_Term'])['LoanAmount'].mean()


# In[46]:


pd.crosstab(data1['LoanAmount'].isnull(), data1['Loan_Amount_Term'])


# In[47]:


lterm360 = data1[(data1["Loan_Amount_Term"]==360) & (data1["LoanAmount"].isnull())].index.tolist()
lterm480 = data1[(data1["Loan_Amount_Term"]==480) & (data1["LoanAmount"].isnull())].index.tolist()


# In[48]:


lterm360 
lterm480


# In[49]:


data1["LoanAmount"].iloc[lterm360]=139.88


# In[50]:


data1["LoanAmount"].iloc[lterm480]=105.857143


# In[51]:


data1.isnull().sum()


# In[52]:


data1.groupby(data1['Loan_Amount_Term'])['LoanAmount'].mean()


# In[53]:


data1["Loan_Amount_Term"][data1["Loan_Amount_Term"].isnull()]= 360


# In[54]:


data1.isnull().sum()


# In[55]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[56]:


standardscaler = StandardScaler()


# In[59]:


data2= pd.get_dummies(data1)


# In[63]:


data_2 = data1.drop(["Loan_ID"], axis=1)


# In[64]:


data_new = pd.get_dummies(data_2)


# In[66]:


data_new.head()


# In[69]:


X_ = np.array(data_new.drop(["outcome_Y"], axis= 1))
y_ = np.array(data_new["outcome_Y"])


# In[70]:


X_train , X_test , y_train , y_test = train_test_split(X_,y_, test_size = 0.2 , random_state = 0)


# In[74]:


from sklearn.preprocessing import StandardScaler


# In[75]:


std = StandardScaler()


# In[76]:


X_train = std.fit_transform(X_train)


# In[77]:


from sklearn.neural_network import MLPClassifier


# In[78]:


mlp= MLPClassifier(hidden_layer_sizes =(13,13),max_iter=500)


# In[79]:


mlp.fit(X_train , y_train)


# In[92]:


mlp.score(X_test, y_test)


# In[84]:


y_pred = mlp.predict(X_test)


# In[85]:


from sklearn.metrics import confusion_matrix


# In[90]:


cm = confusion_matrix(y_test, y_pred)


# In[91]:


cm

