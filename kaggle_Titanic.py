
# coding: utf-8

# In[1]:

#First Data Science Tutorial on kaggle: Titanic
#Import required packages

import csv
import numpy as np


# In[2]:

#now load the train file 'train.csv'
csv_object = csv.reader(open('train.csv','rb')) #r: read only? b: used because open('train.csv') is a file object
header = csv_object.next() #reads out the first line

data = [] #empty array
#create array
for row in csv_object:
    data.append(row)
    
data = np.array(data) #make numpy array


# In[3]:

# define some basic values (number passengers, survived, relative survivors)
print header
n_passengers = len(data)
n_survived = np.sum(data[0::,1].astype(np.float))
rel_survived = n_survived/n_passengers


# In[4]:

#now we create subset of only men or women datasets
women_only_bool = data[0::,4] == 'female'
men_only_bool = data[0::,4] != 'female'

#these arrays of bools can be used to create subsets from the data
women_surv=data[women_only_bool,1]
men_surv=data[men_only_bool,1]
rel_women_survived=np.sum(women_surv.astype(np.float))/len(women_surv)
rel_men_survived=np.sum(men_surv.astype(np.float))/len(men_surv)
#output
print '%s%% of the female passengers survived' % (rel_women_survived*100)
print '%s%% of the male passengers survived' % (rel_men_survived*100)


# In[5]:

#now load the test file
csv_object = csv.reader(open('test.csv','rb'))
csv_object.next()
test_data = [] #empty array
#create array
for row in csv_object:
    test_data.append(row)
    
test_data = np.array(test_data) #make numpy array


# In[6]:

""""
#open the results file
prediction_file = open("genderbasedmodel_py.csv","wb")
prediction_file_object = csv.writer(prediction_file)
#and write predictions (first header)
prediction_file_object.writerow(["PassengerID","Survived"])
#now go through the file
for pid,gender in zip(test_data[:,0],test_data[:,3]):
    if gender=='female':
        prediction_file_object.writerow([pid,'1'])
    else:
        prediction_file_object.writerow([pid,'0'])

prediction_file.close()
"""


# In[7]:

#we now iclude ticket classes and prizes
#for the prizes we make 4 bins: $0-9, $10-19, $21-30, >$30

#apparently for later processing we cut all the prices >$39 to 39
fare_ceiling=40
data[ data[0::,9].astype(np.float) >= fare_ceiling, 9 ] = fare_ceiling - 1.0

#define prize range for each bin
fare_binsize=10
n_price_brackets = fare_ceiling / fare_binsize

#3 classes on board (directly extract from data)
n_classes=len(np.unique(data[0::,2]))

#now create the relative survival table (gender, class, prize)
survival_table = np.zeros((2, n_classes, n_price_brackets))


# In[8]:

for i in range(n_classes):
    for j in range(n_price_brackets):
        #make array of females survived in the respecive category
        females_survived=data[(data[::,4]=='female')&(data[::,2].astype(np.int)==i+1)                            &(data[::,9].astype(np.float)>=j*fare_binsize)&                             (data[::,9].astype(np.float)<(j+1)*fare_binsize),1]
        #same for the males
        males_survived=data[(data[::,4]=='male')&(data[::,2].astype(np.int)==i+1)                            &(data[::,9].astype(np.float)>=j*fare_binsize)&                             (data[::,9].astype(np.float)<(j+1)*fare_binsize),1]
        
        #enter values into array
        survival_table[0,i,j]=np.mean(females_survived.astype(np.float))
        survival_table[1,i,j]=np.mean(males_survived.astype(np.float))
        
#strangely, 'survival_table[survival_table != survival_table ]' picks all the nan entries
survival_table[survival_table != survival_table ] = 0.0
survival_table


# In[9]:

#make clear decision
survival_table[survival_table>=0.5] = 1
survival_table[survival_table<0.5] = 0
survival_table


# In[10]:

#now load the test file, again
csv_object = csv.reader(open('test.csv','rb'))
csv_object.next()
test_data = [] #empty array
#create array
for row in csv_object:
    test_data.append(row)
    
test_data = np.array(test_data) #make numpy array

#and open the results file
prediction_file = open("genderbasedmodel2_py.csv","wb")
prediction_file_object = csv.writer(prediction_file)
#and write predictions (first header)
prediction_file_object.writerow(["PassengerID","Survived"])

#now lets fill in the predictions
for row in test_data:
    #prize binning
    for j in range(n_price_brackets):
        
        #some passengers have no price entry -> bin them according to their classes
        try:
            row[8] = float(row[8])
        except:
            # bin the fare according Pclass
            bin_fare = int(3 - float(row[1]))
            break #break leaves the loop -> efficient

        #if prize above bins, put it in highest bin
        if float(row[8]) >= fare_ceiling:
            bin_fare = n_price_brackets -1
            break
        
        #now normal binning
        if float(row[8]) >= j*fare_binsize and float(row[8]) < (j+1)*fare_binsize:
            bin_fare=j
            break
            
    # the gender case
    if row[3]=='female':
        prediction_file_object.writerow([row[0],                                        '%d' % survival_table[0,int(row[1])-1,bin_fare]])
#        print row[0],type(0),type(int(row[1])-1),type(bin_fare),survival_table[0,int(row[1])-1,bin_fare]
    else:
        prediction_file_object.writerow([row[0],                                         '%d' % survival_table[1,int(row[1])-1,bin_fare]])
#        print row[0],1,int(row[1])-1,bin_fare,survival_table[0,int(row[1])-1,bin_fare];


prediction_file.close()


# In[170]:

#now some playing with Pandas
import pandas as pd
#use pandas data frame
df = pd.read_csv('train.csv', header=0)
# test data
tf = pd.read_csv('test.csv',header=0)
print 'data frime type',type(df)
#df.dtypes
#df.info()
print df.columns.values
#df[df['Cabin']==df['Cabin']]['Cabin']
print df[df['Sex']=='female']['Age'].mean()
print df[df['Sex']=='male']['Age'].mean() #cool
df[df['Age']>55][['Age','Sex']] # SE
df[df['Age'].isnull()][['Age','Sex']] #select entries that have no value in the specific key
for i in range(1,4):
    print 'Class', i, 'No Passengers', len( df[ (df['Sex']=='male') & (df['Pclass']==i) ] ) # must be '&'; 'and' does not work


# In[171]:

import pylab as P
df['Age'].dropna().hist(bins = 16, range=(0,80), alpha = .5 ) # dropna removes missing entries; 
P.show()


# In[184]:

df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )
df['Gender'].head(2)


# In[185]:

df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int) #assigns numbers
tf['Gender'] = tf['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


# In[186]:

median_ages = np.zeros((2,3)) # 2 genders 3 classes
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) &                               (df['Pclass'] == j+1)]['Age'].dropna().median()
 
median_ages

#now create from these values the ages for missing passengers
df['AgeFill'] = df['Age']
tf['AgeFill'] = tf['Age']
for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),                'AgeFill'] = median_ages[i,j]
        tf.loc[ (tf.Age.isnull()) & (tf.Gender == i) & (tf.Pclass == j+1), 'AgeFill']        = median_ages[i,j]

tf[tf['Age'].isnull()][['Sex','Pclass','Age','AgeFill']].head(2)


# In[187]:

# do the same for ticket prices (one is missing in the test set) based on class, gender,
# and departing port (Q/S/C)

#array with the port names
ports=['Q','S','C'];

median_prices = np.zeros((2,3,3)) # 2 genders 3 classes 3 ports
for i in range(2):
    for j in range(3):
        for k in range(3):
            median_prices[i,j,k] = df[(df['Gender'] == i) &                                   (df['Pclass'] == j+1) & (df['Embarked']==ports[k])]                                        ['Fare'].dropna().median()

#now create from these values the ages for missing passengers (training set is already complete)
tf['FareFill'] = tf['Fare']
df['FareFill'] = df['Fare'] #will still be needed for the fitting
for i in range(2):
    for j in range(3):
        for k in range(3):
            tf.loc[ (tf.Fare.isnull()) & (tf.Gender == i) & (tf.Pclass == j+1) & (tf.Embarked==ports[k]), 'FareFill']                = median_prices[i,j,k]

tf[tf['Fare'].isnull()][['Sex','Pclass','Fare','FareFill']].head(2)


# In[188]:

df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.AgeFill * df.Pclass
tf['FamilySize'] = tf['SibSp'] + tf['Parch']
tf['Age*Class'] = tf.AgeFill * tf.Pclass


# In[189]:

df.dtypes[df.dtypes.map(lambda x: x=='object')]
tfd.head(1)


# In[190]:

dfd = df.drop(['PassengerId','Name','Sex','Ticket','Cabin','Embarked','Fare'],axis=1)
dfd = dfd.drop(['Age'],axis=1) # since we have the full estimated age list
tfd = tf.drop(['PassengerId','Name','Sex','Ticket','Cabin','Embarked','Age','Fare'],axis=1)


# In[197]:

train_data = dfd.values
test_data = tfd.values
print train_data.shape, test_data.shape

# import the random forest guy
from sklearn.ensemble import RandomForestClassifier 

# Create the random forest object which will include all the parameters for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::,1::],train_data[0::,0]) # fit(X,y)

# Take the same decision trees and run it on the test data
output = forest.predict(test_data)


# In[203]:

dfp=pd.DataFrame(columns=['PassengerId','Survived'])
dfp['PassengerId']=tf['PassengerId']
dfp['Survived']=output
dfp.to_csv('Titanic_RandForest.csv', index=False)


# In[ ]:




# In[ ]:



