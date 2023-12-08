# imports pandas package
import pandas as pd
# imports numpy package
import numpy as np
# imports matplotlib package
import matplotlib
# imports pylab from matplotlib
from matplotlib import pylab as plt
# imports train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split
# imports KFold from sklearn.model_selection
from sklearn.model_selection import KFold
# imports ColumnTransformer from sklearn.compose
from sklearn.compose import ColumnTransformer
# imports Pipeline from sklearn.pipeline
from sklearn.pipeline import Pipeline
# imports StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler from sklearn.preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler

# converts data in the excel file into pandas dataframe
df = pd.read_csv(r'/Users/jamesro/Documents/DATA1030-Fall2021/Data-1030-Project/Data/student-mat.csv',sep=';')
# converts G3 into binary scale
pass_final = df.G3 >= 10
fail_final = df.G3 < 10
df.loc[pass_final,'G3'] = 1
df.loc[fail_final,'G3'] = 0
# converts G1 into binary scale
pass_first = df.G1 >= 10
fail_first = df.G1 < 10
df.loc[pass_first,'G1'] = 1
df.loc[fail_first,'G1'] = 0
# converts G2 into binary scale
pass_sec = df.G2 >= 10
fail_sec = df.G2 < 10
df.loc[pass_sec,'G2'] = 1
df.loc[fail_sec,'G2'] = 0
# prints dataframe
print(df)

# prints the number of rows
print('number of rows: ' , df.shape[0])
# prints the number of columns
print('number of columns: ' , df.shape[1])

# prints data types of all columns
print(df.dtypes)

# loops through all columns
for (columnName, columnData) in df.iteritems():
    # identifies any columns with less than 10 unique values or has a dtypes of object to be categorical 
    if df[columnName].nunique()<=7 or df[columnName].dtypes == "object":
        # describes the categorical feature
        print(df[columnName].value_counts())
        # creates a bar graph if feature is categorical
        pd.value_counts(df[columnName]).plot.bar()
        # labels y axis as count
        plt.ylabel('count')
        # labels x axis with the column name
        plt.xlabel(columnName)
        # shows the bar graph
        plt.show()
        # creates stacked bar plot for categorical feature and target variable G3
        count_matrix = df.groupby([columnName, 'G3']).size().unstack()
        count_matrix_norm = count_matrix.div(count_matrix.sum(axis=1),axis=0)
        objects = ('Fail', 'Pass')
        count_matrix_norm.plot(kind='bar', stacked=True)
        plt.ylabel('Proportion of Students Passing/Failing')
        plt.legend(labels=objects,loc=4)
        plt.show()
        
    # in every other case, columns are continuous
    else:
        # describes the continuous feature
        print(df[columnName].describe())
        # specifies bins as argument values for histogram
        bins = df[columnName].nunique()
        # creates a histogram if feature is continuous
        df[columnName].plot.hist(bins = bins)
        # labels x axis with the column name
        plt.xlabel(columnName)
        # labels y axis as count
        plt.ylabel('count')
        # shows the histogram
        plt.show()
        # plots category-specific histogram of continuous feature and G3
        categories = df['G3'].unique()
        bin_range = (df[columnName].min(),df[columnName].max())
        objects = ('Fail', 'Pass')
        
        for c in categories:
            plt.hist(df[df['G3']==c][columnName],alpha=0.5,label=c,range=bin_range,bins=20,density=True)
        plt.legend()
        plt.ylabel('Proportion of Passing/Failing Grades')
        plt.xlabel(columnName)
        plt.legend(labels=objects,loc=4)
        plt.show()

# plots bar plot of G3, since it is categorical
pd.value_counts(df['G3']).plot.bar()
plt.ylabel('count')
plt.xlabel('Final Grade')
plt.title('Distribution of Final Grade')
plt.xticks([0,1],['Pass','Fail'])
plt.savefig('/Users/jamesro/Documents/DATA1030-Fall2021/Data-1030-Project/Figures/Final_Distribution.jpg')
plt.show()

# plots stacked bar plot of sex and G3
count_matrix = df.groupby(['sex', 'G3']).size().unstack()
count_matrix_norm = count_matrix.div(count_matrix.sum(axis=1),axis=0)
objects = ('Fail', 'Pass')
count_matrix_norm.plot(kind='bar', stacked=True)
plt.ylabel('Proportion of Students Passing/Failing')
plt.legend(labels=objects,loc=4)
plt.xticks([0,1],['Female','Male'],rotation=0)
plt.title('Final Grade Distribution By Sex')
plt.savefig('/Users/jamesro/Documents/DATA1030-Fall2021/Data-1030-Project/Figures/SEX-GRADE.jpg')
plt.show()

# plots stacked bar plot of G2 and G3
count_matrix = df.groupby(['G2', 'G3']).size().unstack()
count_matrix_norm = count_matrix.div(count_matrix.sum(axis=1),axis=0)
count_matrix_norm.plot(kind='bar', stacked=True)
plt.ylabel('Proportion of Students Passing/Failing (Final Grade)')
plt.xlabel('G2 - Second Period Grades')
plt.xticks([0,1],['Fail','Pass'],rotation=0)
plt.legend(labels=objects,loc=4)
plt.title('Final Grade Distribution By Second Period Grades')
plt.savefig('../Figures/G2-GRADE.jpg', dpi=300)
plt.show()

# plots stacked bar plot of G1 and G3
count_matrix = df.groupby(['G1', 'G3']).size().unstack()
count_matrix_norm = count_matrix.div(count_matrix.sum(axis=1),axis=0)
count_matrix_norm.plot(kind='bar', stacked=True)
plt.ylabel('Proportion of Students Passing/Failing (Final Grade)')
plt.xlabel('G1 - First Period Grades')
plt.xticks([0,1],['Fail','Pass'],rotation=0)
plt.legend(labels=objects,loc=4)
plt.title('Final Grade Distribution By First Period Grades')
plt.savefig('../Figures/G1-GRADE.jpg', dpi=300)
plt.show()

# plots category-specific histogram of absences and G3
categories = df['G3'].unique()
bin_range = (df['absences'].min(),df['absences'].max())

for c in categories:
    plt.hist(df[df['G3']==c]['absences'],alpha=0.5,label=c,range=bin_range,bins=20,density=True)
plt.legend()
plt.ylabel('Proportion of Passing/Failing Grades')
plt.xlabel('Number of Absences')
plt.legend(labels=objects,loc=4)
plt.title('Number of Absences VS Student Performance')
plt.savefig('/Users/jamesro/Documents/DATA1030-Fall2021/Data-1030-Project/Figures/ABSENCE-GRADE.jpg', dpi=300)
plt.show()

# plots category-specific histogram of age and G3 
bin_range = (df['age'].min(),df['age'].max())

for c in categories:
    plt.hist(df[df['G3']==c]['age'],alpha=0.5,label=c,range=bin_range,bins=20,density=True)
plt.legend()
plt.ylabel('Proportion of Passing/Failing Grades')
plt.xlabel('Age')
plt.legend(labels=objects,loc=4)
plt.title('Age VS Student Performance')
plt.savefig('/Users/jamesro/Documents/DATA1030-Fall2021/Data-1030-Project/Figures/AGE-GRADE.jpg')
plt.show()

# plots scatter matrix
pd.plotting.scatter_matrix(df.select_dtypes(int), figsize=(20, 20), marker='o',hist_kwds={'bins': 50}, 
                           s=30, alpha=.1)
plt.savefig('/Users/jamesro/Documents/DATA1030-Fall2021/Data-1030-Project/Figures/SCATTERMATRIX.jpg')
plt.show()

# defines random_state as 42
random_state = 42
# defines X as all feature columns
X = df.loc[:, df.columns != 'G3']
# defines y as the target variable G3
y = df['G3']

# first split to separate out the test set
X_other, X_test, y_other, y_test = train_test_split(X,y,test_size = 0.2,random_state=random_state)

# do KFold split on other
kf = KFold(n_splits=5,shuffle=True,random_state=random_state)
for train_index, val_index in kf.split(X_other,y_other):
    X_train = X_other.iloc[train_index]
    y_train = y_other.iloc[train_index]
    X_val = X_other.iloc[val_index]
    y_val = y_other.iloc[val_index]
    print(X_train.head())
    print(X_val.head())
    print(X_test.head())
    onehot_ftrs = ['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']
    minmax_ftrs = ['age','absences']
    std_ftrs = ['Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout','Dalc','Walc','health','G1','G2']

    # collects all the encoders
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(sparse=False,handle_unknown='ignore'), onehot_ftrs),
            ('minmax', MinMaxScaler(), minmax_ftrs),
            ('std', StandardScaler(), std_ftrs)
            ])

    clf = Pipeline(steps=[('preprocessor', preprocessor)])

    # transforms X_train into X_train_prep
    X_train_prep = clf.fit_transform(X_train)
    # transforms X_val into X_val_prep
    X_val_prep = clf.transform(X_val)
    # transforms X_test into X_test_prep
    X_test_prep = clf.transform(X_test)

    # prints X_train_prep
    print(X_train_prep)
    # prints X_val_prep
    print(X_val_prep)
        # prints X_test_prep
    print(X_test_prep)
    print(X_train_prep.shape)
    print(X_val_prep.shape)
    print(X_test_prep.shape)



    ####### SPLITTING AND PREPROCESSING

    # imports pandas package
import pandas as pd
# imports numpy package
import numpy as np
# imports matplotlib package
import matplotlib
# imports pylab from matplotlib
from matplotlib import pylab as plt
# imports train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split
# imports KFold from sklearn.model_selection
from sklearn.model_selection import KFold
# imports ColumnTransformer from sklearn.compose
from sklearn.compose import ColumnTransformer
# imports Pipeline from sklearn.pipeline
from sklearn.pipeline import Pipeline
# imports StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler from sklearn.preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler

# converts data in the excel file into pandas dataframe
df = pd.read_csv(r'/Users/jamesro/Documents/DATA1030-Fall2021/Data-1030-Project/Data/student-mat.csv',sep=';')
# converts G3 into binary scale
pass_final = df.G3 >= 10
fail_final = df.G3 < 10
df.loc[pass_final,'G3'] = 1
df.loc[fail_final,'G3'] = 0
# converts G1 into binary scale
pass_first = df.G1 >= 10
fail_first = df.G1 < 10
df.loc[pass_first,'G1'] = 1
df.loc[fail_first,'G1'] = 0
# converts G2 into binary scale
pass_sec = df.G2 >= 10
fail_sec = df.G2 < 10
df.loc[pass_sec,'G2'] = 1
df.loc[fail_sec,'G2'] = 0
# prints dataframe
print(df)

y = df['G3']
X = df.loc[:, df.columns != 'G3']

# converts data in the excel file into pandas dataframe
dff = pd.read_csv(r'/Users/jamesro/Documents/DATA1030-Fall2021/Data-1030-Project/Data/student-mat.csv',sep=';')
# converts G3 into binary scale
pass_final = dff.G3 >= 10
fail_final = dff.G3 < 10
dff.loc[pass_final,'G3'] = 1
dff.loc[fail_final,'G3'] = 0
# converts G1 into binary scale
pass_first = dff.G1 >= 10
fail_first = dff.G1 < 10
dff.loc[pass_first,'G1'] = 1
dff.loc[fail_first,'G1'] = 0
# drop G2 feature
dff.drop('G2', axis=1, inplace=True)
print(dff)
yy = dff['G3']
XX = dff.loc[:, dff.columns != 'G3']

# converts data in the excel file into pandas dataframe
dfff = pd.read_csv(r'/Users/jamesro/Documents/DATA1030-Fall2021/Data-1030-Project/Data/student-mat.csv',sep=';')
# converts G3 into binary scale
pass_final = dfff.G3 >= 10
fail_final = dfff.G3 < 10
dfff.loc[pass_final,'G3'] = 1
dfff.loc[fail_final,'G3'] = 0
# drop G1 and G2 features
dfff.drop('G1', axis=1, inplace=True)
dfff.drop('G2', axis=1, inplace=True)
print(dfff)
yyy = dfff['G3']
XXX = dfff.loc[:, dfff.columns != 'G3']

test_sets = []
other_sets = []
test_sets_wo = []
other_sets_wo = []
test_sets_gt = []
other_sets_gt = []

for i in range(10):
    X_other, X_test, y_other, y_test = train_test_split(X,y,test_size = 0.2,random_state=42*i)
    test_sets.append((X_test, y_test))
    other_sets.append((X_other, y_other))

for i in range(10):
    XX_other, XX_test, yy_other, yy_test = train_test_split(XX,yy,test_size = 0.2,random_state=42*i)
    test_sets_wo.append((XX_test, yy_test))
    other_sets_wo.append((XX_other, yy_other))
    
for i in range(10):
    XXX_other, XXX_test, yyy_other, yyy_test = train_test_split(XXX,yyy,test_size = 0.2,random_state=42*i)
    test_sets_gt.append((XXX_test, yyy_test))
    other_sets_gt.append((XXX_other, yyy_other))

import pickle
file = open('../data/data_prep.save', 'wb')
pickle.dump((other_sets, test_sets),file)
file.close()

import pickle
file = open('../data/data_prep_wo.save', 'wb')
pickle.dump((other_sets_wo, test_sets_wo),file)
file.close()


### Tuning hyperparameters

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, ParameterGrid
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
# imports KFold from sklearn.model_selection
from sklearn.model_selection import KFold
# imports ColumnTransformer from sklearn.compose
from sklearn.compose import ColumnTransformer
# imports Pipeline from sklearn.pipeline
from sklearn.pipeline import Pipeline
# imports StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler from sklearn.preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

file = open('../data/data_prep_gt.save', 'rb')
other_sets_gt, test_sets_gt = pickle.load(file)
file.close()

# converts data in the excel file into pandas dataframe
dfff = pd.read_csv(r'/Users/jamesro/Documents/DATA1030-Fall2021/Data-1030-Project/Data/student-mat.csv',sep=';')
# converts G3 into binary scale
pass_final = dfff.G3 >= 10
fail_final = dfff.G3 < 10
dfff.loc[pass_final,'G3'] = 1
dfff.loc[fail_final,'G3'] = 0
# drop G1 and G2 features
dfff.drop('G1', axis=1, inplace=True)
dfff.drop('G2', axis=1, inplace=True)
print(dfff)
yyy = dfff['G3']
XXX = dfff.loc[:, dfff.columns != 'G3']


# imports pandas package
import pandas as pd
# imports numpy package
import numpy as np
# imports matplotlib package
import matplotlib
# imports pylab from matplotlib
from matplotlib import pylab as plt
# imports train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split
# imports KFold from sklearn.model_selection
from sklearn.model_selection import KFold
# imports ColumnTransformer from sklearn.compose
from sklearn.compose import ColumnTransformer
# imports Pipeline from sklearn.pipeline
from sklearn.pipeline import Pipeline
# imports StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler from sklearn.preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler

onehot_ftrs = ['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']
minmax_ftrs = ['age','absences']
std_ftrs = ['Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout','Dalc','Walc','health']

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(sparse=False,handle_unknown='ignore'), onehot_ftrs),
        ('minmax', MinMaxScaler(), minmax_ftrs),
        ('std', StandardScaler(), std_ftrs)
        ])

def MLpipe_KFold_Accu(preprocessor, ML_algo, param_grid):
    '''
    This function splits the data to other/test (80/20) and then applies KFold with 4 folds to other.
    The RMSE is minimized in cross-validation.
    '''
    nr_states = 10
    test_scores = np.zeros(nr_states)
    best_models = []

    for i in range(nr_states):
        
        X_other, y_other = other_sets_gt[i]
        X_test, y_test = test_sets_gt[i]

        kf = KFold(n_splits=4,shuffle=True,random_state=42*i)
         
        pipe = make_pipeline(preprocessor,ML_algo)
        
        grid = GridSearchCV(pipe, param_grid=param_grid, scoring = 'accuracy',
                        cv=kf, return_train_score = True, n_jobs=-1, verbose=True)
        
        grid.fit(X_other, y_other)
        results = pd.DataFrame(grid.cv_results_)
        print('best model parameters:',grid.best_params_)
        
        print('validation score:',grid.best_score_) # this is the mean validation score over all iterations
        # save the model
        best_models.append(grid)
        # calculate RMSE value for test set
        y_test_pred = best_models[-1].predict(X_test)
        test_scores[i] = accuracy_score(y_test,y_test_pred)
        print('test score:',test_scores[i])
        
    return best_models, test_scores, grid, X_test, y_test


print(dfff.G3.value_counts())


baseline_accuracy = dfff.G3.value_counts()[1]/len(dfff)
print("baseline accuracy score: ", baseline_accuracy)


from sklearn.linear_model import LogisticRegression
import math 

l1_param_grid = {
                 'logisticregression__C': [1e-2, 1e-1, 1e0, 1e1, 1e2],
                 'logisticregression__max_iter': [10e5]
                 } 

L1R = LogisticRegression(penalty='l1', solver='saga')
l1_best_models, l1_test_scores, l1_grid, l1_X_test, l1_y_test = MLpipe_KFold_Accu(preprocessor, L1R, l1_param_grid)

l1_mean = np.mean(l1_test_scores)
l1_std = np.std(l1_test_scores)

print('mean test score: ',l1_mean)
print('std of test score: ',l1_std)
print('95% Confidence Interval: ',(l1_mean - 1.96*(l1_std/math.sqrt(5)), l1_mean + 1.96*(l1_std/math.sqrt(5))))
print('standard deviations from baseline: ',(l1_mean - baseline_accuracy)/l1_std)


### Logistic Regression

from sklearn.linear_model import LogisticRegression
import math 

l2_param_grid = {
                 'logisticregression__C': [1e-2, 1e-1, 1e0, 1e1, 1e2], 
                 'logisticregression__max_iter': [10e5]
                 } 

L2R = LogisticRegression(penalty='l2', solver='saga')
l2_best_models, l2_test_scores, l2_grid, l2_X_test, l2_y_test = MLpipe_KFold_Accu(preprocessor, L2R, l2_param_grid)

l2_mean = np.mean(l2_test_scores)
l2_std = np.std(l2_test_scores)

print('mean test score: ',l2_mean)
print('std of test score: ',l2_std)
print('95% Confidence Interval: ',(l2_mean - 1.96*(l2_std/math.sqrt(5)), l2_mean + 1.96*(l2_std/math.sqrt(5))))
print('standard deviations from baseline: ',(l2_mean - baseline_accuracy)/l2_std)


### l2 regression

from sklearn.linear_model import LogisticRegression
import math 

l2_param_grid = {
                 'logisticregression__C': [1e-2, 1e-1, 1e0, 1e1, 1e2], 
                 'logisticregression__max_iter': [10e5]
                 } 

L2R = LogisticRegression(penalty='l2', solver='saga')
l2_best_models, l2_test_scores, l2_grid, l2_X_test, l2_y_test = MLpipe_KFold_Accu(preprocessor, L2R, l2_param_grid)

l2_mean = np.mean(l2_test_scores)
l2_std = np.std(l2_test_scores)

print('mean test score: ',l2_mean)
print('std of test score: ',l2_std)
print('95% Confidence Interval: ',(l2_mean - 1.96*(l2_std/math.sqrt(5)), l2_mean + 1.96*(l2_std/math.sqrt(5))))
print('standard deviations from baseline: ',(l2_mean - baseline_accuracy)/l2_std)

### elastic net

from sklearn.linear_model import LogisticRegression
import math 

en_param_grid = {
                 'logisticregression__C': [1e-2, 1e-1, 1e0, 1e1, 1e2],
                 'logisticregression__l1_ratio': [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
                 'logisticregression__max_iter': [10000]
                 } 

EN = LogisticRegression(penalty='elasticnet', solver='saga')
en_best_models, en_test_scores, en_grid, en_X_test, en_y_test = MLpipe_KFold_Accu(preprocessor, EN, en_param_grid)

en_mean = np.mean(en_test_scores)
en_std = np.std(en_test_scores)

print('mean test score: ',en_mean)
print('std of test score: ',en_std)
print('95% Confidence Interval: ',(en_mean - 1.96*(en_std/math.sqrt(35)), en_mean + 1.96*(en_std/math.sqrt(35))))
print('standard deviations from baseline: ',(en_mean - baseline_accuracy)/en_std)

### random Forest


from sklearn.ensemble import RandomForestClassifier
import math 

rfc_param_grid = {
                   'randomforestclassifier__max_depth': [1, 3, 10, 30, 100],
                   'randomforestclassifier__max_features': [0.5,0.75,1.0] 
                   } 

ML_algo = RandomForestClassifier()
rfc_best_models, rfc_test_scores, rfc_grid, rfc_X_test, rfc_y_test = MLpipe_KFold_Accu(preprocessor, ML_algo, rfc_param_grid)

rfc_mean = np.mean(rfc_test_scores)
rfc_std = np.std(rfc_test_scores)

print('mean test score: ',rfc_mean)
print('std of test score: ',rfc_std)
print('95% Confidence Interval: ',(rfc_mean - 1.96*(rfc_std/math.sqrt(15)), rfc_mean + 1.96*(rfc_std/math.sqrt(15))))
print('standard deviations from baseline: ',(rfc_mean - baseline_accuracy)/rfc_std)



### SVM

from sklearn.svm import SVC
import math 

svc_param_grid = {
                 'svc__gamma': [1000000, 1000, 1, 0.001],
                 'svc__C': [1, 10, 30, 100]
                 } 

SVC = SVC()
svc_best_models, svc_test_scores, svc_grid, svc_X_test, svc_y_test = MLpipe_KFold_Accu(preprocessor, SVC, svc_param_grid)

svc_mean = np.mean(svc_test_scores)
svc_std = np.std(svc_test_scores)

print('mean test score: ',svc_mean)
print('std of test score: ',svc_std)
print('95% Confidence Interval: ',(svc_mean - 1.96*(svc_std/math.sqrt(45)), svc_mean + 1.96*(svc_std/math.sqrt(45))))
print('standard deviations from baseline: ',(svc_mean - baseline_accuracy)/svc_std)


#### KNN
from sklearn.neighbors import KNeighborsClassifier
import math 

knn_param_grid = {
                   'kneighborsclassifier__n_neighbors': [1, 10, 30, 100], 
                   'kneighborsclassifier__weights': ['uniform', 'distance']
                   } 

KNN = KNeighborsClassifier()
knn_best_models, knn_test_scores, knn_grid, knn_X_test, knn_y_test = MLpipe_KFold_Accu(preprocessor, KNN, knn_param_grid)

knn_mean = np.mean(knn_test_scores)
knn_std = np.std(knn_test_scores)

print('mean test score: ',knn_mean)
print('std of test score: ',knn_std)
print('95% Confidence Interval: ',(knn_mean - 1.96*(knn_std/math.sqrt(8)), knn_mean + 1.96*(knn_std/math.sqrt(8))))
print('standard deviations from baseline: ',(knn_mean - baseline_accuracy)/knn_std)



#### compare models

model_name = ["Lasso", "Ridge", "EN", "RF", "SVC", "KNN"]
mean_scores = [l1_mean, l2_mean, en_mean, rfc_mean, svc_mean, knn_mean]
stdev_scores = [l1_std, l2_std, en_std, rfc_std, svc_std, knn_std]

plt.bar(model_name, mean_scores, yerr=stdev_scores, capsize=2)
plt.ylim([0.4,0.8])
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy Score")
plt.xticks(rotation=0)
plt.title("ML Models VS Accuracy Score")
plt.savefig('../figures/mlmodels_accu_gt.jpg', dpi=300)
plt.show()


#### best model selection

# imports pandas package
import pandas as pd
# imports numpy package
import numpy as np
# imports matplotlib package
import matplotlib
# imports pylab from matplotlib
from matplotlib import pylab as plt
# imports train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split
# imports KFold from sklearn.model_selection
from sklearn.model_selection import KFold
# imports ColumnTransformer from sklearn.compose
from sklearn.compose import ColumnTransformer
# imports Pipeline from sklearn.pipeline
from sklearn.pipeline import Pipeline
# imports StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler from sklearn.preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler

# converts data in the excel file into pandas dataframe
dfff = pd.read_csv(r'/Users/jamesro/Documents/DATA1030-Fall2021/Data-1030-Project/Data/student-mat.csv',sep=';')
# converts G3 into binary scale
pass_final = dfff.G3 >= 10
fail_final = dfff.G3 < 10
dfff.loc[pass_final,'G3'] = 1
dfff.loc[fail_final,'G3'] = 0
# drop G1 and G2 features
dfff.drop('G1', axis=1, inplace=True)
dfff.drop('G2', axis=1, inplace=True)
print(dfff)
yyy = dfff['G3']
XXX = dfff.loc[:, dfff.columns != 'G3']


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

def ML_pipeline_kfold(X,y,random_state,n_folds):
    # create a test set
    X_other, X_test, y_other, y_test = train_test_split(X, y, test_size=0.2, random_state = random_state)
    # splitter for _other
    kf = KFold(n_splits=n_folds,shuffle=True,random_state=random_state)
    # create the pipeline: preprocessor + supervised ML method
    cat_ftrs = ['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']
    cont_ftrs = ['age','Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout','Dalc','Walc','health','absences']

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(sparse=False,handle_unknown='ignore'))])
    # standard scaler
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, cont_ftrs),
            ('cat', categorical_transformer, cat_ftrs)])
    pipe = make_pipeline(preprocessor,RandomForestClassifier(n_estimators =  100,random_state=random_state))
    # the parameter(s) we want to tune
    rfc_param_grid = {
                   'randomforestclassifier__max_depth': [1, 3, 10, 30, 100], 
                   'randomforestclassifier__max_features': [0.5,0.75,1.0] 
                   } 
    # prepare gridsearch
    grid = GridSearchCV(pipe, param_grid=rfc_param_grid,cv=kf, return_train_score = True,n_jobs=-1,verbose=10)
    # do kfold CV on _other
    grid.fit(X_other, y_other)
    feature_names = cont_ftrs + \
                list(grid.best_estimator_[0].named_transformers_['cat'][0].get_feature_names(cat_ftrs))
    return grid, np.array(feature_names), X_test, y_test


grid, feature_names, X_test, y_test = ML_pipeline_kfold(XXX,yyy,42,5)
print(grid.best_score_)
print(grid.score(X_test,y_test))
print(grid.best_params_)
print(feature_names)


import shap
shap.initjs() # required for visualizations later on
# create the explainer object with the random forest model
explainer = shap.TreeExplainer(grid.best_estimator_[1])
# transform the test set
X_test_transformed = grid.best_estimator_[0].transform(X_test)
print(np.shape(X_test_transformed))
shap_values = explainer.shap_values(X_test_transformed)
print(shap_values)
print(np.shape(shap_values))

shap.summary_plot(shap_values, X_test_transformed,feature_names = feature_names)

ftr_names = XXX.columns
np.random.seed(42)

nr_runs = 10
scores = np.zeros([len(ftr_names),nr_runs])

test_score = grid.score(X_test,y_test)
print('test score = ',test_score)
print('test baseline = ',np.sum(y_test == 1)/len(y_test))
# loop through the features
for i in range(len(ftr_names)):
    print('shuffling '+str(ftr_names[i]))
    acc_scores = []
    for j in range(nr_runs):
        X_test_shuffled = X_test.copy()
        X_test_shuffled[ftr_names[i]] = np.random.permutation(X_test[ftr_names[i]].values)
        acc_scores.append(grid.score(X_test_shuffled,y_test))
    print('   shuffled test score:',np.around(np.mean(acc_scores),3),'+/-',np.around(np.std(acc_scores),3))
    scores[i] = acc_scores


sorted_indcs = np.argsort(np.mean(scores,axis=1))[::-1]
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(8,6))
plt.boxplot(scores[sorted_indcs].T,labels=ftr_names[sorted_indcs],vert=False)
plt.axvline(test_score,label='test score')
plt.title("Permutation Importances (test set)")
plt.xlabel('score with perturbed feature')
plt.legend(loc = 'lower center')
plt.tight_layout()
plt.savefig('../figures/perm_imp_gt.jpg', dpi=300)
plt.show()

from sklearn.linear_model import LogisticRegression
def ML_pipeline_kfold_LR2(X,y,random_state,n_folds):
    # create a test set
    X_other, X_test, y_other, y_test = train_test_split(X, y, test_size=0.2, random_state = random_state)
    # splitter for _other
    kf = KFold(n_splits=n_folds,shuffle=True,random_state=random_state)
    # create the pipeline: preprocessor + supervised ML method
    cat_ftrs = ['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']
    cont_ftrs = ['age','Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout','Dalc','Walc','health','absences']

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(sparse=False,handle_unknown='ignore'))])
    # standard scaler
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, cont_ftrs),
            ('cat', categorical_transformer, cat_ftrs)])
    final_scaler = StandardScaler()
    pipe = make_pipeline(preprocessor,final_scaler,LogisticRegression(penalty='l2',solver='saga'))
    # the parameter(s) we want to tune
    l2_param_grid = {
                 'logisticregression__C': [1e-2, 1e-1, 1e0, 1e1, 1e2], 
                 'logisticregression__max_iter': [10e5]
                 } 
    # prepare gridsearch
    grid = GridSearchCV(pipe, param_grid=l2_param_grid,cv=kf, return_train_score = True,n_jobs=-1)
    # do kfold CV on _other
    grid.fit(X_other, y_other)
    feature_names = cont_ftrs + \
                list(grid.best_estimator_[0].named_transformers_['cat'][0].get_feature_names(cat_ftrs))
    return grid, np.array(feature_names), X_test, y_test

grid, feature_names, X_test, y_test = ML_pipeline_kfold_LR2(XXX,yyy,42,5)
print('test score:',grid.score(X_test,y_test))
coefs = grid.best_estimator_[-1].coef_[0]
sorted_indcs = np.argsort(np.abs(coefs))

plt.rcParams.update({'font.size': 14})
plt.barh(np.arange(10),coefs[sorted_indcs[-10:]])
plt.yticks(np.arange(10),feature_names[sorted_indcs[-10:]])
plt.xlabel('coefficient')
plt.title('all scaled')
plt.tight_layout()
plt.savefig('../figures/LR_coefs_scaled_gt.jpg',dpi=300)
plt.show()