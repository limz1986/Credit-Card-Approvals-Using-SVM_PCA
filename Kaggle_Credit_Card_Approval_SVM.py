

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from sklearn.utils import resample 
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing 
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import plot_confusion_matrix 
from sklearn.decomposition import PCA 
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, brier_score_loss

df = pd.read_csv (r'C:/Users/65904/Desktop/Machine Learning/Datasets/Credit Card_Approval_dataset.csv')

#EDA
df.dtypes
df.describe()
df['CreditScore'].unique()

#Outlier/Cleaning
credit_outlier_filter = df['CreditScore'] < 0
df.loc[credit_outlier_filter,'CreditScore'] =  df.loc[credit_outlier_filter,'CreditScore'] * -1

for col in ['Income','Age','CreditScore']:
    df.boxplot(column=[col])
    plt.show()

#Dealing with missing data (if needed)

# len(df.loc[(df['XXX'] == 0) | (df['XXX'] == 0)])
# len(df)

# df_no_missing = df.loc[(df['XXX'] != 0) & (df['XXX'] != 0)]
# len(df_no_missing)


#Resampling of data for large datasets
# df_no_default = df_no_missing[df_no_missing['DEFAULT'] == 0]
# df_default = df_no_missing[df_no_missing['DEFAULT'] == 1]



# df_no_default_downsampled = resample(df_no_default,
#                                   replace=False,
#                                   n_samples=1000,
#                                   random_state=42)
# len(df_no_default_downsampled)


# df_default_downsampled = resample(df_default,
#                                   replace=False,
#                                   n_samples=1000,
#                                   random_state=42)
# len(df_default_downsampled)
# df_downsample = pd.concat([df_no_default_downsampled, df_default_downsampled])
# len(df_downsample)


#X,y Split
X = df.drop('Approved', axis=1).copy() 
X.head() 
y = df['Approved'].copy()
y.head()


#One hot encoding 
X_encoded = pd.get_dummies(X, columns=['Industry',
                                       'Ethnicity',
                                       'Citizen'
                                       ])
X_encoded.head()
X_encoded.dtypes

# X,y train test split + Stratification 
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42, stratify = y)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_train_df = pd.DataFrame(X_train_scaled, columns = X_train.columns, index = X_train.index)

#standardizing the out-of-sample data
X_test_scaled = scaler.transform(X_test)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns = X_test.columns, index = X_test.index)


# # Build A Preliminary Support Vector Machine
clf_svm = SVC(random_state=42)
clf_svm.fit(X_train_scaled, y_train)

plot_confusion_matrix(clf_svm, 
                      X_test_scaled, 
                      y_test,
                      values_format='d',
                      display_labels=["Not Approved", "Approved"])


#take note when using kernel as rbf you are required to provide gamma values




# Using  `GridSearchCV()`. 
param_grid = [
  {'C': [0.5, 1, 10, 100], # NOTE: Values for C must be > 0
   'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001], 
   'kernel': ['rbf']},
]


optimal_params = GridSearchCV(
        SVC(), 
        param_grid,
        cv=5,
        scoring='accuracy', 
        verbose=0 
    )

optimal_params.fit(X_train_scaled, y_train)
print(optimal_params.best_params_)


clf_svm = SVC(random_state=42, C=1, gamma= 'scale')
clf_svm.fit(X_train_scaled, y_train)

plot_confusion_matrix(clf_svm, 
                      X_test_scaled, 
                      y_test,
                      values_format='d',
                      display_labels=["Not Approved", "Approved"])


# no of columns in the df
len(df.columns)

#PCA
pca = PCA() 
X_train_pca = pca.fit_transform(X_train_scaled)

per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = [str(x) for x in range(1, len(per_var)+1)]
 
plt.bar(x=range(1,len(per_var)+1), height=per_var)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Components')
plt.title('Screen Plot')
plt.show()


# The screen plot shows that the first principal component, PC1, accounts for a relatively large amount of variation in the raw data, 
# and this means that it will not be good candidate for the x-axis in the 2-dimensional graph. 
# However, PC2 is not much different from PC3 or PC4, which doesn't bode well for dimension reduction. 
# Now we will draw the PCA graph. 

train_pc1_coords = X_train_pca[:, 0] 
train_pc2_coords = X_train_pca[:, 1]

## NOTE:
## pc1 contains the x-axis coordinates of the data after PCA
## pc2 contains the y-axis coordinates of the data after PCA

## Scaling
pca_train_scaled = preprocessing.scale(np.column_stack((train_pc1_coords, train_pc2_coords)))

## Now we optimize the SVM fit to the x and y-axis coordinates
## of the data after PCA dimension reduction...

# Applying Grid Search to find the best model and the best parameters

parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 
               'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]



param_grid = [
  {'C': [1, 10, 100, 1000], 
   'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001], 
   'kernel': ['rbf']},
]

optimal_params = GridSearchCV(
        SVC(), 
        param_grid,
        cv=5,
        scoring='accuracy', 
        verbose=0 
    )

optimal_params.fit(pca_train_scaled, y_train)
print(optimal_params.best_params_)

optimal_params = GridSearchCV(
        SVC(), 
        parameters,
        cv=5,
        scoring='accuracy', 
        verbose=0 
    )

optimal_params.fit(pca_train_scaled, y_train)
print(optimal_params.best_params_)



#Calculating Accuracy and Recall
clf_svm = SVC( kernel='rbf', random_state=42, C=1000, gamma=0.01, probability=True)
classifier = clf_svm.fit(X_train_df, y_train)


predicted = classifier.predict(X_test_scaled_df) 
prob_default = classifier.predict_proba(X_test_scaled_df)
prob_default = [x[1] for x in prob_default] 

print("accuracy:", accuracy_score(y_test, predicted))
print("balanced_accuracy:", balanced_accuracy_score(y_test, predicted))
print("recall:", recall_score(y_test, predicted))
print("brier_score_loss:", brier_score_loss(y_test, prob_default))

# Param_grid
# accuracy: 0.7976878612716763
# balanced_accuracy: 0.7958603896103896
# recall: 0.7792207792207793
# brier_score_loss: 0.15136601316497791




clf_svm = SVC( kernel='rbf', random_state=42, C=1000, gamma=0.01)
classifier = clf_svm.fit(pca_train_scaled, y_train)

## Transform the test dataset with the PCA...
X_test_pca = pca.transform(X_train_scaled)
#X_test_pca = pca.transform(X_test_scaled)
test_pc1_coords = X_test_pca[:, 0] 
test_pc2_coords = X_test_pca[:, 1]

x_min = test_pc1_coords.min() - 1
x_max = test_pc1_coords.max() + 1

y_min = test_pc2_coords.min() - 1
y_max = test_pc2_coords.max() + 1

xx, yy = np.meshgrid(np.arange(start=x_min, stop=x_max, step=0.01),
                     np.arange(start=y_min, stop=y_max, step=0.01))


Z = clf_svm.predict(np.column_stack((xx.ravel(), yy.ravel())))

Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10,10))

ax.contourf(xx, yy, Z, alpha=0.1)

## now create custom colors for the actual data points
cmap = colors.ListedColormap(['#e41a1c', '#4daf4a'])
scatter = ax.scatter(test_pc1_coords, test_pc2_coords, c=y_train, 
               cmap=cmap, 
               s=100, 
               edgecolors='k', ## 'k' = black
               alpha=0.7)

## now create a legend
legend = ax.legend(scatter.legend_elements()[0], 
                   scatter.legend_elements()[1],
                    loc="upper right")
legend.get_texts()[0].set_text("Not Approved")
legend.get_texts()[1].set_text("Approved")

## now add axis labels and titles
ax.set_ylabel('PC2')
ax.set_xlabel('PC1')
ax.set_title('Decison surface using the PCA transformed/projected features')
# plt.savefig('svm_default.png')
plt.show()

