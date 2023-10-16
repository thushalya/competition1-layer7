# %% [markdown]
# Import necssary libraries and modules

# %%
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score,mean_squared_error, r2_score

# %% [markdown]
# Import training, validation and testing datasets

# %%
# Specify the file paths for the datasets
train_path = '/kaggle/input/audiomintest-layer7-dataset/train.csv'
valid_path = '/kaggle/input/audiomintest-layer7-dataset/valid.csv'
test_path = '/kaggle/input/audiomintest-layer7-dataset/test.csv'
# Load the training dataset
train_data = pd.read_csv(train_path)
# Load the validation dataset
valid_data = pd.read_csv(valid_path)
# Load the testing dataset
test_data = pd.read_csv(test_path)


# %% [markdown]
# Visualize original training data

# %%
train_data.head()

# %% [markdown]
# Process the data to remove null values for labels and estimate missing values in features

# %% [markdown]
# Drop the columns where there are null values for the lables in the training dataset

# %%
# Check for null values in train dataset
train_null_counts = train_data.isnull().sum()
print("train null counts : \n {}".format(train_null_counts))

# Drop rows with null values in the final four columns (target labels) for train dataset
train_data = train_data.dropna(subset=train_data.columns[-4:], how='any')

# %% [markdown]
# Fill the null values in the features with their means in the train, valid and test datasets.

# %%
# Fill null values with mean in train dataset
train_data = train_data.fillna(train_data.mean())

# Fill null values with mean in valid dataset
valid_data = valid_data.fillna(valid_data.mean())

# Fill null values with mean in test dataset
test_data = test_data.fillna(test_data.mean())

# %% [markdown]
# Visualize processed training data

# %%
train_data.head()

# %% [markdown]
# Separate features and labels in the train, valid and test datasets

# %%
# Separate features and labels in train dataset
train_features = train_data.iloc[:, :-4]
train_labels = train_data.iloc[:, -4:]

# Separate features and labels in valid dataset
valid_features = valid_data.iloc[:, :-4]
valid_labels = valid_data.iloc[:, -4:]

columns_to_remove = ['ID']
# Create a new DataFrame to hold the dropped column
saved_id_column = test_data[columns_to_remove].copy()

test_features = test_data.drop(columns=columns_to_remove)

# %% [markdown]
# Extract the second label in the train, valid and test datasets

# %%
# get the second label of the train dataset
train_label2 = train_labels.iloc[:,1]

# get the second label of the valid dataset
valid_label2 = valid_labels.iloc[:,1]

# # get the second label of the test dataset
# test_label2 = test_labels.iloc[:,1]

# %% [markdown]
# # Making predictions for Label 2 performing feature engineering.

# %% [markdown]
# Predict label 2 with feature engineering steps and techniques

# %% [markdown]
# ## Feature Engineering

# %% [markdown]
# Use feature selection based on correlation matrix and feature extraction based on PCA

# %% [markdown]
# ### Feature Selection

# %% [markdown]
# Visualize the distribution of the training label 2

# %%
# Plotting the distribution of train_label2
labels, counts = np.unique(train_label2, return_counts=True)

plt.figure(figsize=(10, 6))
plt.xticks(labels)
plt.bar(labels, counts,  color='lightcoral')
plt.xlabel('Target Label 2')
plt.ylabel('Frequency')
plt.title('Distribution of Target Label 2')
plt.show()

# %% [markdown]
# Calculate the correlation matrix of the training data features

# %%
#Calculate the correlation matrix
correlation_matrix = train_features.corr()

mask = np.triu(np.ones_like(correlation_matrix))

# Create a heatmap of the correlation matrix using seaborn
plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, cmap='gray', center=0, mask=mask)
plt.title("Correlation Matrix")
plt.show()

# %% [markdown]
# Identify the features that are highly correlated with each other using the traning dataset

# %%
# Set the threshold for correlation
correlation_threshold = 0.9

highly_correlated = set()

# Find highly correlated features
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname = correlation_matrix.columns[i]
            highly_correlated.add(colname)

print(highly_correlated)

# %% [markdown]
# Remove the previously identified highly correlated features from all the datasets

# %%
# Remove highly correlated features
train_features = train_features.drop(columns=highly_correlated)
valid_features = valid_features.drop(columns=highly_correlated)
test_features = test_features.drop(columns=highly_correlated)

# %% [markdown]
# Display the resulting feature shapes of the datasets

# %%
# Display the filtered train feature count
print("Filtered train features: {}".format(train_features.shape))

# Display the filtered valid feature count
print("Filtered valid features: {}".format(valid_features.shape))

# Display the filtered test feature count
print("Filtered test features: {}".format(test_features.shape))

# %% [markdown]
# Identify the features that are highly correlated with the label using the traning dataset

# %%
# Calculate the correlation matrix between features and train_label2
correlation_with_target = train_features.corrwith(train_label2)

# Set the correlation threshold
correlation_threshold = 0.05

# Select features that meet the correlation threshold
highly_correlated_features = correlation_with_target[correlation_with_target.abs() > correlation_threshold]

print(highly_correlated_features)

# %% [markdown]
# Extract the features that are only highly correlated with the label from all datasets

# %%
# # Drop the features with low correlated in train data
# train_features = train_features[highly_correlated_features.index]

# # Drop the features with low correlated in valid data
# valid_features = valid_features[highly_correlated_features.index]

# # Drop the features with low correlated in test data
# test_features = test_features[highly_correlated_features.index]

# %% [markdown]
# Display the resulting feature shapes of the datasets

# %%
# Display the filtered train feature count
print("Filtered train features: {}".format(train_features.shape))

# Display the filtered valid feature count
print("Filtered valid features: {}".format(valid_features.shape))

# Display the filtered test feature count
print("Filtered test features: {}".format(test_features.shape))

# %% [markdown]
# Standardize the features of all datasets

# %%
# Standardize the features
scaler = RobustScaler()
standardized_train_features = scaler.fit_transform(train_features)
standardized_valid_features = scaler.transform(valid_features)
standardized_test_features = scaler.transform(test_features)

# %% [markdown]
# ### Feature Extraction

# %% [markdown]
# Extract can combine the features that are highly significant in predicting the label using Principal Componenet Analysis(PCA)
# 
# Extract the features that can explain the variance of the label to 99%
# 
# Display the resulting explained variances of each principal component

# %%
variance_threshold = 0.99

# Apply PCA with the determined number of components
pca = PCA(n_components=variance_threshold, svd_solver='full')

pca_train_result = pca.fit_transform(standardized_train_features)
pca_valid_result = pca.transform(standardized_valid_features)
pca_test_result = pca.transform(standardized_test_features)

# Explained variance ratio after dimensionality reduction
explained_variance_ratio_reduced = pca.explained_variance_ratio_
print("Explained Variance Ratio after Dimensionality Reduction:", explained_variance_ratio_reduced)

# Plot explained variance ratio
plt.figure(figsize=(18, 10))
plt.bar(range(1, pca_train_result.shape[1] + 1), explained_variance_ratio_reduced, color='lightcoral')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio per Principal Component (Reduced)')
plt.show()

# Display the reduced train feature matrix
print("Reduced Train feature matrix shape: {}".format(pca_train_result.shape))
# Display the reduced valid feature matrix
print("Reduced valid feature matrix shape: {}".format(pca_valid_result.shape))
# Display the reduced test feature matrix
print("Reduced test feature matrix shape: {}".format(pca_test_result.shape))

# %% [markdown]
# ## Model Selection

# %%
def calc_accuracy_score(model, X_train, y_train, X_val, y_val, verbose = False):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    if verbose:
        print(classification_report(y_val, y_pred))

    return accuracy_score(y_val, y_pred)

# %% [markdown]
# Kneighborsclassifier

# %%
calc_accuracy_score(KNeighborsClassifier(),pca_train_result,train_label1,pca_valid_result,valid_label1)

# %% [markdown]
# CatBoostClassfier

# %%
calc_accuracy_score(CatBoostClassifier(loss_function='MultiClass'
                       , learning_rate = 0.15),pca_train_result,train_label1,pca_valid_result,valid_label1)

# %%
from sklearn.model_selection import GridSearchCV

leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
#Use GridSearch
clf = GridSearchCV(KNeighborsClassifier(), hyperparameters, cv=10)
best_model = clf.fit(X_train_pca,y_train['label_2'])
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

best_model.best_estimator_

# %% [markdown]
# Select the best model. hyperparameter tuning gives SVC(C=1000)

# %%
best_model_label_2 = SVC(C=1000)
pred_label2 = best_model_label_2.fit(pca_train_result, train_label2).predict(pca_test_result)
pred = best_model_label_2.predict(pca_valid_result)
accuracy_score(valid_label2, pred )


# %% [markdown]
# # Generate Output CSV

# %% [markdown]
# Define method to create the csv file

# %%

def create_csv(pred):

  df = pd.DataFrame()

  df.insert(loc=0, column='ID', value=saved_id_column)
  df.insert(loc=1, column='Label 2', value=pred)
 
  df.to_csv('/kaggle/working/190676J_label_2.csv', index=False)

# %% [markdown]
# Create CSV file

# %%

create_csv(pred_label2)


