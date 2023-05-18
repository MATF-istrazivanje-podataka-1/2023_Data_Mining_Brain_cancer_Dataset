from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def print_classes(dim_x, dim_y, y_train, df_pca):
    colmap = {'ependymoma' : 'red',
        'glioblastoma' : 'green',
        'medulloblastoma' : 'blue',
        'pilocytic_astrocytoma' : 'yellow',
        'normal' : 'pink'
        }
   
    custom_lines = [Line2D([0], [0], color='red', lw=2),
                    Line2D([0], [0], color='green', lw=2),
                    Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='yellow', lw=2),
                    Line2D([0], [0], color='pink', lw=2)]
    names = ['ependymoma','glioblastoma','medulloblastoma','pilocytic_astrocytoma','normal']

    figure, axis = plt.subplots(dim_x,dim_y)
    figure.set_figheight(15)
    figure.set_figwidth(15)

    for x in range(dim_x):
        for y in range(1,dim_y+1):
            axis[x][y-1].set_xlabel('pca_' + str(x))
            axis[x][y-1].set_ylabel('pca_' + str(y+x))
            for i in range(len(y_train)):
                axis[x][y-1].scatter(df_pca['pca_' + str(x)][i], df_pca['pca_' + str(x+y)][i], c = colmap[y_train[i]])
            axis[x][y-1].legend(custom_lines,names, loc = 'upper left', fontsize = 'xx-small')
    plt.show()


def load_preprocess_data(path, drop, target):
    """
    Loads dataset, plots variance diagrams and returns transformed(Standardized, PCA) train and test sets

    Args:
        path : Path to the dataset CSV.
        drop : List of columns to drop from X
        target : target variable for classification

    Returns:
        X_train: Training set.
        X_test: Testing set.
        y_train: Target for train.
        y_test: Target for test.
        pca_names: List of PCA names.
        df: Pandas Dataframe of dataset
    """
        
    df = pd.read_csv(path)
    X = df.drop(drop, axis=1)
    y = df[target]

    pca = PCA()
    scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state= 11, stratify= y)

    scaler.fit(X_train)
    X_train_standard = scaler.transform(X_train)
    X_test_standard = scaler.transform(X_test)

    pca.fit(X_train_standard)
    X_train_pca = pca.transform(X_train_standard)
    X_test_pca = pca.transform(X_test_standard)
    explained_variance = pca.explained_variance_ratio_
# Print the cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance)
    pca_names = [f'pca_{i}' for i in range(cumulative_variance.shape[0])]
    #print(cumulative_variance)

    figure, axis = plt.subplots(nrows = 1,ncols = 2)
    figure.set_figheight(9)
    figure.set_figwidth(16)
    figure.set_label("PCA")

    # Plot the explained variance
    axis[0].set_xlabel('Principal Component')
    axis[0].set_ylabel('Explained Variance')
    axis[0].plot(range(len(explained_variance)),explained_variance)
    

    axis[1].plot(range(len(cumulative_variance)),cumulative_variance)
    axis[1].set_xlabel('Number of principal components')
    axis[1].set_ylabel('Explained variance')
    plt.show()

    return X_train_pca, X_test_pca, y_train, y_test, pca_names,df


def plot_class_distribution(df,target_name):
    """
    Plots class distribution from Pandas DataFrame

    Args:
        df: Pandas Dataframe of dataset.
        target_name: Target name (:

    """
    # Plot a histogram of the class distribution
    df[target_name].hist(bins=len(df[target_name].unique()))
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.show()

