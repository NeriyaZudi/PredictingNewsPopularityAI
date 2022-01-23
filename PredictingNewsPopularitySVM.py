import pandas as pd # pandas is used to load and manipulate data
import numpy as np # data manipulation
import time
from matplotlib import pyplot as plt
from sklearn.utils import resample # downsample the dataset
from sklearn.model_selection import train_test_split, cross_val_score  # split  data into training and testing sets
from sklearn import preprocessing # scale and center data
from sklearn.svm import SVC # this will make a support vector machine for classificaiton
from sklearn.model_selection import GridSearchCV # this will do cross validation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, \
    classification_report  # this creates a confusion matrix
from sklearn.decomposition import PCA # to perform PCA to plot the data


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[90m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    # Background colors:
    GREYBG = '\033[100m'
    REDBG = '\033[101m'
    GREENBG = '\033[102m'
    YELLOWBG = '\033[103m'
    BLUEBG = '\033[104m'
    PINKBG = '\033[105m'
    CYANBG = '\033[106m'


def PopularityClassification(df):
    """
    Classification by popularity parameter,
    Decision on the Popularity of a News Report by Number of Shares (1400)
    :param df: data frame on which the classification is made
    :return: data frame with a column of 0 or 1.
    That is true if the article is popular (over 1400 shares) or unpopular (below 1400 shares)
    """
    df.loc[df['shares'] >= 1400, 'shares_threshold'] = 1.0
    df.loc[df['shares'] < 1400, 'shares_threshold'] = 0.0
    return df


def DropingIrrelevantData(df):
    """
    Removing irrelevant information from the data set
    :param df: data frame on which the classification is made
    :return: New data frame with only the relevant information
    """
    df.drop('url', axis=1, inplace=True)
    df.drop('shares', axis=1, inplace=True)
    df.drop('timedelta', axis=1, inplace=True)
    return df


def Downsample(df, sample_number):
    """
    Support Vector Machines are great with small datasets, but not awesome with large ones,
    downsample both categories, Articles that are popular and articles that are not popular
    to the value we get as a parameter to the function each.
    splitting the data into two dataframes, one for  Articles that are popular
    and one for that are not popular.

    :param df: data frame on which the classification is made
    :param sample_number: The value selected for splitting
    :return: New data frames splitting to 2 categories
    """
    df_popular = df[df['shares_threshold'] == 1]
    df_no_popular = df[df['shares_threshold'] == 0]

    df_popular_downsampled = resample(df_popular,
                                      replace=False,
                                      n_samples=sample_number,
                                      random_state=42)
    df_no_popular_downsampled = resample(df_no_popular,
                                         replace=False,
                                         n_samples=sample_number,
                                         random_state=42)
    return df_popular_downsampled, df_no_popular_downsampled


def Classification():
    """
    The Radial Basis Function (RBF) that we are using with our Support Vector Machine assumes that the data are centered and scaled.
     In other words, each column should have a mean value = 0 and a standard deviation = 1.
    So we need to do this to both the training and testing datasets.
    We split the data into training and testing datasets and then scale them
    separately to avoid Data Leakage.
    built a Support Vector Machine for classification we can see how it performs on the Testing Dataset
    and draw a Confusion Matrix.

    :return: X_train, Y_train, Y_test Of SVM to Return of the parameters
    to perform Cross Validation to optimize the parameters.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
    scaler = preprocessing.StandardScaler().fit(X_train)

    clf_svm = SVC(random_state=0, C=100, kernel='rbf')
    clf_svm.fit(X_train, Y_train)
    predictions = clf_svm.predict(X_test)

    cm = confusion_matrix(Y_test, predictions, labels=clf_svm.classes_)
    ac = accuracy_score(Y_test, predictions)
    # Plot non-normalized confusion matrix
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            clf_svm,
            X_test,
            Y_test,
            display_labels=["Not Popular", "Popular"],
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)

        print(title)
        print(bcolors.OKBLUE, disp.confusion_matrix, bcolors.ENDC)
        if title == "Normalized confusion matrix":
            print(bcolors.FAIL, "Not popular classification", round(disp.confusion_matrix[0][0], 3)*100,
                  " %", bcolors.ENDC)
            print(bcolors.OKGREEN,  "Popular classification", round(disp.confusion_matrix[1][1], 3) * 100,
                  " %", bcolors.ENDC)

    print(bcolors.OKBLUE, "The percentage of accuracy is", ac * 100.00, " %", bcolors.ENDC)
    plt.show()
    return X_train, X_test, Y_train, Y_test


def CrossValidation(kernel):
    """
    Cross-validation,is any of various similar model validation techniques for assessing how the results
     of a statistical analysis will generalize to an independent data set.
     Cross-validation is a resampling method that uses different portions of the data to test
     and train a model on different iterations. It is mainly used in settings where the goal is prediction,
     and one wants to estimate how accurately a predictive model will perform in practice.
     Finding the optimal parameters for building Support Vector Machine.

    :param kernel: The kernel function with which we will perform the Cross Validation
    :return: Print the results
    """

    print(bcolors.BOLD, "~~~~~~ Cross-validation ~~~~~~ ", bcolors.ENDC)
    start = time.time()
    classifier = SVC(random_state=0, C=100, kernel=kernel)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    scores = cross_val_score(classifier, X, Y, cv=10, n_jobs=-1)
    print(bcolors.OKGREEN, classification_report(Y_test, y_pred), bcolors.ENDC)
    print(bcolors.OKBLUE, f'Kernel : {kernel}')
    print(" SVM accuracy after 10 fold CV: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2) + ", " + str(
        round(time.time() - start, 3)) + "s", bcolors.ENDC)


def ScreePlot(X):
    """
    determine how accurate the shrunken graph will be. If it's relatively accurate, than it makes sense to draw
    the 2-Dimensional graph. If not, the shrunken graph will not be very useful.
    We can determine the accuracy of the graph by drawing something called a scree plot.
    :param X: The information in the model
    :return: The model on which the training was performed
    """
    pca = PCA()
    X_train_pca = pca.fit_transform(X)

    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = [str(x) for x in range(1, len(per_var) + 1)]

    plt.bar(x=range(1, len(per_var) + 1), height=per_var)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Components')
    plt.title('Scree Plot')
    plt.show()
    return pca, X_train_pca


start = time.time()

# Reading the data set from a file
df = pd.read_csv('OnlineNewsPopularity.csv', sep=r'\s*,\s*',
                             encoding='ascii', engine='python')

# Set a threshold for multiple shares to make a decision on what is considered popular
df = PopularityClassification(df)

# Removing irrelevant information from the database
df = DropingIrrelevantData(df)

# To view the data set Run via debug
print(df)

# Splitting the data set into 2 categories to train the model on a limited number of information
# You can change the number assigned to the splitting
dfs1, dfs2 = Downsample(df, 5000)
df_downsample = pd.concat([dfs1, dfs2])

# We will use the conventional notation of X (capital X) to represent the columns of data
# that we will use to make classifications and y (lower case y) to represent the thing we want to predict.
# In this case, we want to predict Popularity of news articles.
X = df_downsample.drop('shares_threshold', axis=1).copy()
Y = df_downsample['shares_threshold'].copy()

X_train, X_test, Y_train, Y_test = Classification()

end1 = time.time()

# Performing Cross Validation to various kernel functions
CrossValidation("poly")
CrossValidation("rbf")
CrossValidation("sigmoid")

end2 = time.time()

print(bcolors.UNDERLINE, f"Runtime of the SVM model (without cross validation) is {end1 - start} seconds")
print(bcolors.UNDERLINE, f"Runtime of the SVM model model (Total) is {end2 - start} seconds")






