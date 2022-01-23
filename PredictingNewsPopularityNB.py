
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.utils import resample


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


def PrintResult():
    """
    Drawing the confusion matrix to show the prediction results
    :return: None
    """
    # Plot non-normalized confusion matrix
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            classifier,
            X_test,
            y_test,
            display_labels=["Not Popular", "Popular"],
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)
        if title == "Normalized confusion matrix":
            print(bcolors.FAIL, "Not popular classification", round(disp.confusion_matrix[0][0], 3)*100,
                  " %", bcolors.ENDC)
            print(bcolors.OKGREEN,  "Popular classification", round(disp.confusion_matrix[1][1], 3) * 100,
                  " %", bcolors.ENDC)
    plt.show()


def CrossValidation(gnb):
    """
    Cross-validation,is any of various similar model validation techniques for assessing how the results
     of a statistical analysis will generalize to an independent data set.
     Cross-validation is a resampling method that uses different portions of the data to test
     and train a model on different iterations. It is mainly used in settings where the goal is prediction,
     and one wants to estimate how accurately a predictive model will perform in practice.
     Finding the optimal parameters for Naive Bayes Classifier.
     NOTE! There are several options for running Cross-validation to find optimal parameters
    :return: optimal parameters for building Naive Bayes Classifier.
    """
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(gnb, X_train, y_train, cv=10, scoring='accuracy')

    print(bcolors.BOLD, "~~~~~~ Cross-validation ~~~~~~ ", bcolors.ENDC)
    print('Cross-validation scores:{}'.format(scores))
    print(bcolors.FAIL, 'Average cross-validation score: {:.4f}'.format(scores.mean()), bcolors.ENDC)


start = time.time()

# Reading the data set from a file
data = pd.read_csv('OnlineNewsPopularity.csv', sep=r'\s*,\s*',
                                encoding='ascii', engine='python')

# Set a threshold for multiple shares to make a decision on what is considered popular
dataset = PopularityClassification(data)

# Removing irrelevant information from the database
dataset = DropingIrrelevantData(dataset)

# List of columns in the data set
columns = []
for c in range(len(dataset.columns)-3):
    columns.append(c)

# Splitting the data set into 2 categories to train the model on a limited number of information
# You can change the number assigned to the splitting
dfs1, dfs2 = Downsample(dataset, 5000)
df_downsample = pd.concat([dfs1, dfs2])

X = df_downsample.iloc[:, columns].values
y = df_downsample.iloc[:, -1].values
print(dataset)


# Data split to training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the Nave Base model (Gaussian)
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# predict the results
y_pred = classifier.predict(X_test)


cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

# Presenting the matrix of confusion to show the prediction results,
# and the percentage of accuracy
PrintResult()
print(bcolors.OKBLUE, cm, bcolors.ENDC)
print(bcolors.OKBLUE, "The percentage of accuracy is", ac * 100.00, " %", bcolors.ENDC)

CrossValidation(classifier)

end = time.time()

print(bcolors.UNDERLINE, f"Runtime of the Nave Base model is {end - start} seconds")
