"""
Statistics and Trends Assignment
Student Name: Sukesh Kumar Eddagiri
Student ID: 25036788
Dataset: data.csv (StudentsPerformance data)
"""

from corner import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

def plot_relational_plot(df):
    #Creates and saves a Relational plot.
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=df.columns[6], y=df.columns[7],hue="gender", ax=ax)

    ax.set_title('Relational Plot')
    plt.tight_layout()
    plt.savefig('relational_plot.png')
    plt.close()


def plot_categorical_plot(df):
    #Creates and saves a Categorical plot.
   
    fig, ax = plt.subplots()
    categorical_col = df.select_dtypes(include=['object']).columns[0]
    sns.countplot(data=df, x=categorical_col, ax=ax)

    ax.set_title('Categorical Plot')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('categorical_plot.png')
    plt.close()


def plot_statistical_plot(df):
    #Creates and saves a Statistical plot.
    #Correlation Heatmap

    scores_cols = ["math score", "reading score", "writing score"]
    cor = df[scores_cols].corr()

    plt.figure()
    sns.heatmap(cor, annot=True, cmap="coolwarm", fmt=".3f")
    plt.title("Correlation Heatmap of Scores")
    plt.tight_layout()
    plt.savefig('statistical_plot.png')
    plt.close()


def statistical_analysis(df, col: str):
    """
    Calculates statistical moments for a given column.
    Returns:
        tuple: mean, std deviation, skewness, excess kurtosis
    """
    data = df[col].dropna()

    mean = np.mean(data)
    stddev = np.std(data, ddof=1)
    skew = ss.skew(data)
    excess_kurtosis = ss.kurtosis(data)

    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Performs the basic preprocessing and also exploratory analysis.
    """
    print("The first 5 rows:")
    print(df.head())

    print("\nSummary of the statistics:")
    print(df.describe())

    print("\nThe Correlation matrix:")
    print(df.corr(numeric_only=True))

    # Drop missing values
    df = df.dropna()

    return df


def writing(moments, col):
    """
    Prints the interpretation of statistical results.
    """
    print(f'\nFor the attribute {col}:')
    print(f'\nMean = {moments[0]:.2f}, '
          f'\nStandard Deviation = {moments[1]:.2f}, '
          f'\nSkewness = {moments[2]:.2f}, and '
          f'\nExcess Kurtosis = {moments[3]:.2f}.')

    skew = moments[2]
    kurt = moments[3]

    if skew > 0:
        skewness_desc = "right skewed"
    elif skew < 0:
        skewness_desc = "left skewed"
    else:
        skewness_desc = "not skewed"

    if kurt > 0:
        kurtosis_desc = "leptokurtic"
    elif kurt < 0:
        kurtosis_desc = "platycurtic"
    else:
        kurtosis_desc = "mesokurtic"

    print(f'The data was {skewness_desc} and {kurtosis_desc}.')


def main():
    df = pd.read_csv('data.csv')

    df = preprocessing(df)

    # Choose a numerical column for analysis
    col = df.select_dtypes(include=np.number).columns[0]

    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)


if __name__ == '__main__':
    main()

