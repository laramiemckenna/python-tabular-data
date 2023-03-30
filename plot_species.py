import argparse
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def linear_regression(dataframe, species):
    """This function performs linear regression on petal length against sepal length for a species of Iris.

    Parameters:
        dataframe (pandas.DataFrame): The DataFrame containing the Iris data.
        species (str): The species of Iris to perform linear regression on.

    Returns:
        Tuple of x, y, slope, and intercept.
    """
    species_data = dataframe[dataframe.species == species]
    x = species_data.petal_length_cm
    y = species_data.sepal_length_cm
    regression = stats.linregress(x, y)
    slope = regression.slope
    intercept = regression.intercept
    return x, y, slope, intercept

def plot_regression(x, y, slope, intercept, species):
    """Creates a scatter plot with the linear regression line for a specific species of Iris.

    Parameters:
        x (pandas.Series): petal length
        y (pandas.Series): sepal length
        slope (float): the slope of the linear regression line.
        intercept (float): y-inthe tercept of the linear regression line.
        species (str): the species of interest
    """
    plt.scatter(x, y, label = 'Data')
    plt.plot(x, slope * x + intercept, color = "orange", label = 'Fitted line')
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Sepal length (cm)")
    plt.legend()
    plt.title(species)
    plt.savefig(f"{species}_petal_v_sepal_length_regress.png")
    plt.clf() #this command is needed to prevent the graphs from being added to each other into a composite image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file")
    args = parser.parse_args()

    dataframe = pd.read_csv(args.csv_file)
    for species in dataframe.species.unique():
        x, y, slope, intercept = linear_regression(dataframe, species)
        plot_regression(x, y, slope, intercept, species)
