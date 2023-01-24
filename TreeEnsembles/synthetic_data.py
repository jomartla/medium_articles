import pandas as pd
import numpy as np
import random

RANDOM_SEED = 42

def create_data(categorical=True):
    # Creating a dictionary with the data
    dictionary = {"square_meters": [100, 50, 75, 120, 230, 175, 80, 90, 350, 275], 
                  "has_garage": [True, False, False, True, True, False, False, True, False, True], 
                  "has_garden": [False, False, True, False, True, False, False, False, True, True], 
                  "rooms": [2, 1, 2, 3, 4, 3, 3, 3, 6, 4]}

    # Creating a Pandas DataFrame from the dictionary
    dataframe = pd.DataFrame.from_dict(dictionary)

    # Let's create the output variable (price) depending on the features defined above
    # Every square_meter value is 1000
    square_meters_value = (dataframe["square_meters"] * 1000)

    # Having a garage sums a 10% to the house value
    has_garage_value = np.where(dataframe["has_garage"]==True, 1.1, 1)

    # Having a garden sums a 20% to the house value
    has_garden_value = np.where(dataframe["has_garden"]==True, 1.2, 1)

    # We introduce some noise in order to make the prediction task harder (the price can gain or loose a 50 percent of the value)
    noise_value = []
    random.seed(RANDOM_SEED)

    for i in range(0, 10):
        noise_value.append(random.uniform(0.5, 1.5))

    dataframe["price"] = round(square_meters_value * has_garage_value * has_garden_value * noise_value, 0)

    # Finally, the price is converted to a categorical feature if it is required in the parameter of the function.
    if categorical:
        conditions = [dataframe["price"] < 100_000, 
                  (dataframe["price"] >= 100_000) & (dataframe["price"] < 200_000), 
                  dataframe["price"] >= 200_000]
        choices = ["LOW", "MEDIUM", "HIGH"]
        dataframe["price"] = np.select(conditions, choices)
    
    return dataframe