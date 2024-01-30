import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt



# Load the dataset
df = pd.read_csv('Dataset.csv')

# Extracting additional features from the existing columns, such as the length of the restaurant name or address
df['Restaurant Name Length'] = df['Restaurant Name'].apply(len)

# Create a new column for the length of restaurant addresses
df['Address Length'] = df['Address'].apply(len)

# Display the updated DataFrame
print(df.head())
     
# Creating new features like "Has Table Booking" or "Has Online Delivery" by encoding categorical variables
df['Has Table Booking'] = np.where(df['Has Table booking'] == 'Yes', 1, 0)
df['Has Online Delivery'] = np.where(df['Has Online delivery'] == 'Yes', 1, 0)


# Display the updated DataFrame
print(df.head())