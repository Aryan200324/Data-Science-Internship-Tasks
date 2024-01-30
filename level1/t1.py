import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Explore the dataset and identify the number of rows and columns.
file_path = 'D:\internship\cognifyz\Dataset.csv'  # Replace with the actual file path
data = pd.read_csv(file_path)

# Display the number of rows and columns
print(f"Number of rows: {data.shape[0]}")
print(f"Number of columns: {data.shape[1]}")

# Task 2: Check for missing values in each column and handle them accordingly.
print(data.notnull())
missing_values = data.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Handle missing values (replace or drop based on your analysis)
# For example, if you decide to drop rows with missing values:
data = data.dropna()

# Task 3: Perform data type conversion if necessary.
# For example, if 'Column_name' needs to be converted to numeric:
data['Price range'] = pd.to_numeric(data['Price range'], errors='coerce')

# Task 4: Analyze the distribution of the target variable ("Aggregate rating") and identify any class imbalances.
plt.figure(figsize=(8, 6))
data = data.astype({'Aggregate rating':'int'})
sns.distplot(data['Aggregate rating'] , bins=10)
plt.show()
#here we check class imbalances if any
plt.figure(figsize=(10, 6))
sns.countplot(data)
plt.show()
