import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Dataset.csv')

# Display the first few rows of the dataset
print("\n")
print(df.head())

# Basic statistics for numerical columns
numerical_stats = df.describe()
print("\n")
print("Basic Statistics for Numerical Columns:")
print(numerical_stats)

# Explore the distribution of categorical variables
country_distribution = df['Country Code'].value_counts()
city_distribution = df['City'].value_counts()
cuisines_distribution = df['Cuisines'].value_counts()

print("\n")
print("\nDistribution of Country Code:")
print(country_distribution)

print("\n")
print("\nDistribution of City:")
print(city_distribution)

print("\n")
print("\nDistribution of Cuisines:")
print(cuisines_distribution)

# Identify top cuisines and cities with the highest number of restaurants
top_cuisines = df['Cuisines'].value_counts().head(10)
top_cities = df['City'].value_counts().head(10)

print("\n")
print("\nTop 10 Cuisines:")
print(top_cuisines)

print("\n")
print("\nTop 10 Cities with the Highest Number of Restaurants:")
print(top_cities)

# Distribution of categorical variables like 'Country Code', 'City', and 'Cuisines'

# Count Plot Visualization Code for Country Codes
plt.figure(figsize=(8, 5))
sns.countplot(x = df['Country Code'])
plt.xlabel('Country Codes')

plt.ylabel('Number of Restaurants')
plt.title('Distribution of Restaurants by Country Codes')
plt.show()

# Bar plot for Cuisines distribution
plt.figure(figsize=(16, 6))
sns.barplot(x=top_cuisines.index, y=top_cuisines.values, palette='Set3')
plt.title('Top 10 Cuisines')
plt.xlabel('Cuisines')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=45, ha='right')
plt.show()

# Bar plot for Top Cities
plt.figure(figsize=(14, 6))
sns.barplot(x=top_cities.index, y=top_cities.values, palette='pastel')
plt.title('Top 10 Cities with the Highest Number of Restaurants')
plt.xlabel('City')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=45, ha='right')
plt.show()