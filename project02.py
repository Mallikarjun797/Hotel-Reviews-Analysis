import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df=pd.read_csv("hotel reviews.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

#EDA

# Distribution of ratings
plt.figure(figsize=(8,5))
sns.countplot(x='Rating', data=df)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Distribution of sentiment
plt.figure(figsize=(8,5))
sns.countplot(x='Sentiment', data=df, order=['Positive', 'Neutral', 'Negative'])
plt.title('Distribution of Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Ratings by sentiment
plt.figure(figsize=(8,5))
sns.boxplot(x='Sentiment', y='Rating', data=df, order=['Positive', 'Neutral', 'Negative'])
plt.title('Ratings by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Rating')
plt.show()

# Top hotels by number of reviews
top_hotels = df['HotelName'].value_counts().head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_hotels.values, y=top_hotels.index)
plt.title('Top 10 Hotels by Number of Reviews')
plt.xlabel('Number of Reviews')
plt.ylabel('Hotel Name')
plt.show()

# Review over time
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.to_period('M')
monthly_reviews = df.groupby('Month').size()
plt.figure(figsize=(12,5))
monthly_reviews.plot()
plt.title('Number of Reviews Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Reviews')
plt.tight_layout()
plt.show()

# Sentiment by country
top_countries = df['Country'].value_counts().head(5).index
plt.figure(figsize=(10,6))
sns.countplot(x='Country', hue='Sentiment', data=df[df['Country'].isin(top_countries)],
              order=top_countries)
plt.title('Sentiment Distribution by Country (Top 5)')
plt.xlabel('Country')
plt.ylabel('Count')
plt.show()

