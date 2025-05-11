

# Smart Recommendation System Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Environment Setup](#environment-setup)
3. [Dataset Description](#dataset-description)
4. [Data Preprocessing](#data-preprocessing)
5. [Creating User-Item Matrix](#creating-user-item-matrix)
6. [Similarity Calculation](#similarity-calculation)
7. [Recommendation Function](#recommendation-function)
8. [Visualization](#visualization)
9. [Testing and Validation](#testing-and-validation)
10. [Documentation and Submission](#documentation-and-submission)
11. [Conclusion](#conclusion)

---

## 1. Project Overview
**Objective**:  
The goal of this project is to create a recommendation system that suggests movies to users based on their preferences using the MovieLens 20M dataset.

**Types of Recommendations**:
- Collaborative Filtering
- Content-Based Filtering
- Hybrid Approaches

---

## 2. Environment Setup
**Required Libraries**:
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `scikit-learn`: For machine learning algorithms.
- `matplotlib` and `seaborn`: For data visualization.

**Installation**:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 3. Dataset Description
**MovieLens 20M Dataset**:
- Contains 20 million ratings from users on movies.
- Includes two main files:
  - `ratings.csv`: Contains user ratings for movies.
  - `movies.csv`: Contains movie titles and genres.
- Each user has rated at least 20 movies.

---

## 4. Data Preprocessing
**Loading Data**:
```python
import pandas as pd

# Load the ratings and movies data
ratings = pd.read_csv('path_to_your_dataset/ratings.csv')
movies = pd.read_csv('path_to_your_dataset/movies.csv')

# Display the first few rows of the datasets
print(ratings.head())
print(movies.head())
```

**Merging Datasets**:
```python
# Merge ratings with movie titles
data = pd.merge(ratings, movies, on='movieId')
```

**Cleaning Data**:
```python
# Check for missing values
print(data.isnull().sum())

# Drop any rows with missing values
data.dropna(inplace=True)
```

---

## 5. Creating User-Item Matrix
**Pivot Table**:
```python
# Create a user-item matrix
user_item_matrix = data.pivot_table(index='userId', columns='title', values='rating')
```

**Fill Missing Values**:
```python
# Fill NaN values with 0
user_item_matrix.fillna(0, inplace=True)
```

---

## 6. Similarity Calculation
**Cosine Similarity**:
```python
from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
```

---

## 7. Recommendation Function
**Functionality**:
```python
def get_recommendations(user_id, num_recommendations=5):
    # Get similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:num_recommendations + 1]
    
    # Get movie recommendations based on similar users
    recommended_movies = user_item_matrix.loc[similar_users].mean(axis=0).sort_values(ascending=False)
    
    # Return top recommended movies
    return recommended_movies.head(num_recommendations)

# Example usage
recommendations = get_recommendations(user_id=1, num_recommendations=5)
print(recommendations)
```

---

## 8. Visualization
**Plotting Recommendations**:
```python
import matplotlib.pyplot as plt

def plot_recommendations(recommendations):
    recommendations.plot(kind='bar', figsize=(10, 5))
    plt.title('Top Movie Recommendations')
    plt.xlabel('Movies')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45)
    plt.show()

# Plot the recommendations
plot_recommendations(recommendations)
```

---

## 9. Testing and Validation
**Testing the Model**:
- Validate the recommendation system by testing with different user IDs.
- Ensure the recommendations are relevant and diverse.

---

