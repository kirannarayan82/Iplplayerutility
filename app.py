# Install required packages
!pip install streamlit pandas scikit-learn

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Function to generate synthetic data
def generate_data(n_samples):
    np.random.seed(42)
    data = pd.DataFrame({
        'player_id': np.arange(1, n_samples + 1),
        'matches_played': np.random.randint(1, 50, size=n_samples),
        'runs_scored': np.random.randint(0, 1000, size=n_samples),
        'wickets_taken': np.random.randint(0, 50, size=n_samples),
        'catches': np.random.randint(0, 20, size=n_samples)
    })
    return data

# Function to calculate Player Utility Score (PUS)
def calculate_pus(data, weights):
    data['PUS'] = (weights[0] * data['matches_played'] +
                   weights[1] * data['runs_scored'] +
                   weights[2] * data['wickets_taken'] +
                   weights[3] * data['catches'])
    return data

# Generate synthetic data
n_samples = 100
data = generate_data(n_samples)

# Define weights for the Player Utility Score
weights = [0.2, 0.4, 0.3, 0.1]

# Calculate Player Utility Score
data = calculate_pus(data, weights)

# Player Segmentation using K-Means
X = data[['matches_played', 'runs_scored', 'wickets_taken', 'catches']]
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
data['cluster'] = kmeans.labels_

# Train a model to predict Player Utility Score
X_train = data[['matches_played', 'runs_scored', 'wickets_taken', 'catches']]
y_train = data['PUS']
model = LinearRegression().fit(X_train, y_train)

# Streamlit app
st.title("IPL Player Segmentation and Utility Model")

# Display data
st.header("Player Data")
st.write(data)

# Player Segmentation Visualization
st.header("Player Segmentation")
plt.figure(figsize=(10, 6))
plt.scatter(data['runs_scored'], data['wickets_taken'], c=data['cluster'], cmap='viridis')
plt.xlabel('Runs Scored')
plt.ylabel('Wickets Taken')
plt.title('Player Segmentation')
st.pyplot(plt)

# Predict Player Utility Score for a new player
st.header("Predict Player Utility Score")
matches_played = st.number_input('Matches Played', min_value=1, max_value=50)
runs_scored = st.number_input('Runs Scored', min_value=0, max_value=1000)
wickets_taken = st.number_input('Wickets Taken', min_value=0, max_value=50)
catches = st.number_input('Catches', min_value=0, max_value=20)

new_player = np.array([[matches_played, runs_scored, wickets_taken, catches]])
predicted_pus = model.predict(new_player)

st.write(f"Predicted Player Utility Score: {predicted_pus[0]:.2f}")

