import streamlit as st
import pandas as pd
import ast
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

st.title("Group Wine Recommendation System")

# Load the datasets
try:
    st.write("Loading datasets...")
    ratings_data = pd.read_csv("Dataset/last/XWines_Slim_150K_ratings.csv")
    wine_data = pd.read_csv("Dataset/last/XWines_Slim_1K_wines.csv")
    group_data = pd.read_csv("Dataset/last/group_composition.csv")
    st.write("Datasets loaded successfully!")
except FileNotFoundError as e:
    st.error(f"Error: {e}. Please check the dataset paths.")
    st.stop()

# Merge the two datasets on 'WineID'
merged_data = pd.merge(ratings_data, wine_data, on='WineID')

# Function to extract group members
def extract_group_members(group_df, group_id):
    group_row = group_df[group_df['group_id'] == group_id]
    if group_row.empty:
        return None
    group_members_str = group_row['group_members'].values[0]
    group_members = ast.literal_eval(group_members_str)
    return group_members

# Function to filter ratings for group members
def get_group_ratings(group_members, ratings_df):
    return ratings_df[ratings_df['UserID'].isin(group_members)]

# Function to calculate group preferences
def calculate_group_preferences(group_ratings, merged_data):
    top_wines = group_ratings.groupby('WineID')['Rating'].mean().reset_index()
    wine_details = merged_data[merged_data['WineID'].isin(top_wines['WineID'])]
    category_weights = {
        "Type": wine_details['Type'].value_counts().to_dict(),
        "Body": wine_details['Body'].value_counts().to_dict()
    }
    return category_weights

# KNN modeling for similar wines
def find_knn_recommendations(group_ratings, merged_data, k=10):
    features = ['Type', 'Body']
    wine_features = merged_data[features]
    encoder = OneHotEncoder(sparse_output=False)
    encoded_wine_features = encoder.fit_transform(wine_features)
    knn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn_model.fit(encoded_wine_features)

    group_top_wines = group_ratings['WineID'].unique()
    similar_wines = []
    for wine_id in group_top_wines:
        target_wine = merged_data[merged_data['WineID'] == wine_id][features]
        if target_wine.empty:
            continue
        encoded_target_wine = encoder.transform(target_wine)
        distances, indices = knn_model.kneighbors(encoded_target_wine)
        similar_wines.append(merged_data.iloc[indices[0]][['WineID', 'Type', 'Body']])

    return pd.concat(similar_wines).drop_duplicates()

# Function to select the best wine based on preferences
def get_best_wine_based_on_preferences(category_weights, merged_data, group_ratings, n):
    sorted_type = sorted(category_weights['Type'].items(), key=lambda x: x[1], reverse=True)[0][0]
    sorted_body = sorted(category_weights['Body'].items(), key=lambda x: x[1], reverse=True)[0][0]
    
    # Calculate the average rating for each wine in the group if 'AvgRating' doesn't exist
    if 'AvgRating' not in merged_data.columns:
        avg_ratings = group_ratings.groupby('WineID')['Rating'].mean().reset_index()
        avg_ratings.columns = ['WineID', 'AvgRating']
        merged_data = pd.merge(merged_data, avg_ratings, on='WineID', how='left')
    
    best_wine = merged_data[(merged_data['Type'] == sorted_type) & (merged_data['Body'] == sorted_body)]
    
    if not best_wine.empty:
        return best_wine.sort_values(by='AvgRating', ascending=False).head(n)
    else:
        return None


# Function to recommend for a group
def recommend_for_group(group_id, group_data, ratings_data, merged_data, k=10, n=5):
    group_members = extract_group_members(group_data, group_id)
    if not group_members:
        return "Group not found!"
    group_ratings = get_group_ratings(group_members, ratings_data)
    category_weights = calculate_group_preferences(group_ratings, merged_data)
    knn_recommendations = find_knn_recommendations(group_ratings, merged_data, k)
    best_wine = get_best_wine_based_on_preferences(category_weights, merged_data, group_ratings, n)
    return best_wine

# Interface to input Group ID and show recommendations
st.write("Enter a Group ID to get wine recommendations for the group.")
group_id = st.number_input('Enter Group ID:', min_value=0, max_value=239, value=0)

if st.button("Get Recommendations"):
    recommended_wines = recommend_for_group(group_id, group_data, ratings_data, merged_data, k=10, n=5)
    
    if isinstance(recommended_wines, str):
        st.write(recommended_wines)
    else:
        st.write(f"Recommended Wines for Group ID {group_id}:")
        if recommended_wines is None or recommended_wines.empty:
            st.write("No wine recommendations found.")
        else:
            st.dataframe(recommended_wines[['WineID', 'Type', 'Body', 'AvgRating']])
