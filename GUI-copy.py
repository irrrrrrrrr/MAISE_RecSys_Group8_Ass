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




# Function to find the best-rated wine or suggest a completely different one if rating < 4
def recommend_wine_for_user(user_id, merged_data):
    # Filter wines rated by the specific user
    user_wines = merged_data[merged_data['UserID'] == user_id]

    if user_wines.empty:
        return f"No wines found for user {user_id}.", None

    # Find the wine with the highest rating by the user
    best_rated_wine = user_wines.loc[user_wines['Rating'].idxmax()]

    # If the best rating is 4 or higher, return that wine
    if best_rated_wine['Rating'] >= 4:
        return best_rated_wine['WineID'], best_rated_wine['Rating']

    # If no wine has a rating of 4 or higher, find a completely different wine
    else:
        # Define characteristics to consider (adjust based on your dataset)
        characteristics = ['Type', 'Body']

        # Filter out wines that are similar to the one the user rated poorly
        different_wines = merged_data
        for char in characteristics:
            different_wines = different_wines[different_wines[char] != best_rated_wine[char]]

        # If there are still wines left, choose one randomly or based on rating
        if not different_wines.empty:
            recommended_wine = different_wines.sample().iloc[0]  # Sample one random different wine
            return recommended_wine['WineID'], None

        return f"No sufficiently different wines found for user {user_id}.", None

# Function to recommend wine for a group and output in a DataFrame
def recommend_wine_for_group(group_id, group_data, merged_data):
    # Get the group members from the group data
    group_info = group_data[group_data['group_id'] == group_id].iloc[0]
    group_members = eval(group_info['group_members'])  # Assuming group_members is a list stored as a string

    # Create a list to store each user's recommendation
    recommendations = []

    # Loop through each member of the group and get their favorite wine
    for user_id in group_members:
        wine_id, rating = recommend_wine_for_user(user_id, merged_data)
        recommendations.append({
            'group_id': group_id,
            'user_id': user_id,
            'wine_id': wine_id,
            'rating': rating if rating is not None else 'Suggested different wine'
        })

    # Convert the recommendations to a DataFrame
    recommendation_df = pd.DataFrame(recommendations)
    return recommendation_df




# Load the ratings dataset (replace 'ratings.csv' with your actual file path)
ratings_data = pd.read_csv('Dataset/last/XWines_Slim_150K_ratings.csv')

# Group by 'user_id' and find the maximum rating for each user
user_max_ratings = ratings_data.groupby('UserID')['Rating'].max()

# Filter to get only users whose max rating was 3
users_with_max_rating_3 = user_max_ratings[user_max_ratings == 3].index



# create average rating for each wine

ratings = {}
for index, row in ratings_data.iterrows():
    if row['WineID'] not in ratings:
        ratings[row['WineID']] = {"total": row['Rating'], "count": 1}
    else:
        ratings[row['WineID']]["total"] += row['Rating']
        ratings[row['WineID']]["count"] += 1

# add the averages to the df
wine_data["AvgRating"] = 0.0

for index, row in wine_data.iterrows():
    wine_data.loc[index, "AvgRating"] = ratings[row["WineID"]]["total"]/ratings[row["WineID"]]["count"]


from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder


# Function to find the top-rated wine by a user and recommend similar wines using KNN
def recommend_similar_wines(user_id, merged_data, k=10):
    # Filter the wines rated by the specific user
    user_wines = merged_data[merged_data['UserID'] == user_id]

    if user_wines.empty:
        return f"No wines found for user {user_id}."

    # Find the top-rated wine for this user
    top_rated_wine = user_wines.loc[user_wines['Rating'].idxmax()]

    # Extract relevant features (e.g., 'Type', 'Body') for KNN
    features = ['Type', 'Body']  # Adjust based on your dataset

    # Prepare the dataset for KNN (focus on wine characteristics, exclude 'WineID')
    wine_features = merged_data[features]

    # Apply OneHotEncoding to handle categorical variables like 'Type' and 'Body'
    encoder = OneHotEncoder(sparse_output=False)
    encoded_wine_features = encoder.fit_transform(wine_features)

    # Fit KNN model
    knn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn_model.fit(encoded_wine_features)

    # Encode the top-rated wine's features (ensure it is passed as a DataFrame with feature names)
    top_rated_wine_features = pd.DataFrame([top_rated_wine[features]], columns=features)
    encoded_top_rated_wine_features = encoder.transform(top_rated_wine_features)

    # Find K nearest wines to the top-rated wine
    distances, indices = knn_model.kneighbors(encoded_top_rated_wine_features)

    # Get the recommended similar wines (excluding the top-rated wine itself)
    recommended_wines = merged_data.iloc[indices[0]]

    # Exclude the top-rated wine itself from the recommendations
    recommended_wines = recommended_wines[recommended_wines['WineID'] != top_rated_wine['WineID']]

    return recommended_wines[['WineID', 'Type', 'Body']]


# Example usage for a specific user (replace with an actual user_id from your dataset)
  # Replace with the actual user_id you want to check

# Find the best wine for the user

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

# Function to find similar wines using KNN for a given wine
def find_knn_for_wine(wine_id, merged_data, user_id, k=10):
    # Extract relevant features (e.g., 'Type', 'Body') for KNN
    features = ['Type', 'Body']  # Adjust based on your dataset

    # Prepare the dataset for KNN (focus on wine characteristics, exclude 'WineID')
    wine_features = merged_data[features]

    # Apply OneHotEncoding to handle categorical variables like 'Type' and 'Body'
    encoder = OneHotEncoder(sparse_output=False)
    encoded_wine_features = encoder.fit_transform(wine_features)

    # Fit KNN model
    knn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn_model.fit(encoded_wine_features)

    # Get the features of the wine with the specified wine_id
    target_wine = merged_data[merged_data['WineID'] == wine_id]

    if target_wine.empty:
        return f"No wine found with WineID {wine_id}."

    # Encode the target wine's features (ensure it is passed as a DataFrame with feature names)
    target_wine_features = target_wine[features]
    encoded_target_wine_features = encoder.transform(target_wine_features)

    # Find K nearest wines to the target wine
    distances, indices = knn_model.kneighbors(encoded_target_wine_features)

    # Get the recommended similar wines (excluding the target wine itself)
    recommended_wines = merged_data.iloc[indices[0]]

    # Exclude the target wine itself from the recommendations
    recommended_wines = recommended_wines[recommended_wines['WineID'] != wine_id]

    # Return the recommended wines in a summarized format (list of dicts with WineID, Type, Body, Rating)
    recommended_list = recommended_wines[['WineID', 'Type', 'Body', 'Rating']].to_dict(orient='records')
    
    # get the wine that it eas recommended on
    original_wine = wine_data.where(wine_data['WineID'] == wine_id)
    original_wine_rating = ratings_data.where((ratings_data['WineID'] == wine_id) & (ratings_data['UserID'] == user_id)).iloc[0]['Rating']
    recommended_list.append({"WineID": wine_id, "Type": original_wine.iloc[0]["Type"], "Body": original_wine.iloc[0]["Body"], "Rating": original_wine_rating})

    return recommended_list

# Function to find KNN recommendations for all wines in the group recommendation DataFrame
def recommend_similar_wines_for_group(result_df, merged_data, k=10):
    knn_recommendations = []

    # Iterate over each wine recommended to the group
    for index, row in result_df.iterrows():
        wine_id = row['wine_id']
        user_id = row['user_id']

        # Find KNN recommendations for the wine
        knn_result = find_knn_for_wine(wine_id, merged_data, user_id, k)

        # Store the results in a list with the user_id for reference
        knn_recommendations.append({
            'user_id': int(user_id),
            'wine_id': int(wine_id),
            'knn_recommendations': knn_result
        })
    
    # Convert the recommendations into a DataFrame for easy viewing
    knn_recommendation_df = pd.DataFrame(knn_recommendations)
    return knn_recommendation_df

# Example usage (replace 'result_df' with the actual DataFrame from group recommendations)
result_df = recommend_wine_for_group(group_id=1, group_data=group_data, merged_data=merged_data)

# Find KNN recommendations for all wines in the result_df
knn_recommendation_df = recommend_similar_wines_for_group(result_df, merged_data, k=10)


def score_characteristics(recommendations):
    scores = {"Type": {}, "Body": {}}
    score = len(recommendations)
    c = 0
    for wine in recommendations:
        for key, value in wine.items():
            if key in ["WineID", "Rating"]:
                continue
            if key == "Type":
                if value not in scores["Type"]:
                    scores["Type"][value] = 1
                else:
                    scores["Type"][value] += 1
            elif key == "Body":
                if value not in scores["Body"]:
                    scores["Body"][value] = 1
                else:
                    scores["Body"][value] += 1
        c+=1
    
    return scores

def weights_user(knn_recommendation_df):
    category_weights_by_user = []
    for index, row in knn_recommendation_df.iterrows():
        x = score_characteristics(row.knn_recommendations)
        category_weights_by_user.append(x)
        return category_weights_by_user

def category_weights(category_weights_by_user):
    category_weights = {"Type": {}, "Body": {}}
    for i in category_weights_by_user:
        for key, value in i["Type"].items():
            if key not in category_weights:
                category_weights["Type"][key] = value
            else:
                category_weights["Type"][key] += value
            
        for key, value in i["Body"].items():
            if key not in category_weights:
                category_weights["Body"][key] = value
            else:
                category_weights["Body"][key] += value

    # order according to the scores
    category_weights_sorted = {"Type": {}, "Body": {}}
    for key, value in category_weights.items():
        category_weights_sorted[key] = dict(sorted(category_weights[key].items(), key=lambda item: item[1], reverse=True))
    return category_weights_sorted

# select wine with top categories of each type
def get_wine_with_top_categories(category_weights_sorted, top_type_index=1, top_body_index=1):
    top_type = list(category_weights_sorted["Type"].keys())[top_type_index-1]
    top_body = list(category_weights_sorted["Body"].keys())[top_body_index-1]
    
    # find a wine with these categories
    selection = wine_data.where((wine_data['Type'] == top_type) & (wine_data['Body'] == top_body)).dropna()
    
    if selection.shape[0] == 0:
        if top_type_index == top_body_index:
            selection = get_wine_with_top_categories(category_weights_sorted=category_weights_sorted, top_type_index=top_type_index+1, top_body_index=top_body_index)
        else:
            selection = get_wine_with_top_categories(category_weights_sorted=category_weights_sorted,top_type_index=top_type_index, top_body_index=top_body_index+1)
    
    selection.sort_values(by='AvgRating', ascending=False, inplace=True)

    return selection.head(5)


# Interface to input Group ID and show recommendations
st.write("Enter a Group ID to get wine recommendations for the group.")
group_id = st.number_input('Enter Group ID:', min_value=0, max_value=239, value=0)

if st.button("Get Recommendations"):
    result_df = recommend_wine_for_group(group_id, group_data, merged_data)
    knn_recommendation_df = recommend_similar_wines_for_group(result_df, merged_data, k=10)
    print(knn_recommendation_df)
    category_weights_by_user = weights_user(knn_recommendation_df)
    print(category_weights_by_user)
    category_weights_sorted = category_weights(category_weights_by_user)
    recommended_wines = get_wine_with_top_categories(category_weights_sorted=category_weights_sorted, top_type_index=1, top_body_index=1)
    
    if isinstance(recommended_wines, str):
        st.write(recommended_wines)
    else:
        st.write(f"Recommended Wines for Group ID {group_id}:")
        if recommended_wines is None or recommended_wines.empty:
            st.write("No wine recommendations found.")
        else:
            st.dataframe(recommended_wines[['WineID', 'Type', 'Body', 'AvgRating']])
