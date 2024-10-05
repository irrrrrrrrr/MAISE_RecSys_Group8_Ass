import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

def avg_ratings(wine_data, ratings_data):
    ratings = {}
    for index, row in ratings_data.iterrows():
        if row['WineID'] not in ratings:
            ratings[row['WineID']] = {"total": row['Rating'], "count": 1}
        else:
            ratings[row['WineID']]["total"] += row['Rating']
            ratings[row['WineID']]["count"] += 1

    # Add the averages to the df
    wine_data["AvgRating"] = 0.0

    for index, row in wine_data.iterrows():
        wine_data.loc[index, "AvgRating"] = ratings[row["WineID"]]["total"]/ratings[row["WineID"]]["count"]
    
    return wine_data

# Function to find the top 5 wines or suggest 5 completely different ones if no highly rated wines exist
def recommend_wine_for_user(user_id, merged_data, recsys=None, num_recommendations=5):
    if recsys is None:
        # Filter wines rated by the specific user
        user_wines = merged_data[merged_data['UserID'] == user_id]

        if user_wines.empty:
            return f"No wines found for user {user_id}.", None

        # Sort wines by rating, descending, and get the top 5
        top_rated_wines = user_wines.sort_values(by='Rating', ascending=False).head(num_recommendations)

        # If any of the top wines has a rating of 4 or higher, return those
        if top_rated_wines['Rating'].max() >= 4:
            return top_rated_wines[['WineID', 'Rating']].to_dict(orient='records')

        # If no wine has a rating of 4 or higher, find completely different wines
        else:
            # Define characteristics to consider (adjust based on your dataset)
            characteristics = ['Type', 'Body']

            # Filter out wines that are similar to the ones the user rated poorly
            different_wines = merged_data
            for char in characteristics:
                # Filter wines that are different in key characteristics
                different_wines = different_wines[~different_wines[char].isin(user_wines[char])]

            # If there are still wines left, sample 5 random different wines
            if not different_wines.empty:
                recommended_wines = different_wines.sample(n=min(num_recommendations, len(different_wines)))
                return recommended_wines[['WineID', 'Rating']].to_dict(orient='records')

            return f"No sufficiently different wines found for user {user_id}.", None
    else:
        # If using a recommendation system, get the top 5 recommendations
        wines = recsys.recommend(user_id, num_recommendations)
        wines.rename(columns={'item': 'WineID', 'score': 'Rating'}, inplace=True)
        return wines[['WineID', 'Rating']].to_dict(orient='records')

# Function to recommend wine for a group and output in a DataFrame
def recommend_wine_for_group(group_id, group_data, merged_data, individual_recsys, num_recommendations=5):
    # Get the group members from the group data
    group_info = group_data[group_data['group_id'] == group_id].iloc[0]
    group_members = eval(group_info['group_members'])  # Assuming group_members is a list stored as a string

    # Create a list to store each user's recommendation
    recommendations = []

    # Loop through each member of the group and get their favorite wines
    for user_id in group_members:
        wine_recommendations = recommend_wine_for_user(user_id, merged_data, individual_recsys, num_recommendations)
        for wine_rec in wine_recommendations:
            recommendations.append({
                'group_id': group_id,
                'user_id': user_id,
                'wine_id': wine_rec['WineID'],
                'rating': wine_rec['Rating'] if wine_rec['Rating'] is not None else 'Suggested different wine'
            })

    # Convert the recommendations to a DataFrame
    recommendation_df = pd.DataFrame(recommendations)
    return recommendation_df

# Function to find similar wines using KNN for a given wine
def find_knn_for_wine(wine_id, merged_data, user_id, wine_df, ratings_df, k=10):
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

    # Encode the target wine's features
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

    # Get the original wine that was recommended on
    original_wine = wine_df.where(wine_df['WineID'] == wine_id)
    original_wine_rating = ratings_df.where((ratings_df['WineID'] == wine_id) & (ratings_df['UserID'] == user_id)).iloc[0]['Rating']
    recommended_list.append({"WineID": wine_id, "Type": original_wine.iloc[0]["Type"], "Body": original_wine.iloc[0]["Body"], "Rating": original_wine_rating})

    return recommended_list

# Function to find KNN recommendations for all wines in the group recommendation DataFrame
def recommend_similar_wines_for_group(result_df, merged_data, wine_data, ratings_data, k=10):
    knn_recommendations = []

    # Iterate over each wine recommended to the group
    for index, row in result_df.iterrows():
        wine_id = row['wine_id']
        user_id = row['user_id']

        # Find KNN recommendations for the wine
        knn_result = find_knn_for_wine(wine_id, merged_data, user_id, wine_data, ratings_data, k)

        # Store the results in a list with the user_id for reference
        knn_recommendations.append({
            'user_id': int(user_id),
            'wine_id': int(wine_id),
            'knn_recommendations': knn_result
        })
    
    # Convert the recommendations into a DataFrame for easy viewing
    knn_recommendation_df = pd.DataFrame(knn_recommendations)
    return knn_recommendation_df

# Function to calculate the characteristic scores
def score_characteristics(recommendations):
    scores = {"Type": {}, "Body": {}}
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
    return scores

# Function to calculate the category weights by user
def weights_user(knn_recommendation_df):
    category_weights_by_user = []
    for index, row in knn_recommendation_df.iterrows():
        x = score_characteristics(row.knn_recommendations)
        category_weights_by_user.append(x)
    return category_weights_by_user

# Function to calculate overall category weights (continued from where it cut off)
def category_weights(category_weights_by_user):
    category_weights = {"Type": {}, "Body": {}}
    for i in category_weights_by_user:
        for key, value in i["Type"].items():
            if key not in category_weights["Type"]:
                category_weights["Type"][key] = value
            else:
                category_weights["Type"][key] += value

        for key, value in i["Body"].items():
            if key not in category_weights["Body"]:
                category_weights["Body"][key] = value
            else:
                category_weights["Body"][key] += value

    # Sort by score in descending order
    category_weights_sorted = {
        "Type": dict(sorted(category_weights["Type"].items(), key=lambda item: item[1], reverse=True)),
        "Body": dict(sorted(category_weights["Body"].items(), key=lambda item: item[1], reverse=True))
    }
    return category_weights_sorted

# Function to get wine recommendations based on top categories
def get_wine_with_top_categories(category_weights_sorted, wine_data, top_type_index=1, top_body_index=1):
    if top_type_index > len(category_weights_sorted["Type"]) or top_body_index > len(category_weights_sorted["Body"]):
        return pd.DataFrame()  # Return an empty DataFrame if no wines match

    top_type = list(category_weights_sorted["Type"].keys())[top_type_index - 1]
    top_body = list(category_weights_sorted["Body"].keys())[top_body_index - 1]

    selection = wine_data.loc[(wine_data['Type'] == top_type) & (wine_data['Body'] == top_body)]

    if selection.shape[0] == 0:
        if top_type_index == top_body_index:
            return get_wine_with_top_categories(category_weights_sorted, wine_data, top_type_index + 1, top_body_index)
        else:
            return get_wine_with_top_categories(category_weights_sorted, wine_data, top_type_index, top_body_index + 1)

    # Sort by 'AvgRating' and return the top 5 wines
    selection = selection.sort_values(by='AvgRating', ascending=False)
    return selection.head(5)

# Function to handle group wine recommendations
def group_rec(group_id, group_data, merged_data, wine_data, ratings_data, individual_recsys):
    result_df = recommend_wine_for_group(group_id, group_data, merged_data, individual_recsys)
    knn_recommendation_df = recommend_similar_wines_for_group(result_df, merged_data, wine_data, ratings_data, k=10)
    category_weights_by_user = weights_user(knn_recommendation_df)
    category_weights_sorted = category_weights(category_weights_by_user)

    # Get wines based on top categories
    recommended_wines = get_wine_with_top_categories(category_weights_sorted, wine_data, top_type_index=1, top_body_index=1)
    return recommended_wines
