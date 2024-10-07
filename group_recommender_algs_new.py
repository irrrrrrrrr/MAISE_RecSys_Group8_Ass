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

    
    wine_data["AvgRating"] = 0.0

    for index, row in wine_data.iterrows():
        wine_data.loc[index, "AvgRating"] = ratings[row["WineID"]]["total"]/ratings[row["WineID"]]["count"]
    
    return wine_data


def recommend_wine_for_user(user_id, merged_data, recsys=None, num_recommendations=5):
    if recsys is None:
        
        user_wines = merged_data[merged_data['UserID'] == user_id]

        if user_wines.empty:
            return f"No wines found for user {user_id}.", None

        
        top_rated_wines = user_wines.sort_values(by='Rating', ascending=False).head(num_recommendations)

        
        if top_rated_wines['Rating'].max() >= 4:
            return top_rated_wines[['WineID', 'Rating']].to_dict(orient='records')

        
        else:
            
            characteristics = ['Type', 'Body']

            
            different_wines = merged_data
            for char in characteristics:
                
                different_wines = different_wines[~different_wines[char].isin(user_wines[char])]

            
            if not different_wines.empty:
                recommended_wines = different_wines.sample(n=min(num_recommendations, len(different_wines)))
                return recommended_wines[['WineID', 'Rating']].to_dict(orient='records')

            return f"No sufficiently different wines found for user {user_id}.", None
    else:
        
        wines = recsys.recommend(user_id, num_recommendations)
        wines.rename(columns={'item': 'WineID', 'score': 'Rating'}, inplace=True)
        return wines[['WineID', 'Rating']].to_dict(orient='records')


def recommend_wine_for_group(group_id, group_data, merged_data, individual_recsys, num_recommendations=5):
    
    group_info = group_data[group_data['group_id'] == group_id].iloc[0]
    group_members = eval(group_info['group_members'])  

    
    recommendations = []

    
    for user_id in group_members:
        wine_recommendations = recommend_wine_for_user(user_id, merged_data, individual_recsys, num_recommendations)
        for wine_rec in wine_recommendations:
            recommendations.append({
                'group_id': group_id,
                'user_id': user_id,
                'wine_id': wine_rec['WineID'],
                'rating': wine_rec['Rating'] if wine_rec['Rating'] is not None else 'Suggested different wine'
            })

    
    recommendation_df = pd.DataFrame(recommendations)
    return recommendation_df


def find_knn_for_wine(wine_id, merged_data, user_id, wine_df, ratings_df, k=10):
    features = ['Type', 'Body']  

    
    wine_features = merged_data[features]

    
    encoder = OneHotEncoder(sparse_output=False)
    encoded_wine_features = encoder.fit_transform(wine_features)

    
    knn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn_model.fit(encoded_wine_features)

    
    target_wine = merged_data[merged_data['WineID'] == wine_id]

    if target_wine.empty:
        return f"No wine found with WineID {wine_id}."

    
    target_wine_features = target_wine[features]
    encoded_target_wine_features = encoder.transform(target_wine_features)

    
    distances, indices = knn_model.kneighbors(encoded_target_wine_features)

    
    recommended_wines = merged_data.iloc[indices[0]]

    
    recommended_wines = recommended_wines[recommended_wines['WineID'] != wine_id]

    
    recommended_list = recommended_wines[['WineID', 'Type', 'Body', 'Rating']].to_dict(orient='records')

    
    original_wine = wine_df.where(wine_df['WineID'] == wine_id)
    original_wine_rating = ratings_df.where((ratings_df['WineID'] == wine_id) & (ratings_df['UserID'] == user_id)).iloc[0]['Rating']
    recommended_list.append({"WineID": wine_id, "Type": original_wine.iloc[0]["Type"], "Body": original_wine.iloc[0]["Body"], "Rating": original_wine_rating})

    return recommended_list


def recommend_similar_wines_for_group(result_df, merged_data, wine_data, ratings_data, k=10):
    knn_recommendations = []

    
    for index, row in result_df.iterrows():
        wine_id = row['wine_id']
        user_id = row['user_id']

        
        knn_result = find_knn_for_wine(wine_id, merged_data, user_id, wine_data, ratings_data, k)

        
        knn_recommendations.append({
            'user_id': int(user_id),
            'wine_id': int(wine_id),
            'knn_recommendations': knn_result
        })
    
    
    knn_recommendation_df = pd.DataFrame(knn_recommendations)
    return knn_recommendation_df


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
            if key not in category_weights["Type"]:
                category_weights["Type"][key] = value
            else:
                category_weights["Type"][key] += value

        for key, value in i["Body"].items():
            if key not in category_weights["Body"]:
                category_weights["Body"][key] = value
            else:
                category_weights["Body"][key] += value

    
    category_weights_sorted = {
        "Type": dict(sorted(category_weights["Type"].items(), key=lambda item: item[1], reverse=True)),
        "Body": dict(sorted(category_weights["Body"].items(), key=lambda item: item[1], reverse=True))
    }
    return category_weights_sorted


def get_wine_with_top_categories(category_weights_sorted, wine_data, top_type_index=1, top_body_index=1):
    if top_type_index > len(category_weights_sorted["Type"]) or top_body_index > len(category_weights_sorted["Body"]):
        return pd.DataFrame()  

    top_type = list(category_weights_sorted["Type"].keys())[top_type_index - 1]
    top_body = list(category_weights_sorted["Body"].keys())[top_body_index - 1]

    selection = wine_data.loc[(wine_data['Type'] == top_type) & (wine_data['Body'] == top_body)]

    if selection.shape[0] == 0:
        if top_type_index == top_body_index:
            return get_wine_with_top_categories(category_weights_sorted, wine_data, top_type_index + 1, top_body_index)
        else:
            return get_wine_with_top_categories(category_weights_sorted, wine_data, top_type_index, top_body_index + 1)

    
    selection = selection.sort_values(by='AvgRating', ascending=False)
    return selection.head(5)


def group_rec(group_id, group_data, merged_data, wine_data, ratings_data, individual_recsys):
    result_df = recommend_wine_for_group(group_id, group_data, merged_data, individual_recsys)
    knn_recommendation_df = recommend_similar_wines_for_group(result_df, merged_data, wine_data, ratings_data, k=10)
    category_weights_by_user = weights_user(knn_recommendation_df)
    category_weights_sorted = category_weights(category_weights_by_user)

    
    recommended_wines = get_wine_with_top_categories(category_weights_sorted, wine_data, top_type_index=1, top_body_index=1)
    return recommended_wines
