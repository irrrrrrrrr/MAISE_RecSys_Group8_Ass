import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

def avg_ratings(wine_data, ratings_data):
    """
    Calculate the average ratings for each wine and add it as a column 'AvgRating' in wine_data.
    """
    
    ratings = {}
    for _, row in ratings_data.iterrows():
        wine_id = row['WineID']
        if pd.notna(wine_id):
            if wine_id not in ratings:
                ratings[wine_id] = {"total": 0, "count": 0}
            ratings[wine_id]["total"] += row["Rating"]
            ratings[wine_id]["count"] += 1

    
    for index, row in wine_data.iterrows():
        wine_id = row["WineID"]
        if wine_id in ratings:
            wine_data.at[index, "AvgRating"] = ratings[wine_id]["total"] / ratings[wine_id]["count"]
        else:
            wine_data.at[index, "AvgRating"] = None  

    return wine_data


def recommend_wine_for_user(user_id, merged_data, recsys=None):
    if recsys is None:
        
        user_wines = merged_data[merged_data['UserID'] == user_id]

        if user_wines.empty:
            return f"No wines found for user {user_id}.", None

        
        best_rated_wine = user_wines.loc[user_wines['Rating'].idxmax()]

        
        if best_rated_wine['Rating'] >= 4:
            return best_rated_wine['WineID'], best_rated_wine['Rating']

        
        else:
            
            characteristics = ['Type', 'Body']

            
            different_wines = merged_data
            for char in characteristics:
                different_wines = different_wines[different_wines[char] != best_rated_wine[char]]

            
            if not different_wines.empty:
                recommended_wine = different_wines.sample().iloc[0]  
                return recommended_wine['WineID'], None

            return f"No sufficiently different wines found for user {user_id}.", None
    else:
        wine = recsys.recommend(user_id, 1)
        wine.rename(columns={'item': 'wine_id', 'score': 'rating'}, inplace=True)
        return wine.loc[0, 'wine_id'], wine.loc[0, 'rating']


def recommend_wine_for_group(group_id, group_data, merged_data):
    
    group_info = group_data[group_data['group_id'] == group_id].iloc[0]
    group_members = eval(group_info['group_members'])  

    
    recommendations = []

    
    for user_id in group_members:
        wine_id, rating = recommend_wine_for_user(user_id, merged_data)
        recommendations.append({
            'group_id': group_id,
            'user_id': user_id,
            'wine_id': wine_id,
            'rating': rating if rating is not None else 'Suggested different wine'
        })

    
    recommendation_df = pd.DataFrame(recommendations)
    return recommendation_df









































def find_knn_for_wine(wine_id, merged_data, user_id, wine_df, ratings_df, k=10):
    """
    Findet ähnliche Weine basierend auf KNN (K-Nearest Neighbors) für einen gegebenen Wein.
    Parameter:
    - wine_id: Die ID des Zielweins
    - merged_data: Der kombinierte DataFrame mit Weininformationen und Bewertungen
    - user_id: Die ID des Benutzers (wird in der aktuellen Implementierung nicht verwendet)
    - wine_df: Der DataFrame mit den Weindetails
    - ratings_df: Der DataFrame mit den Benutzerbewertungen
    - k: Anzahl der nächste Nachbarn (Default = 10)

    Rückgabe:
    - Eine Liste mit empfohlenen Weinen oder eine Fehlermeldung
    """
    features = ['Type', 'Body']  

    
    if not all([col in merged_data.columns for col in features]):
        raise ValueError(f"Die erforderlichen Spalten {features} sind nicht in 'merged_data' vorhanden. Aktuelle Spalten: {merged_data.columns.tolist()}")

    
    wine_features = merged_data[features].dropna()


    
    if wine_features.empty:
        
        raise ValueError(f"Die wine_features DataFrame ist leer. Stellen Sie sicher, dass die Spalten 'Type' und 'Body' gültige Werte enthalten. "
                         f"Aktuelle Verteilung in merged_data:\n"
                         f"Type: {merged_data['Type'].value_counts(dropna=False)}\n"
                         f"Body: {merged_data['Body'].value_counts(dropna=False)}")

    
    encoder = OneHotEncoder(sparse_output=False)

    


    
    encoded_wine_features = encoder.fit_transform(wine_features)


    
    knn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn_model.fit(encoded_wine_features)

    
    target_wine = merged_data[merged_data['WineID'] == wine_id]
    if target_wine.empty:
        return f"No wine found with WineID {wine_id}."

    
    target_wine_features = target_wine[features].dropna()


    
    if target_wine_features.empty:
        return f"No valid target wine features found for WineID {wine_id}. Überprüfen Sie die Spalten 'Type' und 'Body'."

    
    encoded_target_wine_features = encoder.transform(target_wine_features)


    
    distances, indices = knn_model.kneighbors(encoded_target_wine_features)
    recommended_wines = merged_data.iloc[indices[0]]

    
    recommended_wines = recommended_wines[recommended_wines['WineID'] != wine_id]

    
    recommended_list = recommended_wines[['WineID', 'Type', 'Body', 'Rating']].to_dict(orient='records')
    return recommended_list



def recommend_similar_wines_for_group(result_df, merged_data,wine_data,ratings_data, k=10):
    knn_recommendations = []

    
    for index, row in result_df.iterrows():
        wine_id = row['wine_id']
        user_id = row['user_id']

        
        knn_result = find_knn_for_wine(wine_id, merged_data, user_id,wine_data,ratings_data,k)

        
        knn_recommendations.append({
            'user_id': int(user_id),
            'wine_id': int(wine_id),
            'knn_recommendations': knn_result
        })
    
    
    knn_recommendation_df = pd.DataFrame(knn_recommendations)
    return knn_recommendation_df

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

    
    category_weights_sorted = {"Type": {}, "Body": {}}
    for key, value in category_weights.items():
        category_weights_sorted[key] = dict(sorted(category_weights[key].items(), key=lambda item: item[1], reverse=True))
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

def group_rec(group_id, group_data, merged_data,wine_data,ratings_data):
    result_df = recommend_wine_for_group(group_id, group_data, merged_data)
    knn_recommendation_df = recommend_similar_wines_for_group(result_df, merged_data,wine_data,ratings_data, k=10)
    category_weights_by_user = weights_user(knn_recommendation_df)
    category_weights_sorted = category_weights(category_weights_by_user)

    recommended_wines = get_wine_with_top_categories(category_weights_sorted, wine_data, top_type_index=1, top_body_index=1)
    return recommended_wines