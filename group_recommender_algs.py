import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Funktion zur Berechnung der durchschnittlichen Bewertungen
def avg_ratings(wine_data, ratings_data):
    ratings = {}
    for index, row in ratings_data.iterrows():
        if row['WineID'] not in ratings:
            ratings[row['WineID']] = {"total": row['Rating'], "count": 1}
        else:
            ratings[row['WineID']]["total"] += row['Rating']
            ratings[row['WineID']]["count"] += 1

    # Füge die durchschnittlichen Bewertungen dem DataFrame hinzu
    wine_data["AvgRating"] = 0.0
    for index, row in wine_data.iterrows():
        wine_data.at[index, "AvgRating"] = ratings[row["WineID"]]["total"] / ratings[row["WineID"]]["count"]
    return wine_data


# Empfehlung basierend auf Benutzerpräferenzen
def recommend_wine_for_user(user_id, merged_data, recsys=None):
    if recsys is None:
        # Filtere Weine, die vom Benutzer bewertet wurden
        user_wines = merged_data[merged_data['UserID'] == user_id]
        if user_wines.empty:
            return f"No wines found for user {user_id}.", None

        # Finde den Wein mit der höchsten Bewertung des Benutzers
        best_rated_wine = user_wines.loc[user_wines['Rating'].idxmax()]

        # Wenn die Bewertung >= 4 ist, gebe diesen Wein zurück
        if best_rated_wine['Rating'] >= 4:
            explanation = f"The wine '{best_rated_wine['WineID']}' is recommended because it is your highest-rated wine."
            return best_rated_wine['WineID'], explanation

        # Wenn kein Wein eine hohe Bewertung hat, schlage einen anderen Wein vor
        different_wines = merged_data
        characteristics = ['Type', 'Body']
        for char in characteristics:
            different_wines = different_wines[different_wines[char] != best_rated_wine[char]]
        
        if not different_wines.empty:
            recommended_wine = different_wines.sample().iloc[0]
            explanation = (f"The wine '{recommended_wine['WineID']}' is recommended because it is different from the wines you've rated low, "
                           f"with characteristics such as {recommended_wine['Type']} and {recommended_wine['Body']}.")
            return recommended_wine['WineID'], explanation

        return f"No sufficiently different wines found for user {user_id}.", None
    else:
        wine = recsys.recommend(user_id, 1)
        wine.rename(columns={'item': 'wine_id', 'score': 'rating'}, inplace=True)
        explanation = f"The wine '{wine.loc[0, 'wine_id']}' is recommended based on collaborative filtering."
        return wine.loc[0, 'wine_id'], explanation


# Funktion zur Generierung einer Gruppenempfehlung
def recommend_wine_for_group(group_id, group_data, merged_data, individual_recsys):
    group_info = group_data[group_data['group_id'] == group_id].iloc[0]
    group_members = eval(group_info['group_members'])

    # Empfehlung für jedes Gruppenmitglied
    recommendations = []
    for user_id in group_members:
        wine_id, explanation = recommend_wine_for_user(user_id, merged_data, individual_recsys)
        recommendations.append({
            'group_id': group_id,
            'user_id': user_id,
            'wine_id': wine_id,
            'explanation': explanation
        })

    # Ergebnis als DataFrame zurückgeben
    return pd.DataFrame(recommendations)


# Suche nach ähnlichen Weinen mit KNN
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
    return recommended_list


# Function to find KNN recommendations for all wines in the group recommendation DataFrame
def recommend_similar_wines_for_group(result_df, merged_data,wine_data,ratings_data, k=10):
    knn_recommendations = []

    # Iterate over each wine recommended to the group
    for index, row in result_df.iterrows():
        wine_id = row['wine_id']
        user_id = row['user_id']

        # Find KNN recommendations for the wine
        knn_result = find_knn_for_wine(wine_id, merged_data, user_id,wine_data,ratings_data,k)

        # Store the results in a list with the user_id for reference
        knn_recommendations.append({
            'user_id': int(user_id),
            'wine_id': int(wine_id),
            'knn_recommendations': knn_result
        })
    
    # Convert the recommendations into a DataFrame for easy viewing
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

    # order according to the scores
    category_weights_sorted = {"Type": {}, "Body": {}}
    for key, value in category_weights.items():
        category_weights_sorted[key] = dict(sorted(category_weights[key].items(), key=lambda item: item[1], reverse=True))
    return category_weights_sorted

def get_wine_with_top_categories(category_weights_sorted, wine_data, top_type_index=1, top_body_index=1):
    # Überprüfe, ob die Indizes gültig sind, um IndexError zu vermeiden
    if top_type_index > len(category_weights_sorted["Type"]) or top_body_index > len(category_weights_sorted["Body"]):
        return pd.DataFrame()  # Gebe einen leeren DataFrame zurück, wenn keine Weine mehr übrig sind
    
    # Hole die aktuellen Kategorien basierend auf den Indizes
    top_type = list(category_weights_sorted["Type"].keys())[top_type_index - 1]
    top_body = list(category_weights_sorted["Body"].keys())[top_body_index - 1]
    
    # Suche nach Weinen mit diesen Kategorien
    selection = wine_data.loc[(wine_data['Type'] == top_type) & (wine_data['Body'] == top_body)]

    # Wenn keine Weine übereinstimmen, erhöhe die Indizes und versuche es erneut
    if selection.shape[0] == 0:
        if top_type_index == top_body_index:
            return get_wine_with_top_categories(category_weights_sorted, wine_data, top_type_index + 1, top_body_index)
        else:
            return get_wine_with_top_categories(category_weights_sorted, wine_data, top_type_index, top_body_index + 1)

    # Sortiere nach 'AvgRating' und gebe die Top 5 Ergebnisse zurück
    selection = selection.sort_values(by='AvgRating', ascending=False)
    return selection.head(5)

def group_rec(group_id, group_data, merged_data,wine_data,ratings_data, individual_recsys):
    result_df = recommend_wine_for_group(group_id, group_data, merged_data, individual_recsys)
    knn_recommendation_df = recommend_similar_wines_for_group(result_df, merged_data,wine_data,ratings_data, k=10)
    category_weights_by_user = weights_user(knn_recommendation_df)
    category_weights_sorted = category_weights(category_weights_by_user)
# Beim Aufruf als Positionsargument verwenden
    recommended_wines = get_wine_with_top_categories(category_weights_sorted, wine_data, top_type_index=1, top_body_index=1)
    return recommended_wines
    