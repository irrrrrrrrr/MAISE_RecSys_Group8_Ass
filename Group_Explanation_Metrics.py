import pandas as pd
from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser
from lenskit.algorithms.item_knn import ItemItem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm  # Fortschrittsanzeige

# -------------------- 1. Daten laden --------------------
print('Loading datasets...')
ratings_df = pd.read_csv("Dataset/ratings_splits/temporal_global/filtered/train.csv")
ratings_df_cleaned = ratings_df.drop(columns=['RatingID', 'Date', 'Vintage']).rename(columns={'WineID': 'item', 'UserID': 'user', 'Rating': 'rating'})

ratings_df_test = pd.read_csv("Dataset/ratings_splits/temporal_global/filtered/test.csv")
ratings_df_cleaned_test = ratings_df_test.drop(columns=['RatingID', 'Date', 'Vintage']).rename(columns={'WineID': 'item', 'UserID': 'user', 'Rating': 'rating'})

wines_df = pd.read_csv('Dataset/last/Xwines_Slim_1K_wines.csv', index_col="WineID")
wines_df['WineID'] = wines_df.index

# -------------------- 2. Empfehlungsalgorithmen --------------------
print('Fitting User-User algorithm...')
user_user = UserUser(15, min_nbrs=3)
recsys_user_user = Recommender.adapt(user_user)
recsys_user_user.fit(ratings_df_cleaned)

print('Fitting Item-Item algorithm...')
itemitem = ItemItem(15, min_nbrs=3)
recsys_item_item = Recommender.adapt(itemitem)
recsys_item_item.fit(ratings_df_cleaned)

# -------------------- 3. Anreicherungsfunktion für Empfehlungen --------------------
def enrich_recommendations(recs_df, user_id):
    wine_names = []
    user_ratings = []

    for _, row in recs_df.iterrows():
        wine_row = wines_df[wines_df['WineID'] == row['item']]
        wine_names.append(wine_row.iloc[0]['WineName'] if not wine_row.empty else 'Unknown Wine')
        
        user_rating_row = ratings_df_cleaned_test[(ratings_df_cleaned_test['user'] == user_id)]
        user_rating_wine_row = user_rating_row[user_rating_row['item'] == row['item']]
        user_ratings.append(user_rating_wine_row.iloc[0]['rating'] if not user_rating_wine_row.empty else 'No rating')
    
    recs_df['WineName'] = wine_names
    recs_df['Your Rating'] = user_ratings
    return recs_df

# -------------------- 4. Metrikberechnungsfunktionen --------------------
def calculate_acceptance_rate(recommendations_df):
    accepted_recommendations = recommendations_df[recommendations_df['Your Rating'] >= 4].shape[0]
    total_recommendations = recommendations_df.shape[0]
    return accepted_recommendations / total_recommendations if total_recommendations > 0 else 0

def calculate_satisfaction_score(recommendations_df):
    return recommendations_df['Your Rating'].mean() if not recommendations_df.empty else 0

def calculate_coverage(recommendations_df):
    explained_recommendations = recommendations_df[recommendations_df['Your Rating'].notna()].shape[0]
    total_recommendations = recommendations_df.shape[0]
    return explained_recommendations / total_recommendations if total_recommendations > 0 else 0

def calculate_diversity(recommendations_df, attribute='WineName'):
    if attribute not in recommendations_df.columns:
        return 0
    unique_values = recommendations_df[attribute].nunique()
    total_recommendations = recommendations_df.shape[0]
    return unique_values / total_recommendations if total_recommendations > 0 else 0

def calculate_explanation_coherence(recommendations_df):
    if 'WineName' not in recommendations_df.columns or recommendations_df['WineName'].dropna().empty:
        return 0

    valid_wine_names = recommendations_df['WineName'].dropna()
    valid_wine_names = valid_wine_names[valid_wine_names.apply(lambda x: isinstance(x, str) and x.strip() != "")]

    if valid_wine_names.empty:
        return 0

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(valid_wine_names)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix.mean() if similarity_matrix.size > 1 else 1.0

# -------------------- 5. Metriken für alle Benutzer berechnen mit Fortschrittsanzeige --------------------
print("Calculating metrics for all users...")

user_ids = ratings_df_cleaned_test['user'].unique()
all_user_metrics = []

# Verwende tqdm zur Fortschrittsanzeige
for user_id in tqdm(user_ids, desc="Processing users"):
    # User-User Empfehlungen
    selected_wines_useruser = recsys_user_user.recommend(user_id, 10)
    enriched_useruser_recs = enrich_recommendations(selected_wines_useruser, user_id)

    # Item-Item Empfehlungen
    selected_wines_itemitem = recsys_item_item.recommend(user_id, 10)
    enriched_itemitem_recs = enrich_recommendations(selected_wines_itemitem, user_id)

    # Zusammenführung der Ergebnisse
    useruser_recs_df = enriched_useruser_recs[['WineName', 'Your Rating', 'score']].rename(columns={'score': 'User-User Score'})
    itemitem_recs_df = enriched_itemitem_recs[['WineName', 'score']].rename(columns={'score': 'Item-Item Score'})
    merged_recs_df = pd.merge(useruser_recs_df, itemitem_recs_df, on='WineName', how='outer')
    merged_recs_df['Your Rating'] = pd.to_numeric(merged_recs_df['Your Rating'], errors='coerce')

    # Metriken für diesen Benutzer berechnen
    acceptance_rate = calculate_acceptance_rate(merged_recs_df)
    satisfaction_score = calculate_satisfaction_score(merged_recs_df)
    coverage = calculate_coverage(merged_recs_df)
    coherence = calculate_explanation_coherence(merged_recs_df)
    diversity = calculate_diversity(merged_recs_df, attribute='WineName')

    # Speichern der Metriken
    all_user_metrics.append({
        'UserID': user_id,
        'Acceptance Rate': acceptance_rate,
        'Satisfaction Score': satisfaction_score,
        'Coverage': coverage,
        'Explanation Coherence': coherence,
        'Diversity': diversity
    })

# -------------------- 6. Ergebnisse berechnen und ausgeben --------------------
metrics_df = pd.DataFrame(all_user_metrics)
average_metrics = metrics_df.mean()

print("----- Average Metrics for All Users -----")
print(average_metrics)

# Speichern der Metriken als CSV-Datei
metrics_df.to_csv("user_explanation_metrics.csv", index=False)
print("Metrics for each user saved to 'user_explanation_metrics.csv'.")
