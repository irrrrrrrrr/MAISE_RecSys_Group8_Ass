import pandas as pd
from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser
from lenskit.algorithms.item_knn import ItemItem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm  


print('Loading datasets...')
ratings_df = pd.read_csv("Dataset/ratings_splits/temporal_global/filtered/train.csv")
ratings_df_cleaned = ratings_df.drop(columns=['RatingID', 'Date', 'Vintage']).rename(columns={'WineID': 'item', 'UserID': 'user', 'Rating': 'rating'})

ratings_df_test = pd.read_csv("Dataset/ratings_splits/temporal_global/filtered/test.csv")
ratings_df_cleaned_test = ratings_df_test.drop(columns=['RatingID', 'Date', 'Vintage']).rename(columns={'WineID': 'item', 'UserID': 'user', 'Rating': 'rating'})

wines_df = pd.read_csv('Dataset/last/Xwines_Slim_1K_wines.csv', index_col="WineID")
wines_df['WineID'] = wines_df.index


print('Fitting User-User algorithm...')
user_user = UserUser(15, min_nbrs=3)
recsys_user_user = Recommender.adapt(user_user)
recsys_user_user.fit(ratings_df_cleaned)

print('Fitting Item-Item algorithm...')
itemitem = ItemItem(15, min_nbrs=3)
recsys_item_item = Recommender.adapt(itemitem)
recsys_item_item.fit(ratings_df_cleaned)


def enrich_recommendations(recs_df, user_id, algorithm_type):
    wine_names = []
    user_ratings = []
    explanations = []
    regions = []    
    grape_varieties = []  

    for _, row in recs_df.iterrows():
        
        region = 'Unknown Region'
        grapes = 'Unknown Grapes'
        
        
        wine_row = wines_df[wines_df['WineID'] == row['item']]

        if not wine_row.empty:
            wine_name = wine_row.iloc[0]['WineName']
            region = wine_row.iloc[0]['RegionName'] if 'RegionName' in wine_row.columns and pd.notna(wine_row.iloc[0]['RegionName']) else 'Unknown Region'
            grapes = wine_row.iloc[0]['Grapes'] if 'Grapes' in wine_row.columns and pd.notna(wine_row.iloc[0]['Grapes']) else 'Unknown Grapes'
            if grapes and grapes != 'Unknown Grapes':
                grapes = grapes.replace('[', '').replace(']', '').replace("'", "").strip()

            wine_names.append(wine_name)
        else:
            wine_name = 'Unknown Wine'
            wine_names.append(wine_name)

        
        regions.append(region)
        grape_varieties.append(grapes)

        
        user_rating_row = ratings_df_cleaned_test[(ratings_df_cleaned_test['user'] == user_id)]
        user_rating_wine_row = user_rating_row[user_rating_row['item'] == row['item']]

        if not user_rating_wine_row.empty:
            user_rating = user_rating_wine_row.iloc[0]['rating']
            user_ratings.append(user_rating)
        else:
            user_rating = 'No rating'
            user_ratings.append(user_rating)
            
        
        if algorithm_type == 'User-User':
            explanation = (f"This wine is recommended because users with similar preferences to you gave it an average rating of {row['score']:.2f}. "
                           f"For example, users who rated wines like {wine_name} highly have also enjoyed other wines from the {region} region, "
                           f"which use similar grape varieties such as {grapes}.")
        elif algorithm_type == 'Item-Item':
            explanation = (f"This wine is recommended because it has a similar profile to wines you've liked before. "
                           f"It shares characteristics such as being from the {region} region and using grape varieties like {grapes}. "
                           f"Similarity score: {row['score']:.2f}.")
        else:
            explanation = (f"This wine is a top-rated choice among all users, known for its exceptional balance of flavors and high ratings. "
                           f"It features grape varieties like {grapes} and comes from the {region} region.")

        explanations.append(explanation)

    
    recs_df['WineName'] = wine_names
    recs_df['Your Rating'] = user_ratings
    recs_df['RegionName'] = regions  
    recs_df['Grapes'] = grape_varieties  
    recs_df['Explanation'] = explanations

    return recs_df



def calculate_content_diversity(recommendations_df, content_columns=['RegionName', 'Grapes']):
    """
    Berechnet die Diversität basierend auf den inhaltlichen Eigenschaften der Weine.
    """
    
    for col in content_columns:
        if col not in recommendations_df.columns:
            recommendations_df[col] = 'Unknown'  

    
    recommendations_df['combined_content'] = recommendations_df[content_columns].fillna('').apply(lambda row: ' '.join(row.astype(str)), axis=1)

    
    valid_contents = recommendations_df['combined_content'].replace('', pd.NA).dropna()

    if valid_contents.empty:
        return 0

    
    vectorizer = TfidfVectorizer()
    try:
        content_vectors = vectorizer.fit_transform(valid_contents)
    except ValueError:
        
        return 0

    
    similarity_matrix = cosine_similarity(content_vectors)

    
    num_items = similarity_matrix.shape[0]
    if num_items < 2:
        return 1.0  

    
    sum_similarity = similarity_matrix.sum() - num_items  
    avg_similarity = sum_similarity / (num_items * (num_items - 1))

    
    diversity_score = 1 - avg_similarity
    return diversity_score


def calculate_explanation_coherence(recommendations_df):
    """
    Berechnet die Coherence basierend auf mehreren Attributen der Weine.
    """
    if 'WineName' not in recommendations_df.columns:
        return 0

    
    recommendations_df['combined_content'] = recommendations_df.apply(lambda row: ' '.join(row[['WineName', 'RegionName', 'Grapes']].fillna('').astype(str)), axis=1)

    
    valid_content = recommendations_df['combined_content'].replace('', pd.NA).dropna()

    
    if valid_content.empty:
        return 0

    
    vectorizer = TfidfVectorizer()
    try:
        content_vectors = vectorizer.fit_transform(valid_content)
    except ValueError:
        
        return 0

    
    similarity_matrix = cosine_similarity(content_vectors)

    
    coherence_score = similarity_matrix.mean() if similarity_matrix.size > 1 else 1.0
    return coherence_score



print("Calculating metrics for all users...")

user_ids = ratings_df_cleaned_test['user'].unique()
all_user_metrics = []


for user_id in tqdm(user_ids, desc="Processing users"):
    
    selected_wines_useruser = recsys_user_user.recommend(user_id, 10)
    enriched_useruser_recs = enrich_recommendations(selected_wines_useruser, user_id, algorithm_type='User-User')

    
    selected_wines_itemitem = recsys_item_item.recommend(user_id, 10)
    enriched_itemitem_recs = enrich_recommendations(selected_wines_itemitem, user_id, algorithm_type='Item-Item')

    
    useruser_recs_df = enriched_useruser_recs[['WineName', 'RegionName', 'Grapes', 'Your Rating', 'score']].rename(columns={'score': 'User-User Score'})
    itemitem_recs_df = enriched_itemitem_recs[['WineName', 'RegionName', 'Grapes', 'score']].rename(columns={'score': 'Item-Item Score'})
    merged_recs_df = pd.merge(useruser_recs_df, itemitem_recs_df, on=['WineName', 'RegionName', 'Grapes'], how='outer')
    merged_recs_df['Your Rating'] = pd.to_numeric(merged_recs_df['Your Rating'], errors='coerce')

    coherence = calculate_explanation_coherence(merged_recs_df)
    diversity = calculate_content_diversity(merged_recs_df)

    
    all_user_metrics.append({
        'UserID': user_id,
        'Explanation Coherence': coherence,
        'Diversity': diversity
    })


metrics_df = pd.DataFrame(all_user_metrics)
average_metrics = metrics_df.mean()

print("----- Average Metrics for All Users -----")
print(average_metrics)


metrics_df.to_csv("user_explanation_metrics.csv", index=False)
print("Metrics for each user saved to 'user_explanation_metrics.csv'.")
