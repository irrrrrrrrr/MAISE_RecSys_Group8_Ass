import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser
from lenskit.algorithms.item_knn import ItemItem

st.write('Loading datasets...')
ratings_df = pd.read_csv("Dataset/ratings_splits/temporal_global/filtered/train.csv")
ratings_df_cleaned = ratings_df.drop(columns=['RatingID', 'Date', 'Vintage']).rename(columns={'WineID': 'item', 'UserID': 'user', 'Rating': 'rating'})

ratings_df_test = pd.read_csv("Dataset/ratings_splits/temporal_global/filtered/test.csv") 
ratings_df_cleaned_test = ratings_df_test.drop(columns=['RatingID', 'Date', 'Vintage']).rename(columns={'WineID': 'item', 'UserID': 'user', 'Rating': 'rating'})

wines_df = pd.read_csv('Dataset/last/Xwines_Slim_1K_wines.csv', index_col="WineID")
wines_df['WineID'] = wines_df.index

st.title('Wine Recommendation System')
st.write('Enter a user ID to get recommendations using User-User and Item-Item algorithms.')

selected_user = st.number_input('Enter User ID:', min_value=1, value=1188855)

st.write('Fitting User-User algorithm...')
user_user = UserUser(15, min_nbrs=3)
recsys_user_user = Recommender.adapt(user_user)
recsys_user_user.fit(ratings_df_cleaned)

st.write('Fitting Item-Item algorithm...')
itemitem = ItemItem(15, min_nbrs=3)
recsys_item_item = Recommender.adapt(itemitem)
recsys_item_item.fit(ratings_df_cleaned)

def enrich_recommendations(recs_df, user_id):
    wine_names = []
    user_ratings = []

    for _, row in recs_df.iterrows():
        wine_row = wines_df[wines_df['WineID'] == row['item']]
        
        if not wine_row.empty:
            wine_names.append(wine_row.iloc[0]['WineName'])
        else:
            wine_names.append('Unknown Wine')
        
        user_rating_row = ratings_df_cleaned_test[(ratings_df_cleaned_test['user'] == user_id)]
        user_rating_wine_row = user_rating_row[user_rating_row['item'] == row['item']]
        
        if not user_rating_wine_row.empty:
            user_ratings.append(user_rating_wine_row.iloc[0]['rating'])
        else:
            user_ratings.append('No rating')
    
    recs_df['WineName'] = wine_names
    recs_df['Your Rating'] = user_ratings
    
    return recs_df

def recommend_top_wines():
    avg_ratings = ratings_df_cleaned.groupby('item').rating.mean().reset_index()
    top_wines = avg_ratings.sort_values(by='rating', ascending=False).head(10)
    
    top_wines_enriched = enrich_recommendations(top_wines, None)
    return top_wines_enriched


if selected_user:
    if selected_user in ratings_df_cleaned_test['user'].values:
        st.write(f'Recommendations for User ID {selected_user}:')

        selected_wines_useruser = recsys_user_user.recommend(selected_user, 1000)
        enriched_useruser_recs = enrich_recommendations(selected_wines_useruser, selected_user)

        st.subheader('User-User Recommendations')
        st.dataframe(enriched_useruser_recs[['item', 'WineName', 'score', 'Your Rating']])

        selected_wines_itemitem = recsys_item_item.recommend(selected_user, 1000)
        enriched_itemitem_recs = enrich_recommendations(selected_wines_itemitem, selected_user)

        st.subheader('Item-Item Recommendations')
        st.dataframe(enriched_itemitem_recs[['item', 'WineName', 'score', 'Your Rating']])

        
        rated_recommendations = []

        # Loop through User-User recommendations
        for _, row in enriched_useruser_recs.iterrows():
            if row['Your Rating'] != 'No rating':
                rated_recommendations.append({
                    'WineName': row['WineName'],
                    'Your Rating': row['Your Rating'],
                    'Score': row['score'],
                    'Method': 'User-User'
                })

        # Loop through Item-Item recommendations
        for _, row in enriched_itemitem_recs.iterrows():
            if row['Your Rating'] != 'No rating':
                rated_recommendations.append({
                    'WineName': row['WineName'],
                    'Your Rating': row['Your Rating'],
                    'Score': row['score'],
                    'Method': 'Item-Item'
                })

        
        rated_recommendations_df = pd.DataFrame(rated_recommendations)
        rated_recommendations_df['Your Rating'] = pd.to_numeric(rated_recommendations_df['Your Rating'], errors='coerce')
        rated_recommendations_df = rated_recommendations_df.sort_values(by='Your Rating', ascending=False)

        
        if not rated_recommendations_df.empty:
            st.subheader('Rated Recommendations')
            st.dataframe(rated_recommendations_df)
        else:
            st.write('No rated recommendations found.')

    else:
        st.write(f'User ID {selected_user} not found! Recommending top-rated wines:')
        
        top_wines = recommend_top_wines()
        st.subheader('Top Rated Wines')
        st.dataframe(top_wines[['item', 'WineName', 'rating']])
