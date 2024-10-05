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

st.title('Group Wine Recommendation System')

# Input multiple users
selected_users = st.text_input('Enter User IDs (comma-separated):', '1188855, 1234567')
selected_users = [int(u.strip()) for u in selected_users.split(',')]

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

def get_combined_recommendations(selected_users):
    combined_recs_useruser = pd.DataFrame()
    combined_recs_itemitem = pd.DataFrame()

    for user in selected_users:
        if user in ratings_df_cleaned_test['user'].values:
            user_recs_useruser = recsys_user_user.recommend(user, 1000)
            enriched_useruser = enrich_recommendations(user_recs_useruser, user)
            combined_recs_useruser = pd.concat([combined_recs_useruser, enriched_useruser])

            user_recs_itemitem = recsys_item_item.recommend(user, 1000)
            enriched_itemitem = enrich_recommendations(user_recs_itemitem, user)
            combined_recs_itemitem = pd.concat([combined_recs_itemitem, enriched_itemitem])

    # Aggregate scores (e.g., averaging)
    agg_useruser = combined_recs_useruser.groupby('item').agg({'score': 'mean', 'WineName': 'first'}).reset_index()
    agg_itemitem = combined_recs_itemitem.groupby('item').agg({'score': 'mean', 'WineName': 'first'}).reset_index()

    # Merge the two aggregated recommendation lists
    merged_recs = pd.merge(agg_useruser, agg_itemitem, on='item', suffixes=('_useruser', '_itemitem'))
    
    # Rename the columns after merging to avoid missing columns error
    merged_recs.rename(columns={'score_useruser': 'User-User Score', 'score_itemitem': 'Item-Item Score'}, inplace=True)

    return merged_recs

# Display the recommendations
# Display the recommendations
if selected_users:
    st.write(f'Combined recommendations for User IDs {selected_users}:')
    
    combined_recs = get_combined_recommendations(selected_users)
    
    # Check the column names in combined_recs to debug
    st.write('Columns in combined_recs:', combined_recs.columns)
    
    # Display the first few rows of combined_recs for further inspection
    st.write('First few rows of combined_recs:')
    st.dataframe(combined_recs.head())

    
    # Display two separate tables for User-User and Item-Item recommendations
    if 'WineName_useruser' in combined_recs.columns and 'User-User Score' in combined_recs.columns:
        st.subheader('Top User-User Recommendations')
        
        # Display only User-User recommendations with corresponding wine names and scores
        useruser_recs = combined_recs[['WineName_useruser', 'User-User Score']].dropna().sort_values(by='User-User Score', ascending=False)
        useruser_recs = useruser_recs.rename(columns={'WineName_useruser': 'WineName'})
        st.dataframe(useruser_recs)
    else:
        st.write('Error: User-User recommendations not found!')

    if 'WineName_itemitem' in combined_recs.columns and 'Item-Item Score' in combined_recs.columns:
        st.subheader('Top Item-Item Recommendations')
        
        # Display only Item-Item recommendations with corresponding wine names and scores
        itemitem_recs = combined_recs[['WineName_itemitem', 'Item-Item Score']].dropna().sort_values(by='Item-Item Score', ascending=False)
        itemitem_recs = itemitem_recs.rename(columns={'WineName_itemitem': 'WineName'})
        st.dataframe(itemitem_recs)
    else:
        st.write('Error: Item-Item recommendations not found!')


