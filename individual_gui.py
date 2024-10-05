import streamlit as st
import pandas as pd
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

def enrich_recommendations(recs_df, user_id, algorithm_type):
    wine_names = []
    user_ratings = []
    explanations = []

    for _, row in recs_df.iterrows():
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
            region = 'Unknown Region'
            grapes = 'Unknown Grapes'
            wine_names.append(wine_name)

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
    recs_df['Explanation'] = explanations

    return recs_df

def recommend_top_wines():
    avg_ratings = ratings_df_cleaned.groupby('item').rating.mean().reset_index()
    top_wines = avg_ratings.sort_values(by='rating', ascending=False).head(10)
    top_wines_enriched = enrich_recommendations(top_wines, None, 'Popularity')
    return top_wines_enriched


if selected_user:
    if selected_user in ratings_df_cleaned_test['user'].values:
        st.write(f'Recommendations for User ID {selected_user}:')

        selected_wines_useruser = recsys_user_user.recommend(selected_user, 1000)
        enriched_useruser_recs = enrich_recommendations(selected_wines_useruser, selected_user, 'User-User')
        st.subheader('User-User Recommendations')
        st.dataframe(enriched_useruser_recs[['item', 'WineName', 'score', 'Your Rating']])

        user_user_wine_selection = st.selectbox(
            'Select a wine from User-User recommendations to view detailed explanation:',
            options=enriched_useruser_recs['WineName'].tolist(),
            index=0,
            key='user_user_select'
        )
        if user_user_wine_selection:
            user_user_explanation = enriched_useruser_recs.loc[enriched_useruser_recs['WineName'] == user_user_wine_selection, 'Explanation'].values[0]
            st.write(f"**Explanation for {user_user_wine_selection} (User-User):** {user_user_explanation}")

        selected_wines_itemitem = recsys_item_item.recommend(selected_user, 1000)
        enriched_itemitem_recs = enrich_recommendations(selected_wines_itemitem, selected_user, 'Item-Item')
        st.subheader('Item-Item Recommendations')
        st.dataframe(enriched_itemitem_recs[['item', 'WineName', 'score', 'Your Rating']])

        item_item_wine_selection = st.selectbox(
            'Select a wine from Item-Item recommendations to view detailed explanation:',
            options=enriched_itemitem_recs['WineName'].tolist(),
            index=0,
            key='item_item_select'
        )
        if item_item_wine_selection:
            item_item_explanation = enriched_itemitem_recs.loc[enriched_itemitem_recs['WineName'] == item_item_wine_selection, 'Explanation'].values[0]
            st.write(f"**Explanation for {item_item_wine_selection} (Item-Item):** {item_item_explanation}")

    else:
        st.write(f'User ID {selected_user} not found! Recommending top-rated wines:')
        top_wines = recommend_top_wines()
        st.subheader('Top Rated Wines')
        st.dataframe(top_wines[['item', 'WineName', 'rating']])

        wine_selection = st.selectbox(
            'Select a wine to view detailed explanation:',
            options=top_wines['WineName'].tolist(),
            index=0,
            key='top_wines_select'
        )

        if wine_selection:
            selected_explanation = top_wines.loc[top_wines['WineName'] == wine_selection, 'Explanation'].values[0]
            st.write(f"**Explanation for {wine_selection}:** {selected_explanation}")
