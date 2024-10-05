import streamlit as st
import pandas as pd
from group_recommender_algs import (avg_ratings,group_rec)

st.title("Group Recommendation System")


ratings_data = pd.read_csv("Dataset/last/XWines_Slim_150K_ratings.csv")
wine_data = pd.read_csv("Dataset/last/XWines_Slim_1K_wines.csv")
merged_data = pd.merge(ratings_data, wine_data, on='WineID')
wine_data = avg_ratings(wine_data,ratings_data)
group_data = pd.read_csv("Dataset/last/group_composition.csv")
st.write("Datasets loaded")            


st.write("Enter a Group ID (0-239)")
group_id = st.number_input('Enter Group ID:', min_value=0, max_value=239, value=0)

if st.button("Get Recommendations"):
    recommended_wines = group_rec(group_id, group_data, merged_data,wine_data,ratings_data, None)
    if isinstance(recommended_wines, str):
        st.write(recommended_wines)
    else:
        st.write(f"Recommended Wines for Group ID {group_id}:")
        if recommended_wines is None or recommended_wines.empty:
            st.write("No wine recommendations found.")
        else:
            st.dataframe(recommended_wines[['WineID', 'Type', 'Body', 'AvgRating']])
