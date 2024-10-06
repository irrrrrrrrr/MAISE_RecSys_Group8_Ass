import streamlit as st
import pandas as pd
from group_recommender_algs import avg_ratings, group_rec

# Streamlit App Titel
st.title("Group Recommendation System")

# Lade die Datensätze
ratings_data = pd.read_csv("Dataset/last/XWines_Slim_150K_ratings.csv")
wine_data = pd.read_csv("Dataset/last/XWines_Slim_1K_wines.csv")
merged_data = pd.merge(ratings_data, wine_data, on='WineID')
wine_data = avg_ratings(wine_data, ratings_data)
group_data = pd.read_csv("Dataset/last/group_composition.csv")

# Bestätigung der geladenen Daten
st.write("Datasets loaded successfully.")            

# Eingabefeld für die Group ID
st.write("Enter a Group ID (0-239):")
group_id = st.number_input('Enter Group ID:', min_value=0, max_value=239, value=0)

# Button zur Generierung der Empfehlungen
if st.button("Get Recommendations"):
    # Rufe die `group_rec`-Funktion auf
    recommended_wines = group_rec(group_id, group_data, merged_data, wine_data, ratings_data, None)

    # Überprüfe, ob eine gültige Empfehlung zurückgegeben wurde
    if isinstance(recommended_wines, str):
        st.write(recommended_wines)
    else:
        st.write(f"Recommended Wines for Group ID {group_id}:")
        if recommended_wines is None or recommended_wines.empty:
            st.write("No wine recommendations found.")
        else:
            # Empfohlene Weine in einer Tabelle anzeigen
            st.dataframe(recommended_wines[['WineID', 'Type', 'Body', 'AvgRating']])

            # Erklärungen zu den Empfehlungen anzeigen
            st.subheader("Explanations:")
            for _, row in recommended_wines.iterrows():
                explanation = (f"The wine '{row['WineID']}' was recommended based on its type '{row['Type']}' and body '{row['Body']}', "
                               f"with an average rating of {row['AvgRating']:.2f}.")
                st.write(explanation)
