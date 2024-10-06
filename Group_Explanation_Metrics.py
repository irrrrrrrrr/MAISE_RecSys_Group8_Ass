import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from group_recommender_algs import group_rec, avg_ratings

# -------------------- 1. Daten laden --------------------
print('Loading datasets...')
ratings_df = pd.read_csv("Dataset/last/XWines_Slim_150K_ratings.csv", low_memory=False)
wine_data = pd.read_csv('Dataset/last/XWines_Slim_1K_wines.csv')
wine_data['WineID'] = wine_data.index

# Überprüfe und konvertiere den Datentyp der WineID-Spalten
ratings_df['WineID'] = pd.to_numeric(ratings_df['WineID'], errors='coerce').astype('Int64')
wine_data['WineID'] = pd.to_numeric(wine_data['WineID'], errors='coerce').astype('Int64')

# Entferne fehlerhafte Einträge
ratings_df = ratings_df.dropna(subset=['WineID'])
wine_data = wine_data.dropna(subset=['WineID'])

import pandas as pd

# -------------------- Daten laden --------------------
ratings_df = pd.read_csv("Dataset/last/XWines_Slim_150K_ratings.csv", low_memory=False)
wine_data = pd.read_csv('Dataset/last/XWines_Slim_1K_wines.csv')

# Überprüfen und Bereinigen der Ratings-Daten
ratings_df['WineID'] = pd.to_numeric(ratings_df['WineID'], errors='coerce').astype('Int64')
ratings_df = ratings_df.dropna(subset=['WineID'])

# Bereinigung und Standardisierung der Weindaten
wine_data['Type'] = wine_data['Type'].fillna('Unknown Type')
wine_data['Body'] = wine_data['Body'].fillna('Unknown Body')

# Merge der Daten
merged_data = pd.merge(ratings_df, wine_data, on='WineID', how='inner')

# -------------------- Gruppenkompositionsdaten laden --------------------
group_data = pd.read_csv("Dataset/last/group_composition.csv")




# Prüfen, ob 'group_id' in den Spaltennamen vorhanden ist
if 'group_id' not in group_data.columns:
    raise ValueError("Die Spalte 'group_id' ist nicht im DataFrame 'group_data' vorhanden. Verfügbare Spalten sind: " + ", ".join(group_data.columns))

# -------------------- Gruppen-IDs extrahieren --------------------
group_ids = group_data['group_id'].unique()


# Restlicher Code für die Gruppenempfehlung ...


# Lade die Gruppenkompositionen
group_data = pd.read_csv("Dataset/last/group_composition.csv")

# Berechne durchschnittliche Bewertungen
wine_data = avg_ratings(wine_data, ratings_df)
merged_data = pd.merge(ratings_df, wine_data, on='WineID')
# -------------------- 2. Metrikberechnungsfunktionen --------------------
def calculate_content_diversity(recommendations_df, content_columns=['RegionName', 'Grapes']):
    """
    Berechnet die Diversität basierend auf den inhaltlichen Eigenschaften der Weine.
    """
    for col in content_columns:
        if col not in recommendations_df.columns:
            recommendations_df[col] = 'Unknown'  # Füge eine Standardspalte hinzu, falls sie fehlt

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

# -------------------- 3. Metriken für alle Gruppen berechnen mit Fortschrittsanzeige --------------------
print("Calculating metrics for all groups...")

group_ids = group_data['group_id'].unique()
all_group_metrics = []

for group_id in tqdm(group_ids, desc="Processing groups"):
    # Berechne Empfehlungen für die aktuelle Gruppe
    recommended_wines = group_rec(group_id, group_data, merged_data, wine_data, ratings_df, None)
    
    if recommended_wines is not None and not recommended_wines.empty:
        # Berechne die Diversität und Coherence der Empfehlungen
        diversity = calculate_content_diversity(recommended_wines)
        coherence = calculate_explanation_coherence(recommended_wines)

        # Speichern der Metriken für die aktuelle Gruppe
        all_group_metrics.append({
            'GroupID': group_id,
            'Explanation Coherence': coherence,
            'Diversity': diversity
        })
    else:
        print(f"No recommendations found for Group {group_id}.")

# -------------------- 4. Ergebnisse berechnen und ausgeben --------------------
metrics_df = pd.DataFrame(all_group_metrics)
average_metrics = metrics_df.mean()

print("----- Average Metrics for All Groups -----")
print(average_metrics)

# Speichern der Metriken als CSV-Datei
metrics_df.to_csv("group_explanation_metrics.csv", index=False)
print("Metrics for each group saved to 'group_explanation_metrics.csv'.")
