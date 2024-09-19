import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler

wines_df = pd.read_csv("Dataset/last/XWines_Slim_1K_wines.csv", index_col="WineID")

for column in wines_df.columns:
  print(wines_df[column])

le = LabelEncoder()
wines_df['Elaborate'] = le.fit_transform(wines_df['Elaborate'])
wines_df['Code'] = le.fit_transform(wines_df['Code'])
wines_df['Country'] = le.fit_transform(wines_df['Country'])
wines_df['Type'] = le.fit_transform(wines_df['Type'])

# mlb_grapes = MultiLabelBinarizer()
# wines_df['Grapes'] = wines_df['Grapes'].apply(lambda x: eval(x))
# grapes_encoded = pd.DataFrame(mlb_grapes.fit_transform(wines_df['Grapes']), 
#                               columns=mlb_grapes.classes_)
# wines_df = pd.concat([wines_df, grapes_encoded], axis=1).drop('Grapes', axis=1)

# mlb_food = MultiLabelBinarizer()
# wines_df['Harmonize'] = wines_df['Harmonize'].apply(lambda x: eval(x)) 
# food_encoded = pd.DataFrame(mlb_food.fit_transform(wines_df['Harmonize']), 
#                             columns=mlb_food.classes_)
# wines_df = pd.concat([wines_df, food_encoded], axis=1).drop('Harmonize', axis=1)



print(wines_df.shape)