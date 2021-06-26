import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import pickle
import numpy as np
path = [
    'demographic.csv',
    'diet.csv',
    'examination.csv',
    'labs.csv',
    'questionnaire.csv',

]

#  input


dfname = [
    'dm',
    'di',
    'qs'
]

conditions = {
    'DLQ010': 'Serious difficulty hearing',
    'DLQ020': 'Serious Difficulty seeing',
    'DLQ040': 'Serious difficulty concentrating, remembering or making decisions',
    'DLQ050': 'Serious difficulty walking or climbing',
    'DLQ060': 'difficulty dressing or bathing',

    'CDQ001': 'Pain or discomfort in chest',
    'CDQ008': 'severe pain across chest',
    'CDQ010': 'Shortness of breath',

    'BPQ020': 'hypertension/high blood pressure',
    'BPQ080': 'high blood cholesterol level',

    'AGQ030': 'Hay fever',
    'MCQ010': 'Asthma',
    'MCQ053': 'Anemia/tired blood/low blood',
    'MCQ070': 'Psoriasis',
    'MCQ080': 'Obesity',
    'MCQ082': 'Celiac Disease',
    'MCQ160A': 'Arthritis',
    'MCQ160B': 'Congestive heart failure',
    'MCQ160C': 'Coronary heart disease',
    'MCQ160D': 'Angina',
    'MCQ160E': 'Heart Attack',
    'MCQ160F': 'Stroke',
    'MCQ160G': 'Emphysema',
    'MCQ160K': 'Chronic Bronchitis',

}



best = 0
model_exists = input("does model exist (0 for yes OR int for # of fits for new model): ")
if model_exists == '0':
    print('Loading model...')
    for k in conditions:
        d1 = pd.read_csv(path[0])
        d2 = pd.read_csv(path[1])
        d3 = pd.read_csv(path[2])
        d4 = pd.read_csv(path[3])
        d5 = pd.read_csv(path[4])

        df = pd.concat([d1, d2], axis=1, join='inner')
        df = pd.concat([df, d3], axis=1, join='inner')
        df = pd.concat([df, d4], axis=1, join='inner')
        df = pd.concat([df, d5], axis=1, join='inner')
        df.dropna(axis=1, how='all')
        df.dropna(axis=0, how='all')

        df = df.fillna(0.0)

        df = df.rename(columns={'SEQN': 'ID',
                                'RIDAGEYR': 'Age',
                                'RIAGENDR': 'Gender',
                                'DMDMARTL': 'House_Hold',
                                'BMXBMI': 'BMI',
                                'DR1BWATZ': 'Water_drank',
                                'BMXWT': 'Weight_g',
                                'DMDCITZN': 'In_US',
                                k: conditions.get(k)})
        df[conditions.get(k)] = df[conditions.get(k)].replace(2., 0.)
        df = df[['ID', 'Age', 'Gender', 'House_Hold', 'BMI', 'Weight_g', 'In_US', conditions.get(k)]]
        df = df.loc[:, ~df.columns.duplicated()]

        X = df.drop(columns=[conditions.get(k)])
        y = df[conditions.get(k)]

        model = pickle.load(open(f"{k}.pickle", "rb"))
        print(model.predict(X.head(1)))



else:
    print("Training model " + model_exists + ' times')

    for _ in range(int(model_exists)):

        for k in conditions:
            d1 = pd.read_csv(path[0])
            d2 = pd.read_csv(path[1])
            d3 = pd.read_csv(path[2])
            d4 = pd.read_csv(path[3])
            d5 = pd.read_csv(path[4])

            df = pd.concat([d1, d2], axis=1, join='inner')
            df = pd.concat([df, d3], axis=1, join='inner')
            df = pd.concat([df, d4], axis=1, join='inner')
            df = pd.concat([df, d5], axis=1, join='inner')
            df.dropna(axis=1, how='all')
            df.dropna(axis=0, how='all')



            df = df.fillna(0.0)

            df = df.rename(columns={'SEQN': 'ID',
                                    'RIDAGEYR': 'Age',
                                    'RIAGENDR': 'Gender',
                                    'DMDMARTL': 'House_Hold',
                                    'BMXBMI': 'BMI',
                                    'DR1BWATZ': 'Water_drank',
                                    'BMXWT': 'Weight_g',
                                    'DMDCITZN': 'In_US',
                                    k: conditions.get(k)})
            df[conditions.get(k)] = df[conditions.get(k)].replace(2., 0.)
            df = df[['ID', 'Age', 'Gender', 'House_Hold', 'BMI', 'Weight_g', 'In_US', conditions.get(k)]]
            df = df.loc[:, ~df.columns.duplicated()]

            X = df.drop(columns=[conditions.get(k)])
            y = df[conditions.get(k)]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = DecisionTreeClassifier()


            model.fit(X_train, y_train)

            predict = model.predict(X_test)


            with open(f'{k}.pickle', 'wb') as file:
                pickle.dump(model, file)

            del df




