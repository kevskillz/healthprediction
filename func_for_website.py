import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pickle


# returns array of potential conditions
def predict_conditions(age, gender, told_to_exercise, bmi, water, weight_kg, country):
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
    in_us = 0
    t_o_e = 0
    gen = 0  # female
    if country == 'United States':
        in_us = 1
    if told_to_exercise == 'Yes':
        t_o_e = 1
    if gender == 'Male':
        gen = 1

    retArr = []
    for k in conditions:
        model = pickle.load(open(f"{k}.pickle", "rb"))
        if model.predict([[age, gen, t_o_e, bmi, water, weight_kg, in_us]])[0] == 1.:
            retArr.append(conditions.get(k))
