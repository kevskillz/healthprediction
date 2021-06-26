import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import pickle

path = [
    'demographic.csv',
    'diet.csv',
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
    'DLQ050': 'Serious difficulty walking or climbing stairs',
    'DLQ060': 'difficulty dressing or bathing',

    'CDQ001': 'Pain or discomfort in chest',
    'CDQ008': 'severe pain across chest',
    'CDQ010': 'Shortness of breath when walking up a slight hill',

    'BPQ020': 'hypertension/high blood pressure',
    'BPQ059': 'blood cholesterol checked',
    'BPQ080': 'high blood cholesterol level',

    'AGQ030': 'Hay fever',
    'MCD093': 'First transfusion',
    'MCQ010': 'Asthma',
    'MCQ053': 'Anemia/tired blood/low blood',
    'MCQ070': 'Psoriasis',
    'MCQ080': 'Overweight',
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
    model = pickle.load(open("model.pickle", "rb"))
    for k in conditions:
        print(k + ' ' + conditions.get(k))
        df = {}
        dfn = dict(zip(dfname, path))
        df = {key: pd.read_csv(value) for key, value in dfn.items()}

        Xs = {k: v for k, v in df.items() if k in ['dm', 'di']}

        dfs = Xs.values()

        from functools import partial, reduce

        inner_merge = partial(pd.merge, how='inner', on='SEQN')

        c = reduce(inner_merge, dfs)

        # check if there are duplicated SEQN

        qs = df['qs'][['SEQN', k]]
        c = pd.merge(c, qs, how='left', on='SEQN')

        # MCQ160F (target feature): exclude null values and NA
        c = c[(c.get(k).notnull()) & (c.get(k) != 9)]

        # check MCQ160F
        # exclude non-numeric values
        d = c.select_dtypes(['number'])

        # exclue columns that have over 50% NaN
        d = d.dropna(thresh=0.5 * len(d), axis=1)

        d[k] = d.apply(lambda x: 1 if x.get(k) == 1 else 0, axis='columns')



        from sklearn.impute import SimpleImputer

        imp_mode = SimpleImputer(strategy='most_frequent')

        d = pd.DataFrame(imp_mode.fit_transform(d), columns=d.columns)
        X = d.loc[:, d.columns != k]
        y = d.get(k)

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

        from imblearn.over_sampling import SMOTE, RandomOverSampler

        try:
            smote = SMOTE()
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
            X_test_sm, y_test_sm = smote.fit_resample(X_test, y_test)
        except:
            smote = RandomOverSampler()
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
            X_test_sm, y_test_sm = smote.fit_resample(X_test, y_test)

        y_pred_sm = model.predict(X_test_sm)

        accuracy = accuracy_score(y_test_sm, y_pred_sm)

        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        conf = pd.DataFrame(confusion_matrix(y_test_sm, y_pred_sm), index=['True[0]', 'True[1]'],
                            columns=['Predict[0]', 'Predict[1]'])
        print(conf)

        print()
        print()

else:
    print("Training model " + model_exists + ' times')
    for _ in range(int(model_exists)):
        for k in conditions:
            print(k + ' ' + conditions.get(k))
            df = {}
            dfn = dict(zip(dfname, path))
            df = {key: pd.read_csv(value) for key, value in dfn.items()}

            Xs = {k: v for k, v in df.items() if k in ['dm']}

            dfs = Xs.values()

            from functools import partial, reduce

            inner_merge = partial(pd.merge, how='inner', on='SEQN')

            c = reduce(inner_merge, dfs)

            # check if there are duplicated SEQN

            qs = df['qs'][['SEQN', k]]
            c = pd.merge(c, qs, how='left', on='SEQN')

            # MCQ160F (target feature): exclude null values and NA
            c = c[(c.get(k).notnull()) & (c.get(k) != 9)]

            # check MCQ160F
            # exclude non-numeric values
            d = c.select_dtypes(['number'])

            # exclue columns that have over 50% NaN
            d = d.dropna(thresh=0.5 * len(d), axis=1)

            d[k] = d.apply(lambda x: 1 if x.get(k) == 1 else 0, axis='columns')

            from sklearn.impute import SimpleImputer

            imp_mode = SimpleImputer(strategy='most_frequent')

            d = pd.DataFrame(imp_mode.fit_transform(d), columns=d.columns)
            X = d.loc[:, d.columns != k]
            y = d.get(k)

            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

            from imblearn.over_sampling import SMOTE, RandomOverSampler




            try:
                smote = SMOTE()
                X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
                X_test_sm, y_test_sm = smote.fit_resample(X_test, y_test)
            except:
                smote = RandomOverSampler()
                X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
                X_test_sm, y_test_sm = smote.fit_resample(X_test, y_test)

            X_train_sm = pd.DataFrame(X_train_sm, columns=X.columns)
            X_test_sm = pd.DataFrame(X_test_sm, columns=X.columns)

            model = DecisionTreeClassifier()
            model.fit(X_train_sm, y_train_sm)
            y_pred_sm = model.predict(X_test_sm)

            accuracy = accuracy_score(y_test_sm, y_pred_sm)

            print("Accuracy: %.2f%%" % (accuracy * 100.0))

            if accuracy > best:
                best = accuracy
                with open('model.pickle', 'wb') as file:
                    pickle.dump(model, file)

            conf = pd.DataFrame(confusion_matrix(y_test_sm, y_pred_sm), index=['True[0]', 'True[1]'],
                                columns=['Predict[0]', 'Predict[1]'])
            print(conf)

            print()
            print()
