import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__) # defining flask app

pickled_model = pickle.load(open('Random_forest.pickle','rb'))  # loading the randomforest model


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    age = float(request.form.get('age', False))
    sex = float(request.form.get('sex', False))
    T3 = float(request.form.get('T3', False))
    TT4 = float(request.form.get('TT4', False))
    FTI = float(request.form.get('FTI', False))
    onthyroxine = float(request.form.get('onthyroxine', False))
    sick = float(request.form.get('sick', False))
    queryhypothyroid = float(request.form.get('queryhypothyroid', False))
    psych = float(request.form.get('psych', False))

    # ['age', 'sex', 'on_thyroxine', 'sick', 'query_hypothyroid', 'psych',
    #  'T3', 'TT4', 'FTI']

    # values = ({"age": age, "sex":sex, "on_thyroxine": on_thyroxine,
    #            "query_on_thyroxine":query_on_thyroxine, "on_antithyroid_medication":on_antithyroid_medication,
    #            "sick": sick, "pregnant": pregnant, "thyroid_surgery": thyroid_surgery, "I131_treatment": I131_treatment,
    #            "query_hypothyroid":query_hypothyroid, "query_hyperthyroid":query_hyperthyroid, "lithium":lithium,
    #             "goitre":goitre, "tumor":tumor, "hypopituitary":hypopituitary, "psych":psych, "T3":T3,
    #            "TT4":TT4, "T4U":T4U, "FTI":FTI})


    values = ({"age": age, "sex": sex,
               "T3": T3, "TT4": TT4, "FTI": FTI,
               "onthyroxine": onthyroxine, "sick": sick,
               "queryhypothyroid": queryhypothyroid,
               "psych": psych})


    df_transform = pd.DataFrame.from_dict([values])

    # print("applying transformation\n")

    df_transform.age = df_transform['age'] ** (1 / 2)
    print(df_transform.age)

    #
    df_transform.T3 = df_transform['T3'] ** (1 / 2)
    # print(df_transform.T3)

    df_transform.T4U = np.log1p(df_transform['TT4'])
    # print(df_transform.T4U)

    df_transform.FTI = df_transform['FTI'] ** (1 / 2)
    # print(df_transform.FTI)

    df_transform.to_dict()

    # ['age', 'sex', 'on_thyroxine', 'sick', 'query_hypothyroid', 'psych',
    #  'T3', 'TT4', 'FTI']

    arr = np.asarray([[df_transform.age, sex,
                     df_transform.T3, df_transform.TT4, df_transform.FTI,
                     onthyroxine, sick, queryhypothyroid, psych]]).reshape(1,-1)


    pred = pickled_model.predict(arr)[0]


    if pred == 0:
        res_val = ('compensated_hypothyroid')
    elif pred == 1:
        res_val = ('negative')
    elif pred == 2:
        res_val = ('primary_hypothyroid')
    else:
        res_val = ('secondary_hypothyroid')


    return render_template('index.html', prediction_text='Result: {}'.format(res_val))


if __name__ == '__main__':
    app.run(debug=True)