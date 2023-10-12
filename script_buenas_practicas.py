import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split


def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    data = data[data['gender'] != 'Unknown/Invalid']
    data = data[data['race'] != '?']
    data.replace('abcde', pd.NA, inplace=True)
    data = data.drop_duplicates(keep='first')
    data = data[data['glimepiride-pioglitazone'] != 'Steady']
    data = data[data['acetohexamide'] != 'Steady']
    return data


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    '''Returns the dataset after the process of getting the features and dropping
    not important columns'''
    data = data.drop(columns=['patient_nbr', 'encounter_id', 'out', 'payer_code', 'payer_code_2',
                              'name', 'metformin-rosiglitazone', 'examide', 'citoglipton', 'US',
                              'metformin-pioglitazone', 'glimepiride-pioglitazone',
                              'medical_specialty', 'troglitazone', 'troglitazone_2', 
                              'weight', 'metformin-pioglitazone_2', 'acetohexamide',
                              'glyburide-metformin_2'])
    
    data = data.dropna(axis=0)

    # encode meds
    meds_to_encode = ['glyburide', 'miglitol', 'glipizide',  'metformin', 'acarbose', 
                      'chlorpropamide', 'tolazamide', 'glipizide-metformin', 'glimepiride',
                      'pioglitazone', 'rosiglitazone', 'insulin', 'repaglinide',
                      'nateglinide', 'glyburide-metformin', 'glyburide-metformin']

    for med in meds_to_encode:
        data[med] = data[med].apply(lambda row: get_encoding_med(row))

    # diag encoding
    for diag in ['diag_1', 'diag_2', 'diag_3']:
        data[diag] = data[diag].apply(lambda row: get_clean_diag(row))
     
    # age encoding for sci kit learn
    def get_age_clean(value) -> int:
        'return the age as a number format'
        return value[-3:-1]


    data['age'] = data['age'].apply(lambda row: get_age_clean(row))

    # standarize values
    numeric_columns = ['number_inpatient', 'number_diagnoses', 'time_in_hospital', 
                       'num_lab_procedures','num_medications', 'number_emergency', 
                       'num_procedures', 'number_outpatient', 'admission_source_id', 
                       'admission_type_id', 'discharge_disposition_id', 'time_in_hospital']
    for column in numeric_columns:
        data[column] = get_standarized_values(data[column])

    # make dummie variables
    data = pd.get_dummies(data, columns=['diag_1', 'diag_2', 'diag_3', 'race', 'gender', 
                                         'max_glu_serum', 'A1Cresult', 'change', 
                                        'diabetesMed', 'tolbutamide'],
                        drop_first=True)
    
    # make the target variable binary
    data['readmitted_binary'] = (data['readmitted'] == '<30').astype(int)
    return data


def get_encoding_med(value: str) -> int:
        '''Returns an ascending encoder for the dosis in meds'''
        encoder = {
            'No': 0,
            'Steady': 1,
            'Down': 2,
            'Up': 3,
        }
        return encoder.get(value, 4)


def get_clean_diag(value) -> int:
        '''Returns the encoding of the diagnostics'''
        if value is pd.NA or  value == '?':
            return '0'
        elif value[0] == 'E':
            trunc = float(value[1:]) // 200
            return 'E' + str(trunc)
        elif value[0] == 'V':
            trunc = float(value[1:]) // 20
            return 'V' + str(trunc)
        else:
            trunc = float(value) // 200
            return str(trunc)


def get_standarized_values(values: pd.Series) -> pd.Series:
        '''Returns normalized values'''
        values = values.astype(float)
        mean = values.mean()
        std = values.std()
        normalized_values = (values - mean) / std
        return normalized_values


def split_data(data:pd.DataFrame) -> pd.DataFrame:
        X = data.drop(['readmitted', 'readmitted_binary'], axis=1)
        y = data['readmitted_binary']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test


def modelo_predictivo(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def feature_selection(model, X, y):
    sfm = SelectFromModel(model, threshold=-np.inf, max_features=30) 
    sfm.fit(X, y)
    X_selected = sfm.transform(X)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, 
                                                        y, 
                                                        test_size=0.2, 
                                                        random_state=42)

    # Train your model on the selected features
    model_selected = RandomForestClassifier(random_state=42)
    model_selected.fit(X_train, y_train)
    return model_selected, X_test


def model_eval(model_selected, X_test, y_test) -> None:
    # Make predictions on the test set
    y_pred = model_selected.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f'accuracy = {accuracy}')

    f1 = f1_score(y_test, y_pred)
    print(f'f1 score {f1}')

    # ROC Curve
    y_prediction = model_selected.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prediction)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show() 


if __name__ == "__main__":
    # data
    data = pd.read_csv('diabetes_data/diabetic_data.csv')
    data = data_cleaning(data)
    data = feature_engineering(data)
    
    # model
    X = data.drop(['readmitted', 'readmitted_binary'], axis=1)
    y = data['readmitted_binary']
    X_train, X_test, y_train, y_test = split_data(data)
    model = modelo_predictivo(X_train, y_train)
    model_selected, X_test = feature_selection(model, X, y)

    #results
    model_eval(model_selected, X_test, y_test)
