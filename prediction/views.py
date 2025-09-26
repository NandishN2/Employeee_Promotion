import os
import pandas as pd
from django.shortcuts import render
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# =======================
# File Paths
# =======================
DATA_PATH = os.path.join(settings.BASE_DIR, 'prediction', 'data', 'employee_data.csv')
MODEL_PATH = os.path.join(settings.BASE_DIR, 'prediction', 'naive_bayes_model.pkl')

# =======================
# Explicit mappings
# =======================
DEPARTMENT_MAP = {
    'Sales & Marketing': 0, 'Operations': 1, 'Technology': 2, 'Analytics': 3,
    'R&D': 4, 'Procurement': 5, 'Finance': 6, 'HR': 7, 'Legal': 8
}

REGION_MAP = {
    'region_7':0,'region_22':1,'region_19':2,'region_23':3,'region_26':4,
    'region_2':5,'region_20':6,'region_34':7,'region_1':8,'region_4':9,
    'region_29':10,'region_31':11,'region_15':12,'region_14':13,'region_11':14,
    'region_5':15,'region_28':16,'region_17':17,'region_13':18,'region_16':19,
    'region_25':20,'region_10':21,'region_27':22,'region_30':23,'region_12':24,
    'region_21':25,'region_32':26,'region_6':27,'region_33':28,'region_8':29,
    'region_24':30,'region_3':31,'region_9':32,'region_18':33
}

EDUCATION_MAP = {'Bachelor':0, 'Master':1, 'PhD':2}
GENDER_MAP = {'Male':0, 'Female':1}
RECRUITMENT_CHANNEL_MAP = {'Sourcing':0, 'Referral':1, 'Other':2}


# =======================
# Train Model Function
# =======================
def train_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CSV file not found at {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)

    # Handle missing values
    numeric_cols = ['no_of_trainings','age','previous_year_rating',
                    'length_of_service','awards_won','avg_training_score']
    for col in numeric_cols:
        data[col].fillna(data[col].mean(), inplace=True)

    categorical_cols = ['department','region','education','gender','recruitment_channel']
    for col in categorical_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Encode categorical columns
    data['department'] = data['department'].map(DEPARTMENT_MAP).fillna(-1).astype(int)
    data['region'] = data['region'].map(REGION_MAP).fillna(-1).astype(int)
    data['education'] = data['education'].map(EDUCATION_MAP).fillna(-1).astype(int)
    data['gender'] = data['gender'].map(GENDER_MAP).fillna(-1).astype(int)
    data['recruitment_channel'] = data['recruitment_channel'].map(RECRUITMENT_CHANNEL_MAP).fillna(-1).astype(int)

    # Features and target
    feature_cols = [c for c in data.columns if c not in ['is_promoted', 'employee_id']]
    X = data[feature_cols]
    y = data['is_promoted']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Accuracy and report
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save model
    joblib.dump({'model': model, 'columns': feature_cols}, MODEL_PATH)

    return model, feature_cols, accuracy, report


# =======================
# Prediction View
# =======================
def predict_promotion(request):
    prediction = None

    # Load or train model
    if os.path.exists(MODEL_PATH):
        saved = joblib.load(MODEL_PATH)
        model = saved.get('model')
        feature_columns = saved.get('columns')
        if feature_columns is None or model is None:
            model, feature_columns, _, _ = train_model()
    else:
        model, feature_columns, _, _ = train_model()

    if request.method == 'POST':
        try:
            input_data = {
                'department': DEPARTMENT_MAP.get(request.POST['department'], -1),
                'region': REGION_MAP.get(request.POST['region'], -1),
                'education': EDUCATION_MAP.get(request.POST['education'], -1),
                'gender': GENDER_MAP.get(request.POST['gender'], -1),
                'recruitment_channel': RECRUITMENT_CHANNEL_MAP.get(request.POST['recruitment_channel'], -1),
                'no_of_trainings': int(request.POST.get('no_of_trainings', 0)),
                'age': int(request.POST.get('age', 0)),
                'previous_year_rating': int(request.POST.get('previous_year_rating', 0)),
                'length_of_service': int(request.POST.get('length_of_service', 0)),
                'awards_won': int(request.POST.get('awards_won', 0)),
                'avg_training_score': float(request.POST.get('avg_training_score', 0.0)),
            }

            input_df = pd.DataFrame([input_data], columns=feature_columns)
            pred = model.predict(input_df)[0]
            prediction = 'Yes' if pred == 1 else 'No'

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render(request, 'prediction_1/prediction.html', {'prediction': prediction})


# =======================
# Home View (shows accuracy)
# =======================
def home_view(request):
    model, feature_cols, accuracy, report = train_model()
    return render(request, 'prediction_1/home.html', {
        'accuracy': round(accuracy * 100, 2),  # e.g. 85.23
        'report': report
    })

# =======================
# Employee Search View
# =======================
def employee_search_view(request):
    employee_data = None
    search_value = ''

    if request.method == 'POST':
        search_value = request.POST.get('employeeSearch', '').strip()
        
        df = pd.read_csv(DATA_PATH)
        df.fillna('', inplace=True)

        filtered = df[df['employee_id'].astype(str) == search_value]

        if not filtered.empty:
            filtered = filtered.iloc[:, :-1]  # drop last column if unwanted
            employee_data = filtered.to_dict(orient='records')

    return render(request, 'prediction_1/employee-search.html', {
        'employee_data': employee_data,
        'search_value': search_value
    })

