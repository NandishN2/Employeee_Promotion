Employee Promotion Prediction & Performance Dashboard
Project Overview

This project is a Django web application that predicts employee promotions using a Naive Bayes classifier and displays a Performance Dashboard for HR insights. It also allows HR personnel to search for individual employees and explore their performance metrics.

Features

Promotion Prediction

Predicts whether an employee is likely to be promoted (Yes / No) based on:

Department

Region

Education

Gender

Recruitment channel

Number of trainings

Age

Previous year rating

Length of service

Awards won

Average training score

Performance Dashboard

Displays top performers with their scores.

Shows total employees, total promotions, average training score, and model accuracy in %.

Real-time simulation of employee performance updates (optional).

Employee Search

Search employees by employee_id.

Display detailed performance metrics.

Data Handling

Handles missing values:

Numeric columns → Filled with mean

Categorical columns → Filled with mode

Encodes categorical variables using explicit mappings.

Splits dataset into training/testing sets (80/20).

Saves trained model using joblib.

Technology Stack

Backend: Django, Python

Machine Learning: scikit-learn (Gaussian Naive Bayes)

Data Handling: pandas

Frontend: HTML, CSS (responsive, animated dashboard)

Model Persistence: joblib

File Structure
employee_promotion/
├── manage.py
├── employee_promotion/
│   └── settings.py
├── prediction/
│   ├── data/
│   │   └── employee_data.csv
│   ├── migrations/
│   ├── templates/prediction_1/
│   │   ├── home.html
│   │   ├── prediction.html
│   │   └── employee-search.html
│   └── views.py
├── static/
│   └── css/
│       └── style.css
└── naive_bayes_model.pkl

Setup & Installation

Clone the repository:

git clone <repo_url>
cd employee_promotion


Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


Install required packages:

pip install -r requirements.txt


Required packages: Django, pandas, scikit-learn, joblib

Run migrations:

python manage.py migrate


Start the Django server:

python manage.py runserver


Access the dashboard at:

http://127.0.0.1:8000/

Usage

Home/Dashboard

View total employees, promotions, average training score, and model accuracy.

Check top 10 performers with scores and awards.

Promotion Prediction

Navigate to the prediction page using the "Predict Now" button.

Fill employee details and click "Predict".

Get the predicted promotion status (Yes / No).

Employee Search

Navigate using the "Explore Performance" button.

Enter employee_id to view individual performance metrics.

Model Accuracy

The model uses Gaussian Naive Bayes.

Trained on 80% of the dataset and tested on 20%.

Accuracy is displayed on the dashboard in percentage.

Notes

The application automatically handles missing data.

Categorical values are mapped to integers using predefined dictionaries.

The dashboard uses CSS animations for a modern look.

You can simulate real-time updates for employee performance on the dashboard.

Screenshots

(You can include screenshots of your dashboard and prediction page here)

Future Improvements

Integrate live database instead of CSV.

Use more advanced ML algorithms (Random Forest, XGBoost) to improve accuracy.

Add charts and graphs for employee analytics.

Allow admin login and management of employee data.
