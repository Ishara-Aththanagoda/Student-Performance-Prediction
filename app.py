

from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model 
model = joblib.load("student_performance_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    import pandas as pd

    
    expected_data = {
        "school": "GP",
        "sex": "F",
        "age": 16,
        "address": "U",
        "famsize": "GT3",
        "Parrent_status": "T",
        "Mother_edu": 3,
        "Father_edu": 4,
        "Mother_job": "at_home",
        "Father_job": "teacher",
        "reason_to_chose_school": "course",
        "guardian": "mother",
        "traveltime": 1,
        "weekly_studytime": 2,
        "failures": 0,
        "extra_edu_supp": "no",
        "family_edu_supp": "yes",
        "extra_paid_class": "no",
        "extra_curr_activities": "yes",
        "nursery": "yes",
        "Interested_in_higher_edu": "yes",
        "internet_access": "yes",
        "romantic_relationship": "no",
        "Family_quality_reln": 5,
        "freetime_after_school": 3,
        "goout_with_friends": 3,
        "workday_alcohol_consum": 1,
        "weekend_alcohol_consum": 1,
        "health_status": 3,
        "absences": 4,
        "G1": 15,  # Provided by user (override default if available)
        "G2": 15   # Provided by user (override default if available)
    }
    
    
    form_data = request.form.to_dict()
    for key, value in form_data.items():
        if key in expected_data:
            expected_data[key] = value

    
    input_df = pd.DataFrame([expected_data])
    
    
    input_df["age"] = input_df["age"].astype(int)
    input_df["Mother_edu"] = input_df["Mother_edu"].astype(int)
    input_df["Father_edu"] = input_df["Father_edu"].astype(int)
    input_df["traveltime"] = input_df["traveltime"].astype(int)
    input_df["weekly_studytime"] = input_df["weekly_studytime"].astype(int)
    input_df["failures"] = input_df["failures"].astype(int)
    input_df["Family_quality_reln"] = input_df["Family_quality_reln"].astype(int)
    input_df["freetime_after_school"] = input_df["freetime_after_school"].astype(int)
    input_df["goout_with_friends"] = input_df["goout_with_friends"].astype(int)
    input_df["workday_alcohol_consum"] = input_df["workday_alcohol_consum"].astype(int)
    input_df["weekend_alcohol_consum"] = input_df["weekend_alcohol_consum"].astype(int)
    input_df["health_status"] = input_df["health_status"].astype(int)
    input_df["absences"] = input_df["absences"].astype(int)
    input_df["G1"] = input_df["G1"].astype(float)
    input_df["G2"] = input_df["G2"].astype(float)
    
    
    prediction = model.predict(input_df)
    result = round(prediction[0], 2)
    return render_template('index.html', prediction_text=f"Predicted Final Grade (G3): {result}")

if __name__ == '__main__':
    app.run(debug=True)
