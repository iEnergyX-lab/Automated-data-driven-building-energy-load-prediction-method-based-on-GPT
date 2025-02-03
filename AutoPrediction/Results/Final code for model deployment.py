import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import shap

# Stage 1: Train a model using historical operational data
data_historical = pd.read_csv("Data/Historical_operational_data.csv")
data_historical.dropna(inplace=True)

model_inputs = ["Month", "Hour of the day", "Day of the week", 
                "Outdoor air dry bulb temperature", "Electrical load 1 hour ago", 
                "Electrical load 2 hour ago", "Electrical load 3 hour ago", "Electrical load 24 hour ago"]

train_data = data_historical

xgb_params = {
    'n_estimators': 85,
    'learning_rate': 0.1,
    'random_state': 1
}
xgb_model = XGBRegressor(**xgb_params)
xgb_model.fit(train_data[model_inputs], train_data["Electrical load"])

# Stage 2: Apply the trained model for real-time prediction
data_real_time = pd.read_csv("Data/Real_time_operational_data.csv")
predicted_load = xgb_model.predict(data_real_time[model_inputs])

# Stage 3: Explain the model using Shapley Additive Explanations
explainer = shap.Explainer(xgb_model)
shap_values = explainer(train_data[model_inputs])
shap.summary_plot(shap_values)

predicted_load