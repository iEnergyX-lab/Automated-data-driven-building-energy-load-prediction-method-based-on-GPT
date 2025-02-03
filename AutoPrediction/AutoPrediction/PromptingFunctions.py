def PromptingFunctionForModelTraining(X):
    """Prompting function of model training.
    This prompting function is utilized to generate a prompt of model training.
    
    Inputs
    ----------
    X: Input list for the prompting function. There are 19 elements in X:
      X[0]: Name of a file that stores historical data.
      X[1]: Available variable list in historical data.
      X[2]: Building energy load that needs to be predicted.
      X[3]: Missing data handling method.
      X[4]: Name list of hyper-parameters of the missing data handling method.
      X[5]: Value list of hyper-parameters of the missing data handling method.
      X[6]: Outlier identification method.
      X[7]: Name list of hyper-parameters of the outlier identification method.
      X[8]: Value list of hyper-parameters of the outlier identification method.
      X[9]: Data normalization method.
      X[10]: Name list of hyper-parameters of the data normalization method.
      X[11]: Value list of hyper-parameters of the data normalization method.
      X[12]: Feature engineering method
      X[13]: Name list of hyper-parameters of the feature engineering method.
      X[14]: Value list of hyper-parameters of the feature engineering method.
      X[15]: Data-driven model.
      X[16]: Name list of hyper-parameters of the data-driven model.
      X[17]: Value list of hyper-parameters of the data-driven model.
      X[18]: Model evaluation metric.

    Outputs
    ----------
    Prompt: Prompt of model training.
    
    """
    
    File_name = X[0]
    Available_variables = PromptingFunctionForAvailableVariables(X[1])
    Model_output = X[2]
    Missing_data_handling = PromptingFunctionForMissingDataHandling(X[3], X[4], X[5])
    Outlier_identification = PromptingFunctionForOutlierIdentification(X[3], X[6], X[7], X[8])
    Data_normalization = PromptingFunctionForDataNormalization(X[9], X[10], X[11], Usage = "model_training")
    Feature_engineering = PromptingFunctionForFeatureEngineering(X[12], X[13], X[14])
    Data_denormalization = PromptingFunctionForDataDenormalization(X[9], Usage = "model_training")
    Data_driven_model = PromptingFunctionForDataDrivenModel(X[15], X[16], X[17])
    Accuracy_metric = X[18]
    Rules = PromptingFunctionForExternalKnowledge(X[9], X[12], X[15], Usage = "model_training")
    Code = ExampleCodeForModelTraining(X)
    
    Prompt = f"""You are a programming expert in data-driven building energy load prediction. Please provide Python codes according to my requirements.

My requirements are described as follows:
I have collected the operational data of a building. These data are stored in a file named "{File_name}". The available variables in this file include {Available_variables}. I want to train a data-driven model for one-step ahead {Model_output} prediction of this building.
The following steps should be considered to train this model:
Step 1. Handle the missing data using the {Missing_data_handling}{Outlier_identification}{Data_normalization}
Step 2. Select or extract suitable model inputs using the {Feature_engineering}
Step 3. Divide the data of the model inputs and output into a training set (70%) and testing set (30%) randomly.
Step 4. Apply the training set to train a data-driven one-step ahead {Model_output} prediction model using the {Data_driven_model}
Step 5. {Data_denormalization}Calculate the {Accuracy_metric} of the data-driven model on the testing set. The {Accuracy_metric} should be assigned to the variable named "model_accuracy".

Please observe the following rules when writing the Python codes:{Rules}

There is an example code for building energy load prediction:{Code}
"""

    return Prompt

def PromptingFunctionForModelDeployment(X):
    """Prompting function of model deployment.
    This prompting function is utilized to generate a prompt of model deployment.
    
    Inputs
    ----------
    X: The input list for the prompting function. There are 5 elements in X:
      X[0]: Name of a file that stores historical data.
      X[1]: Name of a file that stores real-time data.
      X[2]: Building energy load that needs to be predicted.
      X[3]: Data normalization method.
      X[4]: Model interpretation method.

    Outputs
    ----------
    Prompt: Prompt of model deployment.
    
    """    
    
    File_name_1 = X[0]
    File_name_2 = X[1]
    Model_output = X[2]
    Data_normalization = PromptingFunctionForDataNormalization(X[3], None, None, Usage = "model_deployment")
    Data_denormalization =  PromptingFunctionForDataDenormalization(X[3], Usage = "model_deployment")
    Rules = PromptingFunctionForExternalKnowledge(None, None, None, Usage = "model_deployment")
    Model_interpretation = X[4]
    
    Prompt = f"""The operational data of this building are collected in real time. The upcoming data will be stored in the file named "{File_name_2}". This file has the same available variables as the file named "{File_name_1}". You have provided a Python code that can utilize the data in "{File_name_1}" to train a one-step ahead {Model_output} prediction model. Please modify this code so that the trained model can be utilized for one-step ahead {Model_output} prediction in real time. The new code should include the following stages:

Stage 1 aims to train a model using the data from "{File_name_1}" based on the steps in the previous code.

Stage 2 aims to apply the model trained in the stage 1 for one-step ahead {Model_output} prediction based on the data from "{File_name_2}". {Data_normalization}The same model inputs as the stage 1 should be utilized. The model trained in the stage 1 should be utilized for {Model_output} prediction. All the predicted {Model_output} should be {Data_denormalization}assigned to the variable named "predicted_load".

Stage 3 aims to apply the {Model_interpretation} method to explain the model trained in the stage 1. The global importance of every model input on the historical data from "{File_name_1}" should be estimated and visualized.

Please observe the following rules when writing the Python codes:{Rules}
"""

    return Prompt

def PromptingFunctionForMissingDataHandling(Missing_data_handling_method, Hyperparameter_names, Hyperparameter_values):
    """Prompting sub-function of missing data handling.
    This prompting sub-function is utilized to generate a prompt of missing data handling.
    
    Inputs
    ----------
    Missing_data_handling_method: Missing data handling method.
    Hyperparameter_names: Name list of hyper-parameters of the missing data handling method.
    Hyperparameter_values: Value list of hyper-parameters of the missing data handling method.

    Outputs
    ----------
    Prompt: Prompt of missing data handling.
    
    """
    
    if len(Hyperparameter_values) == 0: #The missing data handling method doesn't have hyper-parameters.
        Prompt = f"{Missing_data_handling_method}. "
        
    else: #The missing data handling method has hyper-parameters.
        Missing_data_handling_hyperparameters = PromptingFunctionForHyperparameters(Missing_data_handling_method, Hyperparameter_names, Hyperparameter_values)
        Prompt = f"{Missing_data_handling_method}. {Missing_data_handling_hyperparameters}"
        
    return Prompt
        
def PromptingFunctionForAvailableVariables(Variable_list):
    """Prompting sub-function for describing available variables.
    This prompting sub-function is utilized to generate a prompt for describing available variables.
    
    Inputs
    ----------
    Variable_list: Available variable list in historical data.
    
    Outputs
    ----------
    Prompt: The prompt for describing available variables.
    
    """
    
    Prompt = ""
    
    if len(Variable_list) == 1:
        Varaible_name = Variable_list[0]
        Prompt = Prompt+f'"{Varaible_name}"'
        
    else:
        for i in range(len(Variable_list)):
            if i != len(Variable_list)-1:
                Varaible_name = Variable_list[i]
                Prompt = Prompt+f'"{Varaible_name}", '
            else:
                Varaible_name = Variable_list[i]
                Prompt = Prompt+f'"{Varaible_name}"'
                
    return Prompt
        
def PromptingFunctionForOutlierIdentification(Missing_data_handling_method, Outlier_identification_method, Hyperparameter_names, Hyperparameter_values):
    """Prompting sub-function of outlier identification.
    This prompting sub-function is utilized to generate a prompt of outlier identification.
    
    Inputs
    ----------
    Missing_data_handling_method: Missing data handling method.
    Outlier_identification_method: Outlier identification method.
    Hyperparameter_names: Name list of hyper-parameters of the outlier identification method.
    Hyperparameter_values: Value list of hyper-parameters of the outlier identification method.

    Outputs
    ----------
    Prompt: Prompt of outlier identification.
    
    """

    if Outlier_identification_method == "none": #Outlier identification is unnecessary.
        Prompt = ""
    
    elif len(Hyperparameter_values) == 0: #The outlier identification method doesn't have hyper-parameters.
        Prompt = f"Identify the outliers using the {Outlier_identification_method}. The outliers should be handled using the {Missing_data_handling_method}. "        
    
    else: #The outlier identification method has hyper-parameters.
        Outlier_identification_hyperparameters = PromptingFunctionForHyperparameters(Outlier_identification_method, Hyperparameter_names, Hyperparameter_values)
        Prompt = f"Identify the outliers using the {Outlier_identification_method}. {Outlier_identification_hyperparameters}The outliers should be handled using the {Missing_data_handling_method}. "
    
    return Prompt

def PromptingFunctionForDataNormalization(Data_normalization_method, Hyperparameter_names, Hyperparameter_values, Usage = "model_training"):
    """Prompting sub-function of data normalization.
    This prompting sub-function is utilized to generate a prompt of data normalization.
    
    Inputs
    ----------
    Data_normalization_method: Data normalization method.
    Hyperparameter_names: Name list of hyper-parameters of the data normalization method.
    Hyperparameter_values: Value list of hyper-parameters of the data normalization method.
    Usage: Purpose of using this function. There are two options for this input: "model_training" and "model_deployment". "model_training" means that this function is utilized for model training, and "model_deployment" means that this function is utilized for model deployment.

    Outputs
    ----------
    Prompt: Prompt for data normalization.
    
    """
    
    if Usage == "model_training":
        if Data_normalization_method == "none": #Data normalization is unnecessary. 
            Prompt = ""
        
        elif len(Hyperparameter_values) == 0: #The data normalization method doesn't have hyper-parameters. 
            Prompt = f"Normalize the target variable (model output) and other variables using the {Data_normalization_method}."

        else: #The data normalization method has hyper-parameters. 
            Data_normalization_hyperparameters = PromptingFunctionForHyperparameters(Data_normalization_method, Hyperparameter_names, Hyperparameter_values)
            Prompt = f"Normalize the target variable (model output) and other variables using the {Data_normalization_method}. {Data_normalization_hyperparameters}"
    
    if Usage == "model_deployment":
        if Data_normalization_method == "none": #Data normalization is unnecessary. 
            Prompt = "Data normalization is unnecessary. "
            
        else: #Data normalization is necessary. 
            Prompt = "The same data normalization method as the stage 1 should be utilized to normalize the real-time data. "
    
    return Prompt

def PromptingFunctionForFeatureEngineering(Feature_engineering_method, Hyperparameter_names, Hyperparameter_values):
    """Prompting sub-function of feature engineering.
    This prompting sub-function is utilized to generate a prompt of feature engineering.
    
    Inputs
    ----------
    Feature_engineering_method: Feature engineering method.
    Hyperparameter_names: Name list of hyper-parameters of the feature engineering method.
    Hyperparameter_values: Value list of hyper-parameters of the feature engineering method.
    
    Outputs
    ----------
    Prompt: Prompt of feature engineering.
    
    """

    if len(Hyperparameter_values) == 0: #The feature engineering method doesn't have hyper-parameters.
        Prompt = f"{Feature_engineering_method}."        
    
    else: #The outlier identification method has hyper-parameters.
        Feature_engineering_hyperparameters = PromptingFunctionForHyperparameters(Feature_engineering_method, Hyperparameter_names, Hyperparameter_values)
        Prompt = f"{Feature_engineering_method}. {Feature_engineering_hyperparameters}"
    
    return Prompt    

def PromptingFunctionForDataDenormalization(Data_normalization_method, Usage = "model_training"):
    """Prompting sub-function of data denormalization.
    This prompting sub-function is utilized to generate a prompt of data denormalization.
    
    Inputs
    ----------
    Data_normalization_method: Data normalization method.
    Usage: Purpose of using this function. There are two options for this input: "model_training" and "model_deployment". "model_training" means that this function is utilized for model training, and "model_deployment" means that this function is utilized for model deployment.

    Outputs
    ----------
    Prompt: Prompt of data denormalization.
    
    """    
    
    if Usage == "model_training":
        if Data_normalization_method == "none": #Data normalization is unnecessary. 
            Prompt = ""
            
        else: #Data normalization is necessary. 
            Prompt = "Denormalize the actual and predicted model outputs on the testing set. "
            
    if Usage == "model_deployment":
        if Data_normalization_method == "none": #Data normalization is unnecessary. 
            Prompt = ""
            
        else: #Data normalization is necessary. 
            Prompt = "denormalized and "
            
    return Prompt

def PromptingFunctionForDataDrivenModel(Data_driven_model, Hyperparameter_names, Hyperparameter_values):
    """Prompting sub-function for describing data-driven models.
    This prompting sub-function is utilized to generate a prompt for describing data-driven models.
    
    Inputs
    ----------
    Data_driven_model: Data-driven model.
    Hyperparameter_names: Name list of hyper-parameters of the data-driven model.
    Hyperparameter_values: Value list of hyper-parameters of the data-driven model.

    Outputs
    ----------
    Prompt: Prompt for describing data-driven models.
    
    """

    if len(Hyperparameter_values) == 0: #The feature engineering method doesn't have hyper-parameters.
        Prompt = f"{Data_driven_model}."        
    
    else: #The outlier identification method has hyper-parameters.
        Data_driven_model_hyperparameters = PromptingFunctionForHyperparameters(Data_driven_model, Hyperparameter_names, Hyperparameter_values)
        Prompt = f"{Data_driven_model}. {Data_driven_model_hyperparameters}"
    
    return Prompt    
    
def ExampleCodeForModelTraining(X):
    """Prompting function for generating an example code.
    This prompting function is utilized to generate an example code.
    
    Inputs
    ----------
    X: All inputs of the model-training prompting function.

    Outputs
    ----------
    Prompt: Prompt for describing an example code.
    
    """
    
    File_name = X[0]
    Target_variable = X[1][-1]
    
    Code = '''\nimport pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import IsolationForest

# Load the data from CSV file
data = pd.read_csv("%s")

# Step 1: Forward filling missing data
data.fillna(method="ffill", inplace=True)

target_col = "%s"
input_cols = [col for col in data.columns if col != target_col]

# Handle outliers using isolation forest
outlier_detector = IsolationForest(contamination=0.02, random_state=1)
outlier_detector.fit(data)

outliers = outlier_detector.predict(data)
data[outliers == -1] = np.nan
data.fillna(method="ffill", inplace=True)

# Normalize input variables using z-score normalization
input_scaler = StandardScaler()
data[input_cols] = input_scaler.fit_transform(data[input_cols])

# Normalize target variable
target_scaler = StandardScaler()
data[target_col] = target_scaler.fit_transform(data[[target_col]])

# Step 2: Select model inputs using correlation analysis
correlation_threshold = 0.5
correlation_matrix = np.abs(data.corr(method='spearman'))
selected_input_cols = correlation_matrix[correlation_matrix[target_col] > correlation_threshold].index.tolist()
selected_input_cols.remove(target_col)

# Step 3: Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.3, random_state=1)

# Step 4: Train XGBoost model
xgb_params = {
    'n_estimators': 10,
    'learning_rate': 1,
    'random_state': 1
}
xgb_model = XGBRegressor(**xgb_params)
xgb_model.fit(train_data[selected_input_cols], train_data[target_col])

# Step 5: Predict and evaluate the model
test_predictions = xgb_model.predict(test_data[selected_input_cols])
test_predictions = target_scaler.inverse_transform(test_predictions.reshape(-1, 1))
test_actual = target_scaler.inverse_transform(test_data[target_col].values.reshape(-1, 1))

model_accuracy = r2_score(test_actual, test_predictions)'''%(File_name, Target_variable) 

    return Code

def PromptingFunctionForExternalKnowledge(Data_normalization_method, Feature_engineering_method, Data_driven_model, Usage = "model_training"):
    """Prompting sub-function for describing external knowledge.
    This prompting sub-function is utilized to generate a prompt for describing external knowledge.
    
    Inputs
    ----------
    Data_normalization_method: Data normalization method.
    Feature_engineering_method: Feature engineering method.
    Data_driven_model: Data-driven model.
    Usage: Purpose of using this function. There are two options for this input: "model_training" and "model_deployment". "model_training" means that this function is utilized for model training, and "model_deployment" means that this function is utilized for model deployment.

    Outputs
    ----------
    Prompt: Prompt for describing Data_driven_model.
    
    """ 
    
    if Usage == "model_training":
        Prompt = """\n1. Please provide only one complete Python code in your answer. The code will run in Python interpreters.
2. Please set the random state to 1 for the methods with randomness.
3. The target variable (model output) cannot be utilized as the model inputs."""

        m = 4
        if Data_normalization_method != "none": #Data normalization is unnecessary.
            Prompt = Prompt+f"\n{m}. The target variables (model output) and other variables should be normalized by two different scalers. The scaler for the target variable should be utilized to denormalize the actual and predicted model outputs on the testing set."
            m += 1
        
        if Feature_engineering_method == "domain knowledge-based method": #Domain knowledge-based feature engineering method is utilized.
            Prompt = Prompt+f"\n{m}. The domain knowledge-based method usually select the month, hour of the day, day of the week, outdoor air temperature, outdoor air relative humidity, outdoor solar radiation, loads 1-3 hours ago and load 24 hours ago as the inputs of data-driven models."
            m += 1
        
        if Feature_engineering_method == "correlation analysis": #Correlation analysis-based feature engineering method is utilized.
            Prompt = Prompt+f"\n{m}. Correlation analysis calculates the correlation coefficient between every feature and the model output. A feature will be selected as a model input if the absolute value of the correlation between it and the model output is higher than the correlation threshold."
            m += 1
            Prompt = Prompt+f"\n{m}. The Spearman correlation coefficient can be calculated using scipy.stats.spearmanr for correlation analysis. The Pearson correlation coefficient can be calculated using scipy.stats.pearsonr for correlation analysis."
            m += 1
            
        if Data_driven_model == "artificial neural networks": #Artificial neural networks model is utilized.
            Prompt = Prompt+f"\n{m}. The Python package named Scikit-learn should be utilized to train the artificial neural networks model."
            m += 1       

    if Usage == "model_deployment":
        Prompt = """\n1. Please provide only one complete Python code in your answer. The code will run in Python interpreters.
2. The variable "predicted_load" should be a one-dimensional vector.
3. Every step except data splitting should be utilized for model training in the stage 1."""
                
    return Prompt

def PromptingFunctionForSelfCorrection(Error_message):
    """Prompting function of code correction.
    This prompting function is utilized to generate a prompt of code correction.
    
    Inputs
    ----------
    Error_message: Error message of incorrect codes.
        
    Outputs
    ----------
    Prompt: Prompt of code correction.
    
    """
    
    Prompt = f'''Something goes wrong when I run your Python code. Please modify the incorrect Python code. The complete Python code after correction should be provided. 
The error messages of the incorrect Python code are listed as follows:
{Error_message}
'''

    return Prompt

def PromptingFunctionForHyperparameters(Method_name, Hyperparameter_names, Hyperparameter_values):
    """Prompting sub-function for describing hyperparameters.
    This prompting sub-function is utilized to generate a prompt for describing hyperparameters.
    
    Inputs
    ----------
    Method_name: Name of the method with hyperparameters.
    Hyperparameter_names: Name list of hyper-parameters of the method.
    Hyperparameter_values: Value list of hyper-parameters of the method.
        
    Outputs
    ----------
    Prompt: Prompt for describing hyperparameters.
    
    """
    
    Prompt = ""
    
    for i in range(len(Hyperparameter_names)):
        Hyperparameter_name = Hyperparameter_names[i]
        Hyperparameter_value = Hyperparameter_values[i]
        Prompt = Prompt+f"The {Hyperparameter_name} of the {Method_name} should be set to {Hyperparameter_value}. "
        
    return Prompt