import optuna
import pandas as pd
import logging
import sys
from optuna.samplers import TPESampler
import pickle
from AutoPrediction import AutoModelTraining
from AutoPrediction import AutoModelDeployment
import os 

'''
Description:
This file is the main program for GPT-based automated building energy load prediction.

The values of the following variables can be changed according to your requirements:
    GPT_model: You can select the GPT model that you want to use.
    organization_id: You need to provide your organization id of GPT.
    api_key: You need to provide your api key of GPT.
    best_accuracy: You need to set a initial accuracy. The initial accuracy should be set to a value that means the accuracy is very poor, so that the accuracy can be improved.
    Historical_data_file_name: You need to provide the name of the file that stores historical data.
    Real_time_data_file_name: You need to provide the name of the file that stores real-time data.
    Target_load: You need to define the building energy load that you want to predict.
    Missing_data_handling_name: Missing data handling method.
    Missing_data_handling_method_hyperparameter_name: Name list of hyper-parameters of the missing data handling method.
    Missing_data_handling_method_hyperparameter_value: Value list of hyper-parameters of the missing data handling method.
    Outlier_identification_method: Outlier identification method.
    Outlier_identification_method_hyperparameter_name: Name list of hyper-parameters of the outlier identification method.
    Outlier_identification_method_hyperparameter_value: Value list of hyper-parameters of the outlier identification method.
    Data_normalization_method: Data normalization method.
    Data_normalization_method_hyperparameter_name: Name list of hyper-parameters of the data normalization method.
    Data_normalization_method_hyperparameter_value: Value list of hyper-parameters of the data normalization method.
    Feature_engineering_method: Feature engineering method.
    Feature_engineering_method_hyperparameter_name: Name list of hyper-parameters of the feature engineering method.
    Feature_engineering_method_hyperparameter_value: Value list of hyper-parameters of the feature engineering method.
    Data_driven_model: Data-driven model.
    Data_driven_model_hyperparameter_name: Name list of hyper-parameters of the data-driven model.
    Data_driven_model_hyperparameter_value: Value list of hyper-parameters of the data-driven model.
    Model_interpretation_method: Model interpretation method.
    Accuracy_metric: Model evaluation metric.
'''

GPT_model = "gpt-3.5-turbo" #GPT model that you want to use.
organization_id = "Your organization id" #Your organization id of GPT.
api_key = "Your api key" #Your api key of GPT.
best_accuracy = 0 #Initial accuracy.

n_trial = 0
best_messages = []
def objective(trial):
    global best_accuracy
    global best_messages
    global n_trial
    
    Historical_data_file_name = "Data/Historical_operational_data.csv" #Name of the file that stores historical data.
    Raw_data = pd.read_csv(Historical_data_file_name)
    Avaiable_variables = list(Raw_data.keys())
    Target_load = "electrical load" #Building energy load that you want to predict.
    
    ######################Define the search space of Baysian optimization######################
    Missing_data_handling_method = trial.suggest_categorical("missing_data_handling", ["listwise deletion", "forward filling", "backward filling", "linear interpolation"]) #Missing data handling method.
    Missing_data_handling_method_hyperparameter_name = [] #Name list of hyper-parameters of the missing data handling method.
    Missing_data_handling_method_hyperparameter_value = [] #Value list of hyper-parameters of the missing data handling method.

    Outlier_identification_method = trial.suggest_categorical("outlier_identification", ["none", "isolation forest"]) #Outlier identification method.
    if Outlier_identification_method == "isolation forest":
        Outlier_identification_method_hyperparameter_name = ["contamination"] #Name list of hyper-parameters of the outlier identification method.
        Outlier_identification_method_hyperparameter_value = [trial.suggest_float("contamination", 0.01, 0.05)] #Value list of hyper-parameters of the outlier identification method.
    else:
        Outlier_identification_method_hyperparameter_name = [] #Name list of hyper-parameters of the outlier identification method.
        Outlier_identification_method_hyperparameter_value = [] #Value list of hyper-parameters of the outlier identification method.
    
    Data_normalization_method = trial.suggest_categorical("data_normalization", ["none", "max-min normalization", "z-score normalization"]) #Data normalization method.
    Data_normalization_method_hyperparameter_name = [] #Name list of hyper-parameters of the data normalization method.
    Data_normalization_method_hyperparameter_value = [] #Value list of hyper-parameters of the data normalization method.

    Feature_engineering_method = trial.suggest_categorical("feature_engineering", ["domain knowledge-based method", "correlation analysis", "principal component analysis"]) #Feature engineering method.       
    if Feature_engineering_method == "correlation analysis":
        Feature_engineering_method_hyperparameter_name = ["correlation coefficient", "coefficient threshold"] #Name list of hyper-parameters of the feature engineering method.
        correlation_coefficient = trial.suggest_categorical("correlation coefficient", ["Pearson correlation", "Spearman correlation coefficient"])    
        coefficient_threshold = trial.suggest_float("coefficient threshold", 0.50, 0.95)
        Feature_engineering_method_hyperparameter_value = [correlation_coefficient, coefficient_threshold] #Value list of hyper-parameters of the feature engineering method.
    elif Feature_engineering_method == "principal component analysis":
        Feature_engineering_method_hyperparameter_name = ["proportion of variance explained"] #Name list of hyper-parameters of the feature engineering method.
        proportion_of_variance_explained = trial.suggest_float("proportion of variance explained", 0.50, 0.99)   
        Feature_engineering_method_hyperparameter_value = [proportion_of_variance_explained] #Value list of hyper-parameters of the feature engineering method.
    else:
        Feature_engineering_method_hyperparameter_name = [] #Name list of hyper-parameters of the feature engineering method.
        Feature_engineering_method_hyperparameter_value = [] #Value list of hyper-parameters of the feature engineering method.
         
    Data_driven_model = trial.suggest_categorical("model", ["multiple linear regression", "support vector regression", "artificial neural networks", "random forests", "extreme gradient boosting"]) #Data-driven model.     
    if Data_driven_model == "support vector regression":
        Data_driven_model_hyperparameter_name = ["C (cost)", "gamma"] #Name list of hyper-parameters of the data-driven model.
        C = trial.suggest_categorical("C (cost)", [2**-10, 2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**10, 2**9, 2**8, 2**7, 2**6, 2**5, 2**4, 2**3, 2**2, 2**1])    
        gamma = trial.suggest_categorical("gamma", [2**-10, 2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**10, 2**9, 2**8, 2**7, 2**6, 2**5, 2**4, 2**3, 2**2, 2**1])    
        Data_driven_model_hyperparameter_value = [C, gamma] #Value list of hyper-parameters of the data-driven model.
    elif Data_driven_model == "artificial neural networks":
        Data_driven_model_hyperparameter_name = ["number of hidden layers", "activation function", "number of neurons in each hidden layer", "learning rate"] #Name list of hyper-parameters of the data-driven model.
        number_of_hidden_layers = trial.suggest_int("number of hidden layers", 1, 10)    
        activation_function = trial.suggest_categorical("activation function", ["ReLU", "sigmoid", "tanh"])    
        number_of_neurons_in_each_hidden_layer = trial.suggest_int("number of neurons in each hidden layer", 2, 10)    
        learning_rate = trial.suggest_categorical("learning rate", [0.0001, 0.001, 0.01, 0.1, 1])
        Data_driven_model_hyperparameter_value = [number_of_hidden_layers, activation_function, number_of_neurons_in_each_hidden_layer, learning_rate] #Value list of hyper-parameters of the data-driven model.
    elif Data_driven_model == "random forests":
        Data_driven_model_hyperparameter_name = ["number of trees"] #Name list of hyper-parameters of the data-driven model.
        number_of_trees = trial.suggest_int("number of trees", 10, 150, step = 5)  
        Data_driven_model_hyperparameter_value = [number_of_trees] #Value list of hyper-parameters of the data-driven model.
    elif Data_driven_model == "extreme gradient boosting":
        Data_driven_model_hyperparameter_name = ["number of trees", "learning rate"] #Name list of hyper-parameters of the data-driven model.
        number_of_trees = trial.suggest_int("number of trees", 10, 150, step = 5)   
        learning_rate = trial.suggest_categorical("learning rate", [0.0001, 0.001, 0.01, 0.1, 1])    
        Data_driven_model_hyperparameter_value = [number_of_trees, learning_rate] #Value list of hyper-parameters of the data-driven model.
    else:
        Data_driven_model_hyperparameter_name = [] #Name list of hyper-parameters of the data-driven model.
        Data_driven_model_hyperparameter_value = [] #Value list of hyper-parameters of the data-driven model.
    ###########################################################################################

    Accuracy_metric = "coefficient of determination" #Model evaluation metric.
    
    X_training = [Historical_data_file_name, Avaiable_variables, Target_load, 
                  Missing_data_handling_method, Missing_data_handling_method_hyperparameter_name, Missing_data_handling_method_hyperparameter_value,
                  Outlier_identification_method, Outlier_identification_method_hyperparameter_name, Outlier_identification_method_hyperparameter_value, 
                  Data_normalization_method, Data_normalization_method_hyperparameter_name, Data_normalization_method_hyperparameter_value, 
                  Feature_engineering_method, Feature_engineering_method_hyperparameter_name, Feature_engineering_method_hyperparameter_value, 
                  Data_driven_model, Data_driven_model_hyperparameter_name, Data_driven_model_hyperparameter_value, 
                  Accuracy_metric]

    returncode, results = AutoModelTraining.AutoModelTraining(X_training, GPT_model = GPT_model, api_key = api_key, organization_id = organization_id, temperature = 0.6)
    n_trial = AutoModelTraining.Log(n_trial, returncode, results)
    
    if returncode == 0: #Generate a code of model training successfully.
        model_accuracy = results["model accuracy"]
    else: #Cannot generate a code of model training successfully.
        model_accuracy = 0
    
    if best_accuracy < model_accuracy:
        best_messages = [results["all messages of GPT"][-1][0], results["all messages of GPT"][-1][-1]]
        best_messages_save = open('best_messages.pkl', "wb")
        pickle.dump(best_messages, best_messages_save)
        best_messages_save.close()
        best_accuracy = results["model accuracy"]
        
    return model_accuracy

if __name__ == "__main__":
    print("########################################")
    print("Automated model training")
    print("########################################")
    study_name = "GPT-AutoPrediction"
    storage_name = "sqlite:///{}.db".format(study_name)
    AutoModelTraining.LogInitialization()
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(direction="maximize", sampler = TPESampler(seed = 0), study_name = study_name, storage=storage_name)
    study.optimize(objective, n_trials = 100)

    print("########################################")
    print("Automated model deployment (one step)")
    print("########################################")
    study = optuna.create_study(study_name=study_name, sampler = TPESampler(seed = 0), storage=storage_name, load_if_exists=True)
    best_messages_save = open('best_messages.pkl','rb')
    best_messages = pickle.load(best_messages_save)
    best_messages_save.close()
    os.remove('best_messages.pkl')
    
    Historical_data_file_name = "Data/Historical_operational_data.csv" #Name of the file that stores historical data.
    Real_time_data_file_name = "Data/Real_time_operational_data.csv" #Name of the file that stores real-time data.
    Target_load = "electrical load" #Building energy load that you want to predict.
    Data_normalization_method = study.best_trial.params["data_normalization"] #Data normalization method.
    Model_interpretation_method = "Shapley additive explanations" #Model interpretation method.
    X_deployment = [Historical_data_file_name, Real_time_data_file_name, Target_load, Data_normalization_method, Model_interpretation_method]
    
    returncode, results = AutoModelDeployment.AutoModelDeployment(X_deployment, best_messages = best_messages, GPT_model = GPT_model, api_key = api_key, organization_id = organization_id, temperature = 0.6)
    AutoModelDeployment.LogInitialization()
    AutoModelDeployment.Log(returncode, results)
    
    if returncode == 0: #Generate a code of model deployment successfully.
        exec(results["code"])
        with open("Results/Final code for model deployment.py", 'w', encoding="utf-8") as f:
            Code_deployment = results["code"]
            f.write(Code_deployment)