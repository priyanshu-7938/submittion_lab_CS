# import pandas as pd
# from pgmpy.models import BayesianNetwork
# from pgmpy.models import BayesianModel
# from pgmpy.estimators import BayesianEstimator
# from pgmpy.inference import VariableElimination

# data = pd.read_csv('file.data', header=None, names=[
#     'age', 'sex', 'on thyroxine', 'query on thyroxine', 'on antithyroid medication', 
#     'sick', 'pregnant', 'thyroid surgery', 'I131 treatment', 'query hypothyroid', 
#     'query hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 
#     'TSH measured', 'TSH', 'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 
#     'T4U', 'FTI measured', 'FTI', 'TBG measured', 'TBG', 'referral source', 'classification'
# ])

# def preprocess_classification(x):
#     if x.startswith('-'):
#         return 0  # normal
#     elif x[0] in ['A', 'B', 'C', 'D']:
#         return 2  # hyperthyroid
#     elif x[0] in ['E', 'F', 'G', 'H']:
#         return 1  # hypothyroid
#     else:
#         return 3  # other

# data['classification'] = data['classification'].apply(preprocess_classification)
# features = ['TT4', 'T3', 'TSH']
# target = 'classification'

# data_features = data[features]
# data_target = data[target]


# from sklearn.preprocessing import LabelEncoder

# le = LabelEncoder()
# data_target = le.fit_transform(data_target)

# # model = BayesianModel([('TT4', 'classification'), ('T3', 'classification'), ('TSH', 'classification')])
# # estimator = BayesianEstimator(model)
# model = BayesianNetwork([('TT4', 'classification'), ('T3', 'classification'), ('TSH', 'classification')])
# estimator = BayesianEstimator(model, data)

# # estimator.fit(data_features, data_target)
# estimator.estimate_parameters(data)


# def query_interface():
#     print("Welcome to the Bayesian Network Query Interface!")
#     print("You can input new data to get the predicted classification.")

#     while True:
#         # Get the input data from the user
#         TT4 = float(input("Enter the value of TT4: "))
#         T3 = float(input("Enter the value of T3: "))
#         TSH = float(input("Enter the value of TSH: "))

#         # Create a new data point
#         new_data = pd.DataFrame({'TT4': [TT4], 'T3': [T3], 'TSH': [TSH]})

#         # Perform inference
#         posterior = infer.query(['classification'], evidence=new_data)

#         # Print the predicted classification
#         print("Predicted classification:", posterior.values)

#         # Ask the user if they want to continue
#         cont = input("Do you want to continue? (yes/no): ")
#         if cont.lower() != "yes":
#             break

# # Run the query interface
# query_interface()


# code 2 

# import pandas as pd
# from pgmpy.models import BayesianNetwork
# from pgmpy.estimators import BayesianEstimator
# from pgmpy.inference import VariableElimination
# from sklearn.preprocessing import LabelEncoder

# # Load the dataset
# data = pd.read_csv('file.data', header=None, names=[
#     'age', 'sex', 'on thyroxine', 'query on thyroxine', 'on antithyroid medication', 
#     'sick', 'pregnant', 'thyroid surgery', 'I131 treatment', 'query hypothyroid', 
#     'query hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 
#     'TSH measured', 'TSH', 'T3 measured', 'T3', 'TT4 measured', 
#     'TT4', 'T4U measured', 'T4U', 'FTI measured', 'FTI', 
#     'TBG measured', 'TBG', 'referral source', 'classification'
# ])

# # Preprocess classification data
# def preprocess_classification(x):
#     if x.startswith('-'):
#         return 0  # normal
#     elif x[0] in ['A', 'B', 'C', 'D']:
#         return 2  # hyperthyroid
#     elif x[0] in ['E', 'F', 'G', 'H']:
#         return 1  # hypothyroid
#     else:
#         return 3  # other

# data['classification'] = data['classification'].apply(preprocess_classification)

# # Define features and target variable
# features = ['TT4', 'T3', 'TSH']
# target = 'classification'

# data_features = data[features]
# data_target = data[target]

# # Encode target variable
# le = LabelEncoder()
# data_target_encoded = le.fit_transform(data_target)

# # Create Bayesian Network model and estimator
# model = BayesianNetwork([('TT4', 'classification'), ('T3', 'classification'), ('TSH', 'classification')])
# estimator = BayesianEstimator(model, data)

# # Fit the model with data
# estimator.estimate_parameters(data)

# # Create inference object
# infer = VariableElimination(model)

# def query_interface():
#     print("Welcome to the Bayesian Network Query Interface!")
#     print("You can input new data to get the predicted classification.")

#     while True:
#         # Get the input data from the user
#         TT4 = float(input("Enter the value of TT4: "))
#         T3 = float(input("Enter the value of T3: "))
#         TSH = float(input("Enter the value of TSH: "))

#         # Create a new data point for inference
#         new_data = pd.DataFrame({'TT4': [TT4], 'T3': [T3], 'TSH': [TSH]})

#         # Perform inference to get posterior probabilities
#         posterior = infer.query(['classification'], evidence=new_data.to_dict(orient='records')[0])

#         # Print the predicted classification probabilities
#         print("Predicted classification probabilities:", posterior)

#         # Ask the user if they want to continue
#         cont = input("Do you want to continue? (yes/no): ")
#         if cont.lower() != "yes":
#             break

# # Run the query interface
# query_interface()

# code 3
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('file.data', header=None, names=[
    'age', 'sex', 'on thyroxine', 'query on thyroxine', 'on antithyroid medication', 
    'sick', 'pregnant', 'thyroid surgery', 'I131 treatment', 'query hypothyroid', 
    'query hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 
    'TSH measured', 'TSH', 'T3 measured', 'T3', 'TT4 measured', 
    'TT4', 'T4U measured', 'T4U', 'FTI measured', 'FTI', 
    'TBG measured', 'TBG', 'referral source', 'classification'
])

def preprocess_classification(x):
    if x.startswith('-'):
        return 0  # normal
    elif x[0] in ['A', 'B', 'C', 'D']:
        return 2  # hyperthyroid
    elif x[0] in ['E', 'F', 'G', 'H']:
        return 1  # hypothyroid
    else:
        return 3  # other

data['classification'] = data['classification'].apply(preprocess_classification)

features = ['TT4', 'T3', 'TSH']
target = 'classification'

data = data[['TT4', 'T3', 'TSH', 'classification']]

data_features = data[features]
data_target = data[target]

le = LabelEncoder()
data_target_encoded = le.fit_transform(data_target)

model = BayesianNetwork([('TT4', 'classification'), ('T3', 'classification'), ('TSH', 'classification')])
print(model)
estimator = BayesianEstimator(model, data)
estimate = estimator.estimate_parameters(data)
infer = VariableElimination(model)

def query_interface():
    print("Welcome to the Bayesian Network Query Interface!")
    print("You can input new data to get the predicted classification.")

    while True:
        TT4 = float(input("Enter the value of TT4: "))
        T3 = float(input("Enter the value of T3: "))
        TSH = float(input("Enter the value of TSH: "))

        new_data = pd.DataFrame({'TT4': [TT4], 'T3': [T3], 'TSH': [TSH]})

        posterior = infer.query(['classification'], evidence=new_data.to_dict(orient='records')[0])

        print("Predicted classification probabilities:", posterior)

        cont = input("Do you want to continue? (yes/no): ")
        if cont.lower() != "yes":
            break

query_interface()