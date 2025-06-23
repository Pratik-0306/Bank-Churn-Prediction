## importing important libraries:
import datetime 
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

# importing the trained model:
model = load_model('model.h5')

## load the pickle files:
with open("onehot_encode_geography.pkl", 'rb') as file:
    onehot_encode_geography = pickle.load(file)

with open("label_encode_gender.pkl", 'rb') as file:
      label_encode_gender=pickle.load(file)

with open("scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app:
st.title("Bank Customer Churn Prediction")

# User Input:
creditScore = st.number_input("Credit Score") 
Geography = st.selectbox("Geography",onehot_encode_geography.categories_[0])
Gender = st.selectbox("Gender", label_encode_gender.classes_)
Age = st.selectbox('Age', range(18, 93))  # 93 is exclusive, so this includes 92
Tenure = st.slider('Tenure',0,10)
Balance = st.number_input("Balance")
NumOfProducts = st.slider("Num Of Products",1,4)
HasCrCard = st.selectbox("Has Credit Card", 1, 0)
IsActiveMember = st.selectbox("Is_Active_Member", [0,1])
EstimatedSalary = st.number_input("Estimated Salary")

# prepare the input data:
input_data = ({
    'CreditScore': [creditScore], 
'Geography': [Geography],
'Gender':[Gender],
'Age': [Age],
'Tenure': [Tenure] ,
'Balance': [Balance],
'NumOfProducts': [NumOfProducts],
'HasCrCard': [HasCrCard],
'IsActiveMember': [IsActiveMember],
'EstimatedSalary': [EstimatedSalary]
})

# onehot encode to "Geography":
geo_encoder = onehot_encode_geography.transform([input_data["Geography"]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoder, columns=onehot_encode_geography.get_feature_names_out(["Geography"]))


# Fisrt converting the Dict to the dataframe:
input_df=pd.DataFrame(input_data)
 
## encode categorcial variables:
input_df["Gender"] = label_encode_gender.transform(input_df["Gender"])

# concatenate the 2 dataframe for geography 
input_df=pd.concat([input_df.drop("Geography",axis=1),geo_encoded_df],axis=1)

# print(input_df.dtypes)
# print(input_df.head())

# scale the input data:
input_scaled = scaler.transform(input_df)

# Prediction:
prediction = model.predict(input_scaled)
print(prediction)
prediction_probability = prediction[0][0]
print(prediction_probability)

st.write(f"Churn Probability : {prediction_probability: .2f}")

# Output:
if prediction_probability > 0.5:
    st.write("The Customer is likely to churn")
else:
    st.write("The Customer is not likely to churn")    



