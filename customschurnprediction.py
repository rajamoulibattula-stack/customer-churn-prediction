# import libraries
import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# read dataset
df=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Feature X and Target y
df=df.drop("customerID",axis=1)

# convert numeric and filling missing value
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")

df["TotalCharges"]=df["TotalCharges"].fillna(df["TotalCharges"].mean())

# Target variable
y=df['Churn'].map({'No':0,'Yes':1})
# Feature x
X=df.drop('Churn',axis=1)

X=pd.get_dummies(X,drop_first=True)

model_columns=X.columns

# standardscaler
scaler=StandardScaler()
X=pd.DataFrame(
    scaler.fit_transform(X),
    columns=model_columns
)

# train_test_split
X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create a model
model=LogisticRegression(max_iter=1000)

# train the model
model.fit(X_train,y_train)

# save model
with open("model.pkl","wb") as f:
    pickle.dump(model,f) # # Save trained model so we can predict later without retraining.

# save scaler
with open("scaler.pkl","wb") as f:
    pickle.dump(scaler,f) # # Save scaler because new input must be scaled exactly like training data.

# save columns
with open("columns.pkl","wb") as f:
    pickle.dump(model_columns,f)


# load saved files
with open("model.pkl","rb") as f:
    loded_model=pickle.load(f)

with open("scaler.pkl","rb") as f:
    scaler=pickle.load(f)

with open("columns.pkl","rb") as f:
    model_columns=pickle.load(f)

# streamlit 
st.title("CustomerChurnPrediction")
st.write("You can enter detail it predict the user continue the churn or not")

gender = st.selectbox("Gender", ["Female", "Male"])

SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])

Partner = st.selectbox("Partner", ["No", "Yes"])

Dependents = st.selectbox("Dependents", ["No", "Yes"])

tenure = st.number_input("Tenure", min_value=0)

PhoneService = st.selectbox("Phone Service", ["No", "Yes"])

MultipleLines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

OnlineBackup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

DeviceProtection = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

TechSupport = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

StreamingTV = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

Contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0
)

TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0
)

# prediction button
if st.button("Predict"):
    input_data=pd.DataFrame([[
        gender,
        SeniorCitizen,
        Partner,
        Dependents,
        tenure,
        PhoneService,
        MultipleLines,
        InternetService,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection,
        TechSupport,
        StreamingTV,
        StreamingMovies,
        Contract,
        PaperlessBilling,
        PaymentMethod,
        MonthlyCharges,
        TotalCharges
    ]],columns=[
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges"
    ])

    # convert text to numbers
    input_data = pd.get_dummies(input_data, drop_first=True)

    # Match training columns (FIXED)
    input_data = input_data.reindex(
    columns=model_columns,
    fill_value=0
    )

    # scale input
    input_data = scaler.transform(input_data)

    # prediction
    prediction = loded_model.predict(input_data)
    probability = loded_model.predict_proba(input_data)

    # check wheather the customer is churn or not using if-else
    if prediction[0]==1:
        st.error(
            f"Customer is likely to churn"
            f"({probability[0][1]*100:.2f}%)"
        )
    else:
        st.success(
            f"Customer is likely to stay"
            f"({probability[0][0]*100:.2f}%)"
        )