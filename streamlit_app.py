from operator import index
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
import utils as ut

# init OpenAI client with groq endpoint
client = OpenAI(
    base_url = "https://api.groq.com/openai/v1",
    api_key = os.environ.get("GROQ_API_KEY")
)

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

xgboost_model = load_model('xgb_model.pkl')
naive_bayes_model = load_model('nb_model.pkl')
random_forest_model = load_model('rf_model.pkl')
svm_model = load_model('svm_model.pkl')
voting_classifier_model = load_model('voting_clf.pkl')
# xgb_model = load_model('xgb_model.pkl')
xgboost_SMOTE_model = load_model('xgboost-SMOTE.pkl')
knn_model = load_model('knn_model.pkl')
dt_model = load_model('dt_model.pkl')

# make input into a data frame and dict to format 
def prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary):
    input_dict = {
        'Credit_Score' : credit_score,
        'Age' : age,
        'Tenure' : tenure,
        'Balance' : balance,
        'NumOfProducts' : num_products,
        'HasCrCard' : int(has_credit_card),
        'IsActiveMember' : int(is_active_member),
        'EstimatedSalary' : estimated_salary,
        'Geography_France': 1 if location == 'France' else 0,
        'Geography_Germany': 1 if location == 'Germany' else 0,
        'Geography_Spain': 1 if location == 'Spain' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0
    }

    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict
    

# make prediction
def make_predictions(input_df, input_dict):

    # can use whatever models here
    # [0][1] = get prob of customer churning
    probabilities = {
        'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
        'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
        'KNN': knn_model.predict_proba(input_df)[0][1]
    }
    avg_probability = np.mean(list(probabilities.values()))

    col1,col2 = st.columns(2)

    with col1:
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"The Customer has a {avg_probability*100:.2f}% chance of churning.")

    with col2:
        fig_probs = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)


    # # displaying on FE
    # st.markdown('### Model Probabilities') 
    # for model,prob in probabilities.items():
    #     st.write(f"{model} {prob}")
    # st.write(f"Average Probability: {avg_probabiity}")
    return avg_probability


# give the explination of each customer
def explain_prediction(probability, input_dict, surname):
    prompt = f""" You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of machine learning models.

    Your machine learning model has predicted that a customer named {surname} has a {round(probability *100, 1)}% chance of churning. based on the information below.

    Heres the customers information:
    {input_dict}

    Here are the machine learning models top 10 most important features for predicting churn:

            Feature | Importance
    -------------------------------------
        NumOfProducts | 0.323888
        IsActiveMember | 0.164146
        Age | 0.109550
        Geography_Germany | 0.091373
        Balance | 0.052786
        Geography France | 0.046463
        Gender_Female | 0.045283
        Geography_Spain | 0.036855
        Credit_Score | 0.035005
        Estimated Salary | 0.032655
        HasCrCard | 0.031940
        Tenure | 0.030054
        Gender_Male | 0.000000


    {pd.set_option('display.max_columns', None)}

    Here are summary stats for churned customers:
    {df[df['Exited'] == 1].describe()}

    Here are summary stats for non-churned customers:
    {df[df['Exited'] == 0].describe()}

- If the customer has over a 40% risk of churning, generate a 3 sentence explanation of why the customer is at risk of churning

- If the customer has less than a 40% risk of churning, generate a 3 sentence explanation of why the customer is not at risk of churning

    Dont mention the probability of churning, or the machine learning model, or say anything like "Based on the machine learning model's prediction and top 10 most important features", just explain the prediction
    """

    print("Explination Prompt", prompt)

    raw_response = client.chat.completions.create(
        # console.groq.com/settings/limits
        # to choose different models
        model = "llama-3.2-3b-preview",
        messages = [{
            "role": "user",
            "content": prompt
        }]
    )

    return raw_response.choices[0].message.content

def generate_percentiles(df, input_dict):
    all_num_products = df['NumOfProducts'].sort_values().tolist()
    all_balances = df['Balance'].sort_values().tolist()
    all_estimated_salaries = df['EstimatedSalary'].sort_values().tolist()
    all_tenures = df['Tenure'].sort_values().tolist()
    all_credit_scores = df['CreditScore'].sort_values().tolist()

    product_rank = np.searchsorted(all_num_products, input_dict['NumOfProducts'], side='right')
    balance_rank = np.searchsorted(all_balances, input_dict['Balance'], side='right')
    salary_rank = np.searchsorted(all_estimated_salaries, input_dict['EstimatedSalary'], side='right')
    tenure_rank = np.searchsorted(all_tenures, input_dict['Tenure'], side='right')
    credit_rank = np.searchsorted(all_credit_scores, input_dict['Credit_Score'], side='right') 

    N = 10000

    percentiles = {
        'CreditScore': int(np.floor((credit_rank / N) * 100)),
        'Tenure': int(np.floor((tenure_rank / N) * 100)),
        'EstimatedSalary': int(np.floor((salary_rank / N) * 100)),
        'Balance': int(np.floor((balance_rank / N) * 100)),
        'NumOfProducts': int(np.floor((product_rank / N) * 100)),
    }

    return percentiles


def generate_email(probability, input_dict, explanation, surname):
    prompt = f"""
        You're a amanager at HS Bank. You're responsible for ensuring customers stay with the bank and are incentivized to sustain their loyalty with various offers, such as free credit cards, loans, and discounts.

        You saw a customer named {surname} has a {round(probability *100, 1)}% probability of churning

        Here is the customers information:
        {input_dict}

        Here is some explanation as to why the customer might be at risk of churning:
        {explanation}

        Generate an email to the customer based on their information, asking them to stay if they're at risk of churning, or offering them incentives to stay to become more loyal to the bank.

        Make sure to list out a set of incentives for the customer to stay with the bank based on their information in bullet point format. Dont ever mention the probability of churning, or the machine learning model, or say anything like "Based on the machine learning model" to the customer. And if you give a signatur in the mail, make sure its "Serafin Patino | Customer Relationship Manager at HS Bank"
    """

    raw_response = client.chat.completions.create(
        model = "llama-3.1-8b-instant",
        messages = [{
            "role": "user",
            "content": prompt
        }]
    )

    print("\n\nEMAIL PROMPT", prompt)
    return raw_response.choices[0].message.content

st.title("Customer Churn Prediction")

df = pd.read_csv("churn.csv")

# iterate thru rows and create a string of customers formated ID - Name

customers = [
    f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()
]

# display dropdown menu customers
selected_customer_option = st.selectbox("Select a customer", customers)

# store CID and Name from selected customer

if selected_customer_option:
    selected_customer_ID = int(selected_customer_option.split("-")[0])
    selected_customer_Name = selected_customer_option.split("-")[1]
    
    # get data for selected customer
    
    selected_customer = df.loc[df['CustomerId'] == selected_customer_ID].iloc[0]
    print("selected ID", selected_customer_ID)
    print("selected Name", selected_customer_Name)
    print(selected_customer)

    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value = int(selected_customer['CreditScore'])
        )

        location = st.selectbox("Location", ["Spain", "France", "Germany"],
                                index = ["Spain", "France", "Germany"].index(
                                selected_customer['Geography']))

        gender = st.radio("Gender", ["Male", "Female"],
                          index = 0 if selected_customer['Gender'] == "Male" else 1
                         )

        age = st.number_input(
            "Age", 
            min_value=18,
            max_value=100,
            value = int(selected_customer['Age']))

        tenure = st.number_input(
            "Tenure (Years)", 
            min_value=0,
            max_value=50,
            value = int(selected_customer['Tenure'])
        )

    with col2:
        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value = float(selected_customer['Balance']))

        num_products = st.number_input(
            "Number of Products",
            min_value = 1,
            max_value = 10,
            value = int(selected_customer['NumOfProducts'])
            )

        has_credit_card = st.checkbox(
            "Has Credit Card",
            value = bool(selected_customer['HasCrCard'])
            )

        is_active_member = st.checkbox(
            "Is Active Member",
            value = bool(selected_customer['IsActiveMember'])
        )

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value = 0.0,
            value = float(selected_customer['EstimatedSalary'])
        )

    input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary)

    avg_probability = make_predictions(input_df, input_dict)

    percentiles = generate_percentiles(df, input_dict)
    fig = ut.create_percentile_chart(percentiles)
    st.plotly_chart(fig, use_container_width=True)

    explanation = explain_prediction(avg_probability, input_dict, selected_customer['Surname'])

    st.markdown("---")
    st.subheader("Explination of Prediction")
    st.markdown(explanation)

    email = generate_email(avg_probability, input_dict, explanation, selected_customer['Surname'])
    st.markdown("---")
    st.subheader("Email to Customer")
    st.markdown(email)
