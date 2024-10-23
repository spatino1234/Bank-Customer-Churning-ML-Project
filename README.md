Overview
This Streamlit application predicts customer churn for a bank using various machine learning models. It provides an interactive interface for selecting customers, viewing their details, and getting predictions on their likelihood to churn. The app also generates explanations for the predictions and personalized emails for customer retention.

Features
Interactive customer selection
Display of customer details
Churn prediction using multiple machine learning models
Visualization of churn probability
Comparison of predictions across different models
Percentile analysis of customer attributes
AI-generated explanation of churn prediction
AI-generated personalized retention email
Installation
Clone this repository:

git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
Install the required packages:

pip install -r requirements.txt
Set up your GROQ API key:

Create a .streamlit/secrets.toml file in the project directory
Add your GROQ API key to this file:
GROQ_API_KEY = "your-api-key-here"
Usage
Run the Streamlit app:

streamlit run streamlit_app.py
Open your web browser and go to the URL provided by Streamlit

Use the dropdown menu to select a customer.

View the customer's details, churn prediction, and other insights.

Scroll down to see the AI-generated explanation and personalized email.

Files
streamlit_app.py: Main Streamlit application script
utils.py: Utility functions for creating charts and visualizations
churn.csv: Dataset containing customer information
xgb_model.pkl, nb_model.pkl, etc.: Trained machine learning models
Models Used
XGBoost
Random Forest
Support Vector Machine
K-Nearest Neighbors
Naive Bayes
Decision Tree
Dependencies
streamlit
pandas
numpy
scikit-learn
xgboost
plotly
openai
Note
This application uses the GROQ API for generating explanations and emails. Ensure you have a valid API key and sufficient credits to use these features.
