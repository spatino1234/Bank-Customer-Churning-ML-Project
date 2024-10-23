
# Customer Churn Prediction App

This **Streamlit** application predicts customer churn for a bank using multiple machine learning models. It provides an intuitive and interactive interface for selecting customers, viewing their details, and predicting their likelihood of churning. Additionally, the app generates AI-powered explanations for the predictions and creates personalized emails aimed at customer retention.

---

## ✨ Features

- **Interactive Customer Selection**: Choose a customer from a dropdown menu.
- **Customer Details Display**: View detailed information about the selected customer.
- **Churn Prediction**: Leverages multiple machine learning models to predict churn probability.
- **Churn Probability Visualization**: See a graphical representation of the churn likelihood.
- **Model Comparison**: Compare predictions across different machine learning models.
- **Percentile Analysis**: View how the customer’s attributes compare to others.
- **AI-Generated Explanations**: Receive AI-driven explanations for churn predictions.
- **AI-Generated Retention Emails**: Personalized customer retention emails based on prediction insights.

---

## 🚀 Installation

1. **Clone this repository**:

    ```bash
    git clone https://github.com/yourusername/customer-churn-prediction.git
    cd customer-churn-prediction
    ```

2. **Install the required packages**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up your GROQ API key**:

    - Create a `.streamlit/secrets.toml` file in the project directory.
    - Add your **GROQ API key** to this file:

      ```toml
      GROQ_API_KEY = "your-api-key-here"
      ```

---

## 🛠️ Usage

1. **Run the Streamlit app**:

    ```bash
    streamlit run streamlit_app.py
    ```

2. **Open your browser** at the URL provided by Streamlit.

3. **Select a customer** from the dropdown menu.

4. **View customer details**, including their predicted likelihood to churn and key insights.

5. **Scroll down** to see AI-generated explanations and a personalized retention email.

---

## 📂 Files

- `streamlit_app.py`: Main Streamlit application script.
- `utils.py`: Utility functions for generating charts and visualizations.
- `churn.csv`: Dataset containing customer information.
- `xgb_model.pkl`, `nb_model.pkl`, etc.: Pre-trained machine learning models used in the app.

---

## 🤖 Models Used

- **XGBoost**
- **Random Forest**
- **Support Vector Machine**
- **K-Nearest Neighbors**
- **Naive Bayes**
- **Decision Tree**

---

## 📦 Dependencies

- **streamlit**
- **pandas**
- **numpy**
- **scikit-learn**
- **xgboost**
- **plotly**
- **openai**

---

## ⚠️ Note

This application leverages the **GROQ API** for generating explanations and retention emails. Make sure you have a valid API key and enough credits to use these features.
