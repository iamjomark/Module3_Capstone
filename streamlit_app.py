# Streamlit App
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Load the trained pipeline
model = joblib.load("saved_models/LGBM_grid_f2.pkl")
threshold = 0.2

# Set page title
st.title("🧳 Travel Insurance Claim Predictor")
st.write("This tool uses a LightGBM model to estimate the risk of travel insurance claims.")

# Input form
st.header("Client Policy Information")

product_name = st.selectbox("Product Name", ['Annual Silver Plan', 'Cancellation Plan', 'Basic Plan',
       '2 way Comprehensive Plan', 'Bronze Plan',
       '1 way Comprehensive Plan', 'Rental Vehicle Excess Insurance',
       'Single Trip Travel Protect Gold', 'Silver Plan', 'Value Plan',
       '24 Protect', 'Annual Travel Protect Gold', 'Comprehensive Plan',
       'Ticket Protector', 'Travel Cruise Protect',
       'Single Trip Travel Protect Silver',
       'Individual Comprehensive Plan', 'Gold Plan', 'Annual Gold Plan',
       'Child Comprehensive Plan', 'Premier Plan',
       'Annual Travel Protect Silver',
       'Single Trip Travel Protect Platinum',
       'Annual Travel Protect Platinum',
       'Spouse or Parents Comprehensive Plan',
       'Travel Cruise Protect Family'])
agency = st.selectbox("Agency", ['C2B', 'EPX', 'JZI', 'CWT', 'LWC', 'ART', 'CSR', 'RAB', 'KML',
       'SSI', 'TST', 'TTW', 'ADM', 'CCR', 'CBH'])
destination = st.selectbox("Destination", ['SINGAPORE', 'MALAYSIA', 'INDIA', 'UNITED STATES',
       'KOREA, REPUBLIC OF', 'THAILAND', 'GERMANY', 'JAPAN', 'INDONESIA',
       'VIET NAM', 'AUSTRALIA', 'FINLAND', 'UNITED KINGDOM', 'SRI LANKA',
       'SPAIN', 'HONG KONG', 'MACAO', 'CHINA', 'UNITED ARAB EMIRATES',
       'IRAN, ISLAMIC REPUBLIC OF', 'TAIWAN, PROVINCE OF CHINA', 'POLAND',
       'CANADA', 'OMAN', 'PHILIPPINES', 'GREECE', 'BELGIUM', 'TURKEY',
       'BRUNEI DARUSSALAM', 'DENMARK', 'SWITZERLAND', 'NETHERLANDS',
       'SWEDEN', 'MYANMAR', 'KENYA', 'CZECH REPUBLIC', 'FRANCE',
       'RUSSIAN FEDERATION', 'PAKISTAN', 'ARGENTINA',
       'TANZANIA, UNITED REPUBLIC OF', 'SERBIA', 'ITALY', 'CROATIA',
       'NEW ZEALAND', 'PERU', 'MONGOLIA', 'CAMBODIA', 'QATAR', 'NORWAY',
       'LUXEMBOURG', 'MALTA', "LAO PEOPLE'S DEMOCRATIC REPUBLIC",
       'ISRAEL', 'SAUDI ARABIA', 'AUSTRIA', 'PORTUGAL', 'NEPAL',
       'UKRAINE', 'ESTONIA', 'ICELAND', 'BRAZIL', 'MEXICO',
       'CAYMAN ISLANDS', 'PANAMA', 'BANGLADESH', 'TURKMENISTAN',
       'BAHRAIN', 'KAZAKHSTAN', 'TUNISIA', 'IRELAND', 'ETHIOPIA',
       'NORTHERN MARIANA ISLANDS', 'MALDIVES', 'SOUTH AFRICA',
       'VENEZUELA', 'COSTA RICA', 'JORDAN', 'MALI', 'CYPRUS', 'MAURITIUS',
       'LEBANON', 'KUWAIT', 'AZERBAIJAN', 'HUNGARY', 'BHUTAN', 'BELARUS',
       'MOROCCO', 'ECUADOR', 'UZBEKISTAN', 'CHILE', 'FIJI',
       'PAPUA NEW GUINEA', 'ANGOLA', 'FRENCH POLYNESIA', 'NIGERIA',
       'MACEDONIA, THE FORMER YUGOSLAV REPUBLIC OF', 'NAMIBIA', 'GEORGIA',
       'COLOMBIA', 'SLOVENIA', 'EGYPT', 'ZIMBABWE', 'BULGARIA', 'BERMUDA',
       'URUGUAY', 'GUINEA', 'GHANA', 'BOLIVIA', 'TRINIDAD AND TOBAGO',
       'VANUATU', 'GUAM', 'UGANDA', 'JAMAICA', 'LATVIA', 'ROMANIA',
       'REPUBLIC OF MONTENEGRO', 'KYRGYZSTAN', 'GUADELOUPE', 'ZAMBIA',
       'RWANDA', 'BOTSWANA', 'GUYANA', 'LITHUANIA', 'GUINEA-BISSAU',
       'SENEGAL', 'CAMEROON', 'SAMOA', 'PUERTO RICO', 'TAJIKISTAN',
       'ARMENIA', 'FAROE ISLANDS', 'DOMINICAN REPUBLIC',
       'MOLDOVA, REPUBLIC OF', 'BENIN', 'REUNION'])
distribution_channel = st.selectbox("Distribution Channel", ["Online", "Travel Agent"])

age = st.number_input("Customer Age", min_value=0, value=30)
duration = st.number_input("Trip Duration (days)", min_value=1, value=7)
commission = st.number_input("Commission ($)", min_value=0.0, value=50.0)
net_sales = st.number_input("Net Sales ($)", min_value=0.0, value=500.0)

# Prepare input
input_data = pd.DataFrame([{
    "Product Name": product_name,
    "Agency": agency,
    "Destination": destination,
    "Distribution Channel": distribution_channel,
    "Age": age,
    "Duration": duration,
    "Commission": commission,
    "Net Sales": net_sales
}])

# Prediction
if st.button("Predict Claim Risk"):
    proba = model.predict_proba(input_data)[0][1]
    pred = int(proba >= threshold)

    st.write(f"**Claim Probability:** {proba:.2%}")
    if pred:
        st.error("⚠️ High risk of a claim!")
    else:
        st.success("✅ Low risk of a claim.")

    st.caption(f"Model threshold: {threshold} (optimized for F2 score)")

# Feature importance visualization
st.subheader("Global Feature Importance")

# Get feature importances from the LightGBM classifier inside the pipeline
feature_importance = model.named_steps['classifier'].feature_importances_

# Get feature names after preprocessing (one-hot encoding creates multiple features)
feature_names = model.named_steps['preprocess'].get_feature_names_out()

# Create a dataframe and sort by importance
fi_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# Show bar chart
st.bar_chart(fi_df.set_index('Feature'))

