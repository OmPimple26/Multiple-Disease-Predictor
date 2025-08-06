# app.py – Streamlit Interface for Multiple Disease Prediction with All Models
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import google.generativeai as genai
from gemini_helper import configure_gemini, get_gemini_response

st.set_page_config(page_title="Multiple Disease Predictor", layout="wide")
st.title("🩺 Multiple Disease Predictor")

# Load all trained models
model_files = {
    "Logistic Regression": "logistic_regression_model.joblib",
    "Decision Tree": "decision_tree_model.joblib",
    "Random Forest": "random_forest_model.joblib",
    "Naive Bayes": "naive_bayes_model.joblib",
    "Support Vector Machine": "support_vector_machine_model.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors_model.joblib",
    "XGBoost": "xgboost_model.joblib"
}

# Load shared components
scaler = joblib.load("scaler.joblib")
label_encoder = joblib.load("label_encoder.pkl")
symptoms_list = joblib.load("symptoms_list.joblib")
best_model_name = joblib.load("best_model_name.joblib")

# Sidebar – Model selection
with st.sidebar:
    st.header("Select Prediction Model")
    selected_model_name = st.selectbox("Choose a model to use:", list(model_files.keys()))
    st.success(f"🏆 Best Model Based on Accuracy: {best_model_name}")
    st.markdown("---")
    st.caption("Tip: Select more symptoms for better accuracy.")

# Load selected model
model = joblib.load(model_files[selected_model_name])

# Main Form
st.subheader("Input Patient Symptoms")
with st.form("predict_form"):
    selected_symptoms = st.multiselect("Choose symptoms:", options=symptoms_list, default=[])
    gender = st.selectbox("Gender (if applicable):", ["Female", "Male"])
    submit = st.form_submit_button("🔍 Predict Disease")

if submit:
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom to proceed with prediction.")
    else:
     # Prepare input row
     input_row = {col: 0 for col in symptoms_list}
     for symptom in selected_symptoms:
         input_row[symptom] = 1

     if "gender" in symptoms_list:
         input_row["gender"] = 1 if gender == "Male" else 0

     input_df = pd.DataFrame([input_row])[symptoms_list]
     scaled_input = scaler.transform(input_df)

     # Predict
     pred_index = model.predict(scaled_input)[0]
     pred_name = label_encoder.inverse_transform([pred_index])[0]

     st.success(f"### 🩺 Predicted Disease: **{pred_name}**")

     # Classification Report
     y_pred = model.predict(scaled_input)
     y_true = [pred_index]  # Simulated ground truth for formatting
     st.subheader("📊 Classification Report")
     try:
        report_dict = classification_report(
            y_true,
            y_pred,
            labels=np.unique(y_true),
            target_names=label_encoder.inverse_transform(np.unique(y_true)),
            zero_division=0,
            output_dict=True
        )
        report_df = pd.DataFrame(report_dict).T
        st.dataframe(report_df.style.format("{:.2f}"))
     except Exception as e:
        st.warning(f"Error generating report: {e}")

     # Display all prediction probabilities
     st.write("🔢 Probabilities for all diseases:")
     probs = model.predict_proba(scaled_input)[0]
     prob_df = pd.DataFrame({"Disease": label_encoder.classes_, "Probability": probs})
     st.dataframe(prob_df.sort_values("Probability", ascending=False))

    #  # 📉 Feature Importance (only for models that support it)
    #  if hasattr(model, "feature_importances_"):
    #     st.subheader("📉 Top Feature Importances")
    #     importances = model.feature_importances_
    #     importance_df = pd.DataFrame({
    #        "Symptom": symptoms_list,
    #         "Importance": importances
    #         }).sort_values("Importance", ascending=False).head(20)
        
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     sns.barplot(data=importance_df, y="Symptom", x="Importance", palette="viridis", ax=ax)
    #     ax.set_title("Top 20 Important Symptoms")
    #     st.pyplot(fig)
        
    # Prescription suggestion
     prescriptions = {
        "Fungal infection": "Use antifungal creams or oral medication, maintain hygiene, and keep affected area dry.",
        "Drug Reaction": "Stop the offending drug immediately, use antihistamines or corticosteroids as needed.",
        "Chronic cholestasis": "Use ursodeoxycholic acid, avoid fatty foods, manage underlying liver disorder.",
        "Heart attack": "Administer aspirin, nitroglycerin, and seek emergency cardiac care immediately.",
        "Jaundice": "Treat underlying liver issue, hydrate well, avoid alcohol and fatty foods.",
        "Chicken pox": "Use calamine lotion, antihistamines for itchiness, and isolate until scabs form.",
        "Acne": "Use topical retinoids or antibiotics, avoid oily cosmetics, and cleanse skin regularly.",
        "Impetigo": "Apply topical antibiotics, maintain skin hygiene, and avoid scratching.",
        "Gastroenteritis": "Rest, hydrate with oral rehydration salts, and follow a light diet.",
        "Paralysis (brain hemorrhage)": "Immediate hospitalization, physical therapy, and medications to manage bleeding.",
        "Psoriasis": "Apply steroid creams, moisturizers, and consider phototherapy or immunosuppressants.",
        "Allergy": "Take antihistamines, avoid known allergens, and use epinephrine for severe reactions.",
        "Hypertension": "Use antihypertensive drugs, reduce salt intake, and monitor blood pressure regularly.",
        "AIDS": "Begin antiretroviral therapy (ART), maintain hygiene, and monitor CD4 count regularly.",
        "Cervical spondylosis": "Use pain relief medication, perform neck exercises, and maintain correct posture.",
        "Arthritis": "Use NSAIDs, stay active with light exercise, and apply hot/cold compresses.",
        "Hepatitis C": "Start antiviral therapy, avoid alcohol, and monitor liver function tests.",
        "Bronchial Asthma": "Use inhaled corticosteroids and bronchodilators, avoid allergens.",
        "Urinary tract infection": "Take antibiotics, increase water intake, and maintain personal hygiene.",
        "Dimorphic hemmorhoids(piles)": "Use topical ointments, eat high-fiber foods, and consider surgical options if severe.",
        "Peptic ulcer diseae": "Use antacids or proton pump inhibitors, avoid spicy food, and test for H. pylori.",
        "Osteoarthristis": "Use painkillers, maintain joint movement with light exercise, and reduce joint strain.",
        "(vertigo) Paroymsal  Positional Vertigo": "Perform Epley maneuver, avoid sudden head movements, and use vestibular suppressants.",
        "Hepatitis B": "Start antiviral medication if chronic, avoid alcohol, and monitor liver function.",
        "Varicose veins": "Use compression stockings, elevate legs, and consider laser/surgical treatment if needed.",
        "Alcoholic hepatitis": "Completely abstain from alcohol, eat a liver-friendly diet, and take prescribed medication.",
        "Migraine": "Use pain relievers or triptans during attacks, rest in a dark room, and avoid known triggers.",
        "Diabetes": "Manage with insulin or oral medication, monitor blood sugar, and follow diabetic diet.",
        "GERD": "Take antacids or PPIs, eat smaller meals, and avoid lying down after eating.",
        "Hypothyroidism": "Use levothyroxine as prescribed, monitor TSH levels, and take medicine on empty stomach.",
        "Malaria": "Take antimalarial drugs like chloroquine/artemisinin, rest, and stay hydrated.",
        "Hyperthyroidism": "Use antithyroid medication or beta blockers, and monitor hormone levels regularly.",
        "Dengue": "Stay hydrated, monitor platelet count, and avoid NSAIDs like ibuprofen.",
        "Hypoglycemia": "Eat or drink fast-acting carbs (like glucose tablets), and monitor blood sugar levels.",
        "Pneumonia": "Use antibiotics (for bacterial), rest, hydrate well, and use oxygen therapy if needed.",
        "Common Cold": "Use decongestants, drink warm fluids, and rest; usually resolves on its own.",
        "Typhoid": "Take antibiotics like ciprofloxacin, maintain hydration, and eat soft food.",
        "Hepatitis E": "Ensure good hydration, avoid alcohol, and rest; usually self-limiting.",
        "hepatitis A": "Rest, eat a liver-friendly diet, and avoid alcohol; usually resolves on its own.",
        "Tuberculosis": "Use multi-drug regimen (RIPE: Rifampin, Isoniazid, etc.) for 6–9 months.",
        "Hepatitis D": "Treat hepatitis B co-infection, avoid alcohol, and monitor for chronic liver damage."
        # Add more as needed
     }
    
     common_care = "\n\n📌 General Advice:\n- Always consult a licensed medical professional for personalized treatment.\n- Maintain a healthy diet and stay hydrated.\n- Follow your medication schedule strictly.\n- Report any unexpected symptoms immediately.\n- Do not self-medicate."

     suggestion = prescriptions.get(pred_name, "Consult a licensed doctor for personalized treatment.") + common_care
     st.subheader("💊 Prescription Suggestion")
     st.info(suggestion)

     # Download prescription as text
     def get_download_link(text, filename):
        b64 = base64.b64encode(text.encode()).decode()
        return f'<a href="data:file/txt;base64,{b64}" download="{filename}">📄 Download Prescription</a>'
     
     download_text = f"Predicted Disease: {pred_name}\nModel: {selected_model_name}\nPrescription: {suggestion}"
     st.markdown(get_download_link(download_text, f"prescription_{pred_name.lower().replace(' ', '_')}.txt"), unsafe_allow_html=True)

# # AI Assistant Chatbot
# st.subheader("💬 Ask AI Assistant")

# api_key = st.text_input("🔑 Enter your Gemini API Key", type="password")
# user_query = st.text_area("🤖 Ask a question about health, symptoms, or disease:")

# if st.button("🧠 Get AI Response"):
#     if not api_key:
#         st.error("Please enter your Gemini API Key.")
#     elif not user_query:
#         st.warning("Please enter a question to ask the AI.")
#     else:
#         with st.spinner("Generating response..."):
#             model = configure_gemini(api_key)
#             answer = get_gemini_response(user_query, model)
#             st.success("✅ Response:")
#             st.write(answer)


# ----------------------------------
# AI Chat Assistant Section (Gemini)
# ----------------------------------

st.markdown("---")
st.subheader("💬 AI Chat Assistant (Gemini)")

api_key = st.text_input("🔑 Enter your Gemini API Key", type="password")
user_query = st.text_area("🤖 Ask a health-related question (e.g., symptoms, disease, lifestyle advice):")

if st.button("🧠 Get AI Response"):
    if not api_key:
        st.error("Please enter your Gemini API Key.")
    elif not user_query:
        st.warning("Please ask a valid question.")
    else:
        with st.spinner("Generating response from Gemini..."):
            model = configure_gemini(api_key)
            answer = get_gemini_response(user_query, model)
            st.success("✅ Gemini AI Response:")
            st.write(answer)

st.caption("© 2025 Om Pimple – Developed for Healthcare AI Prediction")

# --------------------------------------------------------------------------------------------------------------------------------------------

# For hardcoded API key demonstration purposes only
# import streamlit as st
# from gemini_helper import get_gemini_response

# st.set_page_config(page_title="Gemini AI Chatbot", layout="centered")

# st.title("🤖 Gemini 2.5 Pro Chatbot")
# st.markdown("Ask any question and get AI-generated answers instantly!")

# # Text input from user
# user_prompt = st.text_area("📥 Enter your prompt:", height=150)

# # Button to get AI response
# if st.button("🚀 Get AI Response"):
#     if user_prompt.strip() == "":
#         st.warning("Please enter a valid prompt.")
#     else:
#         with st.spinner("Generating response..."):
#             output = get_gemini_response(user_prompt)
#         st.markdown("### 🧠 AI Response:")
#         st.success(output)