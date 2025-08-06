# gemini_helper.py
import google.generativeai as genai

def configure_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-pro")

def get_gemini_response(prompt, model):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# --------------------------------------------------------------------------------------------------------------------------------------------

# For Hardcoded API key for demonstration purposes
# import google.generativeai as genai

# # ✅ Paste your Gemini 2.5 Pro API key here
# GEMINI_API_KEY = "YOUR_API_KEY_HERE"

# # Configure Gemini once using the hardcoded key
# genai.configure(api_key=GEMINI_API_KEY)

# # Load the Gemini model
# model = genai.GenerativeModel(model_name="gemini-2.5-pro")

# def get_gemini_response(prompt):
#     try:
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"❌ Error: {str(e)}"