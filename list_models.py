import google.generativeai as genai

# Replace this with your actual API key
api_key = ""

genai.configure(api_key=api_key)

models = genai.list_models()

for model in models:
    print(f"📌 Model name: {model.name}")
    print(f"🔧 Model generation capabilities: {model.supported_generation_methods}")
    print("-" * 40)
