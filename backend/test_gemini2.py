import google.generativeai as genai
try:
    genai.configure(api_key="AIzaSyCl9hbLYxkJJrVt_71Dr0q-W3tO5hhQOZo")
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Say hello")
    print("SUCCESS")
    print(response.text)
except Exception as e:
    print(f"FAILED: {e}")
