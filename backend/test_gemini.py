import google.generativeai as genai
try:
    genai.configure(api_key="AIzaSyCl9hbLYxkJJrVt_71Dr0q-W3tO5hhQOZo")
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Hello")
    print("SUCCESS: ", response.text)
except Exception as e:
    print("FAILED: ", e)
