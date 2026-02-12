from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# 1. SETUP APP & TEMPLATES
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 2. LOAD MODEL & VECTORIZER
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# 3. PREPROCESSING FUNCTION (Must be same as training)
nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# 4. ROUTE: SHOW THE DASHBOARD (GET)
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ... (Imports and setup remain the same) ...

# 5. ROUTE: HANDLE PREDICTION (POST)
@app.post("/predict", response_class=HTMLResponse)
def predict_spam(request: Request, message: str = Form(...)):
    # --- RULE BASED OVERRIDE (Hybrid Engine) ---
    # If the message contains suspicious links, we flag it immediately.
    suspicious_patterns = ["http", "https", "www.", ".com", "bit.ly", "tinyurl"]
    is_suspicious_link = any(pattern in message.lower() for pattern in suspicious_patterns)

    # A. Preprocess the form data
    transformed_sms = transform_text(message)
    
    # B. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    
    # C. Predict using ML Model
    result = model.predict(vector_input)[0]
    
    # --- COMBINE LOGIC ---
    # If ML says Spam OR it has a suspicious link, we call it Spam.
    if result == 1 or is_suspicious_link:
        prediction_text = "Spam"
    else:
        prediction_text = "Not Spam"
    
    # D. Return the page with results
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "prediction": prediction_text,
        "message": message
    })