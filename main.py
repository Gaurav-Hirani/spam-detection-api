import os
import pickle
import string
import nltk
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- 1. VERCEL-SPECIFIC NLTK SETUP ---
# Vercel is read-only. We must force NLTK to download to /tmp (the only writeable folder).
nltk_data_path = os.path.join('/tmp', 'nltk_data')

# Create the directory if it doesn't exist
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Add this path to NLTK so it knows where to look
nltk.data.path.append(nltk_data_path)

# Download all necessary resources to /tmp
# Added 'punkt_tab' to resolve the LookupError seen in logs
resources = ['punkt', 'punkt_tab', 'stopwords']

for resource in resources:
    try:
        if resource == 'stopwords':
            nltk.data.find('corpora/stopwords')
        elif resource == 'punkt_tab':
            nltk.data.find('tokenizers/punkt_tab')
        else:
            nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_path)

# --- 2. SETUP APP & TEMPLATES ---
app = FastAPI()

# Use absolute path for templates to avoid "TemplateNotFound" errors on cloud
current_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# --- 3. LOAD MODEL & VECTORIZER ---
with open(os.path.join(current_dir, 'vectorizer.pkl'), 'rb') as f:
    tfidf = pickle.load(f)
with open(os.path.join(current_dir, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)

# --- 4. PREPROCESSING FUNCTION ---
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    # word_tokenize uses punkt and punkt_tab
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

# --- 5. ROUTE: SHOW THE DASHBOARD (GET) ---
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- 6. ROUTE: HANDLE PREDICTION (POST) ---
@app.post("/predict", response_class=HTMLResponse)
def predict_spam(request: Request, message: str = Form(...)):
    # --- RULE BASED OVERRIDE (Hybrid Engine) ---
    suspicious_patterns = ["http", "https", "www.", ".com", "bit.ly", "tinyurl"]
    is_suspicious_link = any(pattern in message.lower() for pattern in suspicious_patterns)

    # A. Preprocess the form data
    transformed_sms = transform_text(message)
    
    # B. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    
    # C. Predict using ML Model
    result = model.predict(vector_input)[0]
    
    # --- COMBINE LOGIC ---
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
