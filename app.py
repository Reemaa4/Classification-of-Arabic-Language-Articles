from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load the saved model and tokenizer
model = BertForSequenceClassification.from_pretrained('./arabic_bert_classifier/Bert_fine_tuned', use_safetensors=True)
tokenizer = BertTokenizer.from_pretrained('./arabic_bert_classifier/Bert_fine_tuned')

# Categories
categories = ['Culture', 'Economy', 'International', 'Local', 'Religion', 'Sports']

# Home route for input form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle text classification
@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        # Get the text input from the form
        input_text = request.form['text']

        # Tokenize the input
        inputs = tokenizer([input_text], truncation=True, padding=True, max_length=512, return_tensors='pt')

        # Make prediction
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        # Get the category name from the prediction
        predicted_category = categories[prediction]

        return render_template('index.html', prediction=predicted_category)

if __name__ == '__main__':
    app.run(debug=True)
