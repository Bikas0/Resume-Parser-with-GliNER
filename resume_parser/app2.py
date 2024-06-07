from flask import Flask, request, jsonify, render_template
from PyPDF2 import PdfReader
from gliner import GLiNER
from pytess_read import text_data
import torch
import os
import re

app = Flask(__name__)

# Function to Extract Text From PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        results = []
        for page in reader.pages:
            text = page.extract_text()
            results.append(text)
        return ' '.join(results)

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint to process resume and extract information
@app.route('/extract_resume_info', methods=['POST'])
def extract_resume_info():
    try:
        # Get file from request
        pdf_file = request.files['file']
        pdf_path = os.path.join('/tmp', pdf_file.filename)
        pdf_file.save(pdf_path)

        # Extract text from the uploaded PDF
        text = text_data(pdf_path)
        text = "\n".join(filter(str.strip, text.splitlines()))

        # Load pre-trained GLiNER model with GPU support
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GLiNER.from_pretrained("/model/huggingface/gliner-multitask-large-v0.5").to(device)

        question = "Extract resume owner's name, location, email, phone, university, and degree information from the text"

        labels = ["names", "location", "email", "phone", "university", "degree"]

        input_ = question + text
        answers = model.predict_entities(input_, labels)
        print("Ans : ", answers)

        # Step 1: Group items by label
        grouped_data = {}
        for item in answers:
            label = item['label']
            if label not in grouped_data:
                grouped_data[label] = []
            grouped_data[label].append(item)

        # Step 2: Find items that meet the threshold for each label
        threshold = 0.50  # 80% threshold
        filtered_items = {}
        for label, items in grouped_data.items():
            # Filter items by score threshold
            filtered_items[label] = [item for item in items if item['score'] > threshold]

        # Step 3: Create the highest_items dictionary with only the first item for each label
        highest_items = {}
        for label, items in filtered_items.items():
            if items:
                highest_items[label] = items[0]['text'].replace("\n", " ")

        # Post-processing specific labels
        if 'email' in highest_items:
            highest_items['email'] = highest_items['email'].replace(" ", "")
        if 'location' in highest_items:
            del highest_items['location']

        # Check if phone number is missing and use regex to extract it if necessary
        if 'phone' not in highest_items:
            phone_match = re.search(r'\b(\+8801|01)\d{9}\b', text)
            if phone_match:
                highest_items['phone'] = phone_match.group(0)

        return jsonify({'text': text, 'extracted_info': highest_items})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
