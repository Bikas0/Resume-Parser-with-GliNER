from gliner import GLiNER
import PyPDF2 as pdf
from PyPDF2 import PdfReader, PdfWriter
from pytess_read import text_data
from PyPDF2 import PdfReader
from gliner import GLiNER
from pytess_read import text_data
import torch
import os
import re

# Function to Extract Text From PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path,"rb") as f:
        reader = PdfReader(f)
        results = []
        for i in range(0,len(reader.pages)): # prev read.getNumPages()
            selected_page = reader.pages[i]
            text = selected_page.extract_text()
            results.append(text)
        return ' '.join(results) # convert list to a single doc


# creating a pdf reader object
# text = extract_text_from_pdf("/home/bikas/resume_parser/cv/sanjida_resume.pdf")

text = text_data("/home/bikas/resume_parser/cv/Md Bikasuzzaman Resume.pdf")
text = "\n".join(filter(str.strip, text.splitlines()))
# print(text)

model = GLiNER.from_pretrained("/model/huggingface/gliner-multitask-large-v0.5")
# model = GLiNER.from_pretrained("/model/huggingface/knowledgator/gliner-multitask-large-v0.5")
question = "Extarct resume owners name, location, email, phone, university, and degree information from the text"

labels = ["names", "location", "email", "phone", "university", "degree"]

input_ = question+text
answers = model.predict_entities(input_, labels)
print(answers)

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
import re

# Check if phone number is missing or invalid and use regex to extract it if necessary
phone_regex = r'\b(\+8801|01)\d{9}\b'

# Check if phone number is missing or invalid and use regex to extract it if necessary
if 'phone' not in highest_items or highest_items.get('phone') == 'phone':
    # Search for phone number in the text using regex
    phone_match = re.search(phone_regex, text)
    if phone_match:
        # Update highest_items with the extracted phone number
        highest_items['phone'] = phone_match.group(0)
    else:
        # If phone number is not found, set it to blank
        highest_items['phone'] = ''

# Output the results
print(highest_items)

