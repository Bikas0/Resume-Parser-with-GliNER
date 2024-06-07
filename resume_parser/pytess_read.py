import cv2
import os
import pytesseract
from pdf2image import convert_from_path

###   PyTesseract Setup   ###
tess_path = "/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = tess_path
os.environ["PATH"] += os.pathsep + os.path.dirname(tess_path)
os.environ["TESSDATA_PREFIX"] = "/home/bikas/resume_parser/tessdata"


def pytesseract_text_extractor(image_path):
    filenames = []
    info = ''
    for file in sorted(os.listdir(image_path)):
        filenames.append(file)

    for file in filenames:
        img = cv2.imread(os.path.join(image_path,file))
        
        # print(img.shape)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # text = pytesseract.image_to_string(gray, lang='ben+eng',config='--psm 8 ')
        text = pytesseract.image_to_string(gray,lang='Bengali',config='--psm 4 ')
        os.remove(os.path.join(image_path,file))
        info += text
        
    info  = "Hello, My Name is " + info   
    return info


def pdf_to_jpg(pdf_files, output_folder):
    # for pdf_file in pdf_files:
    # Get the base filename without extension
    base_filename = os.path.splitext(os.path.basename(pdf_files))[0]
    # Convert PDF to images
    # images = convert_from_path(pdf_file, poppler_path='C:/Program Files/poppler-24.02.0/Library/bin')
    images = convert_from_path(pdf_files)
    for i, image in enumerate(images):
        # Save each page as JPG
        image.save(f"{output_folder}/{base_filename}_page_0{i+1}.jpg", "JPEG")
    result = pytesseract_text_extractor(output_folder)
    return result


# Convert PDF files to JPG images
def text_data(pdf_directory):
    output_folder = "output_image"
    os.makedirs(output_folder, exist_ok=True)
    text = pdf_to_jpg(pdf_directory, output_folder)
    return text