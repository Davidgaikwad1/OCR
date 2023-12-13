import os
import cv2
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from tqdm import tqdm

def convert_pdf_to_images(pdf_path, output_folder="output_images", dpi=500, poppler_path=None):
    # Create a new folder for output images
    os.makedirs(output_folder, exist_ok=True)

    # Convert PDF to images
    images = convert_from_path(pdf_path, dpi, poppler_path=poppler_path)

    # Save each page as a separate JPEG image
    for i, image in tqdm(enumerate(images), desc="Converting PDF to Images", total=len(images)):
        image.save(f"{output_folder}/page{i + 1}.jpg", "JPEG")

    return images  # Return the list of images

def process_image(image_path, output_folder="temp"):
    # Create a new folder for processed images
    os.makedirs(output_folder, exist_ok=True)

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{output_folder}/index_gray.jpg", gray)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    cv2.imwrite(f"{output_folder}/index_blur1.jpg", blur)

    thersh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cv2.imwrite(f"{output_folder}/index_thresh.jpg", thersh)

    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
    cv2.imwrite(f"{output_folder}/index_karnal.jpg", kernal)

    dilate = cv2.dilate(thersh, kernal, iterations=1)
    cv2.imwrite(f"{output_folder}/index_dilate.jpg", dilate)

    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])

    return image, cnts  # Return both image and cnts

def perform_ocr(image_path, output_folder="output_text"):
    # Create a new folder for OCR text results
    os.makedirs(output_folder, exist_ok=True)

    img = Image.open(image_path)
    ocr_results = pytesseract.image_to_string(img)
    
    # Save OCR results to a text file
    with open(f"{output_folder}/ocr_results.txt", 'a') as file:
        file.write(f"OCR Results for {image_path}:\n{ocr_results}\n\n")

def main(pdf_path, output_folder="output_images", poppler_path=None):
    images = convert_pdf_to_images(pdf_path, output_folder, poppler_path=poppler_path)
    
    for i in tqdm(range(1, len(images) + 1), desc="Processing OCR"):
        image_path = f"{output_folder}/page{i}.jpg"
        image, cnts = process_image(image_path)
        perform_ocr(image_path)

# Example usage
pdf_file_path = r"David_Gaikwad.pdf"
output_images_folder = r"E:\techIt\img"
poppler_bin_path = r"C:\Program Files\poppler-23.11.0\Library\bin"

main(pdf_file_path, output_images_folder, poppler_path=poppler_bin_path)
