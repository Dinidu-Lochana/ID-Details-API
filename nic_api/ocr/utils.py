import cv2
import pytesseract
import numpy as np
import re
from PIL import Image



def preprocess_image(image):
    # Check if image is a PIL image
    if isinstance(image, Image.Image):
        image = image.convert("RGB")  # Ensure it's in RGB mode
        image = np.array(image)  # Convert to NumPy array

    # Check if image is already a NumPy array but in the wrong format
    if not isinstance(image, np.ndarray):
        raise ValueError("Image is not a valid NumPy array or PIL Image.")

    print(f"Image dtype: {image.dtype}, shape: {image.shape}")  # Debugging

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return gray

def extract_text_from_image(image):
    processed_img = preprocess_image(image)
    return pytesseract.image_to_string(processed_img, lang="eng")

def extract_nic_details(front_text, back_text):
    """ Extracts relevant details from OCR text. """

    # ID Number
    nic_match = re.search(r'\b\d{12}\b', front_text)

    # Name
    name_match = re.search(r'([A-Z\s]{10,})', back_text)

    # DOB
    dob_match = re.search(r'\b\d{4}-\d{2}-\d{2}\b', back_text)

    # Address
    address_match = re.search(r'ADDRESS:\s*(.+)', back_text, re.IGNORECASE)

    return {
        "National ID Number": nic_match.group(0) if nic_match else "Not Found",
        "Full Name": name_match.group(0).strip() if name_match else "Not Found",
        "Date of Birth": dob_match.group(0) if dob_match else "Not Found",
        "Address": address_match.group(1).strip() if address_match else "Not Found"
    }