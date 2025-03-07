import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
from datetime import datetime, timedelta
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to remove background from the image to improve OCR accuracy
def remove_background(image):
    """
    Removes background from the image to improve OCR accuracy.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
        if len(image.shape) == 2:  # If grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Check if image has channels
    if len(image.shape) < 3:
        return image
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped = image[y:y+h, x:x+w]
    else:
        cropped = image
    
    return cropped

# Function to preprocess the image for better OCR accuracy
def preprocess_image(image, preprocess_type='default'):
    """
    Preprocesses the image to enhance text readability.
    
    Args:
        image: Input image
        preprocess_type: Type of preprocessing to apply ('default', 'text', 'number')
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
        if len(image.shape) == 2:  # If grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Remove background
    image = remove_background(image)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    if preprocess_type == 'text':
        # For text fields - higher contrast with less noise reduction
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Sharpen the image
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Binarization with higher threshold for text
        _, binary = cv2.threshold(sharpened, 160, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        return binary
        
    elif preprocess_type == 'number':
        # For numeric fields - more noise reduction
        enhanced = cv2.equalizeHist(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
        
    else:  # default
        # Balanced approach
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary

# Function to extract text from specific regions of the ID card
def extract_by_region(image):
    """
    Extracts text from specific regions of the ID card.
    """
    # Convert PIL Image to OpenCV format if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        if len(image.shape) == 2:  # If grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Get dimensions
    height, width = image.shape[:2]
    
    # Define approximate regions for different fields
    # These would need adjustment based on your specific ID card layout
    regions = {
        "name": {
            "coords": (int(width * 0.3), int(height * 0.2), int(width * 0.9), int(height * 0.35)),
            "preprocess": "text",
            "psm": 7  
        },
        "nic": {
            "coords": (int(width * 0.3), int(height * 0.35), int(width * 0.7), int(height * 0.45)),
            "preprocess": "number",
            "psm": 7  
        },
        "dob": {
            "coords": (int(width * 0.3), int(height * 0.45), int(width * 0.7), int(height * 0.55)),
            "preprocess": "number",
            "psm": 7  
        },
        "address": {
            "coords": (int(width * 0.3), int(height * 0.55), int(width * 0.9), int(height * 0.8)),
            "preprocess": "text",
            "psm": 6 
        }
    }
    
    results = {}
    
    for field, settings in regions.items():
        x1, y1, x2, y2 = settings["coords"]
        preprocess_type = settings["preprocess"]
        psm = settings["psm"]
        
        # Extract region
        if y1 >= height or x1 >= width or y2 <= 0 or x2 <= 0:
            logger.warning(f"Region coordinates for {field} out of bounds")
            continue
            
        # Ensure coordinates are valid
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid region dimensions for {field}")
            continue
            
        roi = image[y1:y2, x1:x2]
        
        # Skip if ROI is empty
        if roi.size == 0:
            logger.warning(f"Empty ROI for {field}")
            continue
        
        # Apply specialized preprocessing
        processed_roi = preprocess_image(roi, preprocess_type)
        
        # Debug: Save preprocessed region images
        # cv2.imwrite(f"debug_{field}_preprocessed.jpg", processed_roi)
        
        # Try multiple language configs
        texts = []
        lang_configs = [
            'sin', 'sin+tam'
        ]
        
        for lang in lang_configs:
            try:
                text = pytesseract.image_to_string(
                    processed_roi, 
                    lang=lang, 
                    config=f'--psm {psm}'
                )
                if text.strip():
                    texts.append(text.strip())
            except Exception as e:
                logger.error(f"Error with {lang} for {field}: {e}")
        
        # Store the best result
        if texts:
            results[field] = max(texts, key=len)
        else:
            results[field] = ""
    
    return results

# Function to extract text using Tesseract with multiple languages
def extract_text(image):
    """
    Extracts text using Tesseract OCR with multiple language support.
    """
    preprocessed = preprocess_image(image)
    texts = []
    
    # Try with different language configurations for better results
    lang_configs = [
        ('sin', '--psm 6'),
        ('sin+tam', '--psm 6'),
    ]
    
    for lang, config in lang_configs:
        try:
            text = pytesseract.image_to_string(preprocessed, lang=lang, config=config)
            texts.append(text)
        except Exception as e:
            logger.error(f"Tesseract error with {lang}: {e}")
    
    return texts

# Function to clean and validate the extracted text
def clean_and_validate_text(texts):
    """
    Cleans and filters extracted text.
    """
    cleaned_texts = []
    for text in texts:
        text = ''.join(char for char in text if char.isprintable())
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > 5:
            cleaned_texts.append(text)
    return cleaned_texts

# Function to extract NIC number
def extract_nic_number(text, region_data=None):
    """
    Extracts NIC number from text.
    """
    # First try region-specific data if available
    if region_data and 'nic' in region_data and region_data['nic']:
        # Clean the region data to keep only digits and 'V'/'v'
        cleaned = re.sub(r'[^\dVv]', '', region_data['nic'])
        
        # Check for valid NIC patterns
        if re.match(r'\d{9}[Vv]', cleaned) or re.match(r'\d{12}', cleaned):
            return cleaned
    
    # Patterns for old (9 digits + V/v) and new (12 digits) NIC formats
    patterns = [
        r'\b\d{9}[vV]\b',   # Old format with 'V'
        r'\b\d{9}v\b',      # Old format with lowercase 'v'
        r'\b\d{12}\b'       # New format (12 digits)
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0]
    
    # If no match found, try a more permissive pattern
    # This might include spaces or other separators
    permissive_pattern = r'\d{9}[\s-]*[vV]|\d{11,12}'
    matches = re.findall(permissive_pattern, text)
    if matches:
        # Clean up the result
        result = re.sub(r'[^\dVv]', '', matches[0])
        # Validate length after cleanup
        if len(result) == 10 and result[-1].upper() == 'V' or len(result) == 12:
            return result
    
    return None

# Function to extract name
def extract_name(text, region_data=None):
    """
    Enhanced function to extract name from text using patterns in Sinhala, Tamil, and English.
    Implements multiple extraction strategies and post-processing for better accuracy.
    """
    # First try region-specific data if available
    if region_data and 'name' in region_data and region_data['name']:
        name_text = region_data['name']
        
        # 1. Check if it contains name identifiers and extract accordingly
        name_identifiers = [ "නම", "පැයර්", "සම්පූර්ණ නම"]
        for identifier in name_identifiers:
            if identifier in name_text.lower():
                # Extract the part after the identifier
                match = re.search(f'{identifier}\s*[:;]?\s*([^\n]+)', name_text, re.IGNORECASE)
                if match:
                    return clean_name(match.group(1).strip())
                    
        # 2. If no identifier but the text looks like a name, use it directly
        name_candidate = name_text.strip()
        if len(name_candidate) > 3 and not re.search(r'\d{3,}', name_candidate):
            # Filter out likely non-name text
            if not any(x in name_candidate.lower() for x in ['address', 'nic', 'dob', 'ලිපිනය']):
                return clean_name(name_candidate)
    
    # 3. Try to find name in the whole text using various patterns
    # Patterns for name in different languages
    patterns = [
        # Sinhala
        r'නම\s*[:;]?\s*([අ-ෆ\s\.A-Za-z]+)',
        r'සම්පූර්ණ\s*නම\s*[:;]?\s*([අ-ෆ\s\.A-Za-z]+)',
        # Tamil
        r'பெயர்\s*[:;]?\s*([^\n]+)',
        # General pattern (look for content after name label)
        r'(?:නම|பெயர்)\s*[:;]?\s*([^\n]+)'
    ]
    
    name_candidates = []
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            clean_match = clean_name(match)
            if is_valid_name(clean_match):
                name_candidates.append(clean_match)
    
    # 4. Try to extract multi-word sequences that look like names (fallback)
    if not name_candidates:
        # Look for sequences of words that might be names (no numbers, reasonable length)
        words = re.findall(r'[A-Za-zඅ-ෆ][A-Za-zඅ-ෆ\s\.]{2,}[A-Za-zඅ-ෆ]', text)
        for word in words:
            if len(word) > 5 and is_valid_name(word):
                name_candidates.append(clean_name(word))
    
    # Return the best candidate
    if name_candidates:
        # Sort by length (prefer longer names) and then by quality score
        return max(name_candidates, key=lambda x: (name_quality_score(x), len(x)))
    
    return None

def clean_name(name):
    """
    Cleans a name string by removing unwanted characters and extra spaces.
    """
    if not name:
        return name
        
    # Remove unwanted characters, keeping alphanumeric, Sinhala chars, spaces, dots, commas, hyphens
    cleaned = re.sub(r'[^A-Za-zඅ-ෆ0-9\s\.,\-]', ' ', name)
    
    # Normalize spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Remove common prefix/suffix contaminants
    prefixes_to_remove = ['නම', 'සම්පූර්ණ නම',]
    for prefix in prefixes_to_remove:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    
    # Remove numeric suffixes/prefixes that might be ID numbers
    cleaned = re.sub(r'^\d+\s*', '', cleaned)
    cleaned = re.sub(r'\s*\d+$', '', cleaned)
    
    # Remove any leading/trailing punctuation
    cleaned = cleaned.strip('.,:-')
    
    return cleaned

def is_valid_name(name):
    """
    Checks if a string is likely to be a valid name.
    """
    if not name or len(name) < 3:
        return False
        
    # Must not contain too many digits
    if re.search(r'\d{3,}', name):
        return False
        
    # Check for very common non-name words
    non_name_keywords = ['address', 'nic', 'dob', 'ලිපිනය', 'card', 'id', 'republic']
    if any(keyword in name.lower() for keyword in non_name_keywords):
        return False
        
    # Check length (most names are between 5 and 50 characters)
    if len(name) < 5 or len(name) > 50:
        return False
        
    # Names usually have more than one word
    if ' ' not in name and '.' not in name:
        return False
        
    return True

def name_quality_score(name):
    """
    Assigns a quality score to a name candidate.
    Higher score means better quality name.
    """
    score = 0
    
    # Names with appropriate word count (2-5 words) get higher scores
    word_count = len(name.split())
    if 2 <= word_count <= 5:
        score += 3
    elif word_count > 5:
        score -= (word_count - 5)  # Penalize very long names
        
    # Names with mixed case (e.g., "John Smith") are likely proper names
    if re.search(r'[A-Z][a-z]', name):
        score += 2
        
    # Names with initials (like "J. Smith" or "A.B. Fernando") are likely proper names
    if re.search(r'[A-Za-z]\.\s', name):
        score += 2
        
    # Presence of Sinhala characters might indicate a name on a Sinhala ID
    if re.search(r'[අ-ෆ]', name):
        score += 1
        
    # Penalize very short or very long names
    if len(name) < 8:
        score -= 1
    if len(name) > 30:
        score -= 1
        
    return score

# Function to extract date of birth
def extract_dob(text, region_data=None, nic_number=None):
    """
    Extracts date of birth from text.
    """
    # First try region-specific data if available
    if region_data and 'dob' in region_data and region_data['dob']:
        # Look for date patterns in the region data
        date_matches = re.findall(r'\d{1,4}[-./]\d{1,2}[-./]\d{1,4}', region_data['dob'])
        if date_matches:
            return date_matches[0]
    
    # Patterns for date of birth in different formats and languages
    patterns = [
        # Sinhala
        r'උපන්\s*දිනය\s*[:]\s*(\d{2}[-./]\d{2}[-./]\d{4})',
        r'උපන්\s*දිනය\s*[:]\s*(\d{4}[-./]\d{2}[-./]\d{2})',
        # English
        r'Date of Birth\s*[:]\s*(\d{2}[-./]\d{2}[-./]\d{4})',
        r'DOB\s*[:]\s*(\d{2}[-./]\d{2}[-./]\d{4})',
        # Generic
        r'(?:Date of Birth|DOB|උපන්\s*දිනය|பிறந்த திகதி)\s*[:]\s*(\d{1,4}[-./]\d{1,2}[-./]\d{1,4})'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0]
    
    # If we have the NIC number, try to extract DOB from it
    if nic_number:
        dob_from_nic = extract_dob_from_nic(nic_number)
        if dob_from_nic:
            return dob_from_nic
    
    # Try to find any date format in the text as a fallback
    date_pattern = r'\b\d{1,4}[-./]\d{1,2}[-./]\d{1,4}\b'
    date_matches = re.findall(date_pattern, text)
    if date_matches:
        return date_matches[0]
    
    return None

# Function to extract date of birth from NIC number
def extract_dob_from_nic(nic):
    """
    Extracts date of birth from NIC number.
    """
    if not nic:
        return None
        
    try:
        # Old format (9 digits + V/v)
        if len(nic) == 10 and nic[9].upper() == 'V':
            year_days = int(nic[:7])
            # Determine century (if days > 500, it's a female, but year calculation is the same)
            gender_days = year_days % 1000
            if year_days < 5000000:  # Born before 2000
                year = 1900 + int(year_days / 1000)
            else:  # Born after 2000
                year = 2000 + int(year_days / 1000) - 5
            
            # Calculate the date
            # Days are counted from Jan 1st (day 1)
            date = datetime(year, 1, 1) + timedelta(days=gender_days-1)
            return date.strftime('%Y-%m-%d')
            
        # New format (12 digits)
        elif len(nic) == 12:
            # The year is directly expressed in the first 4 digits
            year = int(nic[0:4])
            # Days are in digits 5-8
            days = int(nic[4:7])
            # Adjust for gender
            if days > 500:
                days -= 500
            
            # Calculate the date
            date = datetime(year, 1, 1) + timedelta(days=days-1)
            return date.strftime('%Y-%m-%d')
            
    except Exception as e:
        logger.error(f"Error extracting DOB from NIC {nic}: {e}")
    
    return None

# Function to extract address
def extract_address(text, region_data=None):
    """
    Extracts address from text.
    """
    # First try region-specific data if available
    if region_data and 'address' in region_data and region_data['address']:
        # Check if it contains address identifiers
        if "address" in region_data['address'].lower() or "ලිපිනය" in region_data['address']:
            # Extract the part after the identifier
            match = re.search(r'(?:Address|ලිපිනය|முகவரி)\s*[:]\s*([^\n]+(?:\n[^\n]+)*)', region_data['address'], re.IGNORECASE)
            if match:
                return match.group(1).strip()
        else:
            # If the text is long enough and contains typical address components
            address_candidate = region_data['address'].strip()
            if len(address_candidate) > 10 and re.search(r'\d+|road|street|මාවත|පාර', address_candidate, re.IGNORECASE):
                return address_candidate
    
    # Patterns for address in different languages
    patterns = [
        # Sinhala
        r'ලිපිනය\s*[:]\s*([අ-ෆ0-9\s,\./-]+)',
        # English
        r'Address\s*[:]\s*([A-Za-z0-9\s,\./-]+)',
        # Generic
        r'(?:Address|ලිපිනය|முகவரி)\s*[:]\s*([^\n]+(?:\n[^\n]+)*)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Return the longest match (most likely to be complete)
            return max(matches, key=len).strip()
    
    return None

# Main function to extract all NIC details
def extract_nic_details(front_image, back_image):
    """
    Extracts NIC details from front and back images by calling individual extraction functions.
    """
    # Extract text from both sides of the ID
    front_texts = extract_text(front_image)
    back_texts = extract_text(back_image)
    
    # Clean and combine all extracted text
    all_texts = clean_and_validate_text(front_texts + back_texts)
    combined_text = ' '.join(all_texts)
    
    # Try region-based extraction on both images
    front_regions = extract_by_region(front_image)
    back_regions = extract_by_region(back_image)
    
    # Combine region data
    region_data = {}
    for field in ['name', 'nic', 'dob', 'address']:
        front_value = front_regions.get(field, '')
        back_value = back_regions.get(field, '')
        # Choose the non-empty value or the longer one if both exist
        if front_value and back_value:
            region_data[field] = front_value if len(front_value) >= len(back_value) else back_value
        else:
            region_data[field] = front_value or back_value
    
    # Add additional preprocessing specifically for name extraction
    # Try to enhance name region detection
    if 'name' not in region_data or not region_data['name']:
        # Look for name in the top portions of both images
        height_front, width_front = front_image.shape[:2] if isinstance(front_image, np.ndarray) else (front_image.size[1], front_image.size[0])
        top_region_front = front_image[0:int(height_front*0.3), 0:width_front] if isinstance(front_image, np.ndarray) else front_image.crop((0, 0, width_front, int(height_front*0.3)))
        
        # Process this region specifically for text
        top_processed = preprocess_image(top_region_front, 'text')
        top_text = pytesseract.image_to_string(top_processed, lang='sin+eng', config='--psm 6')
        
        if top_text and len(top_text) > 5:
            region_data['name'] = top_text
    
    # Extract each detail using specialized functions
    nic_number = extract_nic_number(combined_text, region_data)
    name = extract_name(combined_text, region_data)
    dob = extract_dob(combined_text, region_data, nic_number)
    address = extract_address(combined_text, region_data)
    
    # English translations for the field labels
    details = {
        "ජාතික හැඳුනුම්පත් අංකය": nic_number if nic_number else "සොයාගත නොහැක",
        "සම්පූර්ණ නම": name if name else "සොයාගත නොහැක",
        "උපන්දිනය": dob if dob else "සොයාගත නොහැක",
        "ලිපිනය": address if address else "සොයාගත නොහැක"
    }
    
    return details