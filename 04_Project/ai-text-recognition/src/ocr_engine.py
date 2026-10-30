"""Phase 2 & 3: PROCESS & OUTPUT — OCR Engine.

Uses pytesseract to extract text and CONFIDENCE SCORES.
Filters out any text where the model's confidence is below 80%.
"""

import pytesseract
import pandas as pd
from io import StringIO

# --- IMPORTANT: SET YOUR TESSERACT PATH HERE ---
# If you installed Tesseract in the default location, uncomment the line below.
# If you installed it elsewhere, update the path.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text_with_confidence(image, confidence_threshold: int = 80) -> pd.DataFrame:
    """Extracts text and filters by confidence score.
    
    Args:
        image: The pre-processed OpenCV image matrix.
        confidence_threshold: Minimum confidence (0-100) to accept a word.
        
    Returns:
        DataFrame with columns: 'text', 'conf'
    """
    # Get detailed TSV data from Tesseract
    data = pytesseract.image_to_data(
        image, 
        output_type=pytesseract.Output.DICT,
        config="--psm 6" # PSM 6 = Assume a single uniform block of text
    )

    results = []
    
    # Iterate over detected words
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = int(float(data['conf'][i])) # Confidence is returned as string
        
        # If text is empty or confidence is below the threshold, skip it
        if not text or conf < confidence_threshold:
            continue
            
        results.append({"text": text, "confidence": conf})

    return pd.DataFrame(results)