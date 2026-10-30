"""Entry point — Project 4: Image/Text Recognition (DecodeLabs).

IPO Framework:
    INPUT:   Load raw image.
    PROCESS: Pre-process (grayscale, blur, threshold) -> OCR -> Filter by confidence.
    OUTPUT:  Display extracted text + visual confirmation of pre-processed image.
"""

from pathlib import Path
from src.preprocessor import preprocess_image
from src.ocr_engine import extract_text_with_confidence

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("  Project 4: AI Text Recognition (OCR)")
    print("=" * 60)

    input_image = DATA_DIR / "sample_invoice.png"
    processed_image = OUTPUT_DIR / "processed_image.png"

    # 1. INPUT & PROCESS: Pre-processing
    print("\n[1/3] Pre-processing image (Grayscale -> Blur -> Threshold)...")
    try:
        processed_matrix = preprocess_image(
            image_path=str(input_image),
            output_path=str(processed_image)
        )
        print(f"      [OK] Saved pre-processed image to {processed_image}")
    except Exception as e:
        print(f"      [X] Error: {e}")
        return

    # 2. PROCESS: OCR with 80% Confidence Gate
    print("\n[2/3] Running Tesseract OCR (Filtering confidence < 80%)...")
    df_results = extract_text_with_confidence(
        processed_matrix, 
        confidence_threshold=80
    )

    # 3. OUTPUT: Display Results
    print("\n[3/3] Extracted Text Results:")
    print("-" * 50)
    
    if df_results.empty:
        print("No text found above 80% confidence.")
    else:
        # Reconstruct the text into a readable string
        final_text = " ".join(df_results["text"].tolist())
        print(final_text)
        print("-" * 50)
        print(f"Total high-confidence words: {len(df_results)}")
        
        # Show detailed breakdown
        print("\nDetailed Breakdown (Text : Confidence):")
        for row in df_results.itertuples():
            print(f"  - {row.text:<15} : {row.confidence}%")

if __name__ == "__main__":
    main()