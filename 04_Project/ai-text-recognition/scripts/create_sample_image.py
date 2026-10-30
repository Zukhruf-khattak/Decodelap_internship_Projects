"""Generates a sample image with text for the OCR pipeline."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

def main():
    out_path = Path("data") / "sample_invoice.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a white canvas
    img = Image.new("RGB", (800, 300), color="white")
    draw = ImageDraw.Draw(img)

    # Try to load a default font, fallback to default if missing
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    # Draw some sample text (simulating an invoice)
    draw.text((50, 50), "DECODELABS INVOICE #0042", fill="black", font=font)
    draw.text((50, 120), "Item: Neural Network Module", fill="black", font=font)
    draw.text((50, 190), "Total: $1,499.00", fill="black", font=font)

    # Add some artificial noise (black dots)
    for _ in range(500):
        x = random.randint(0, 800)
        y = random.randint(0, 300)
        draw.point((x, y), fill="gray")

    img.save(out_path)
    print(f"[OK] Generated sample image at {out_path}")

if __name__ == "__main__":
    main()