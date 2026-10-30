# Project 4 — Image or Text Recognition (Basic)

**Author:** AI Engineering Intern
**Organization:** DecodeLabs
**Batch:** 2026
**Track:** Artificial Intelligence — Optional Mastery Phase

## 1. Executive Summary
Project 4 marks the graduation from structured data (CSVs, spreadsheets) to unstructured data (images). Over 80% of enterprise data lives in unstructured formats like scanned documents and video feeds. 

This project implements **Path 1: Optical Character Recognition (OCR)**. We built a pipeline that ingests raw visual data, applies systematic image pre-processing, feeds it into a pre-trained Deep Learning model (Tesseract), and outputs machine-readable text while strictly enforcing an 80% confidence threshold to prevent hallucinations.

## 2. Methodology — The IPO Framework

| Phase | Implementation |
| :--- | :--- |
| **INPUT** | Load a raw image (simulating an invoice document). |
| **PROCESS** | OpenCV Pre-processing → Tesseract OCR → Softmax Confidence Filtering. |
| **OUTPUT** | Extracted text strings + visual confirmation of the thresholded image. |

## 3. Systematic Image Pre-Processing
Raw images are cluttered with shadows, chromatic noise, and uneven lighting. We used `OpenCV` to clean the data:
1. **Grayscale Conversion:** Collapses the 3D RGB matrix (Height x Width x 3) into a 1D intensity matrix. 
2. **Gaussian Blur:** Smooths the image to eliminate micro-imperfections.
3. **Adaptive Thresholding (Otsu's Method):** Forces every pixel into a binary choice (Black or White). This separates the text foreground from the noisy background.

## 4. The Confidence Filter (The 80% Gate)
An AI does not "know" what an object is; it calculates statistical probability. If we accept all outputs, the model will hallucinate.
By extracting the confidence score from Tesseract's output, we applied a strict filter:
`if confidence >= 0.80: accept_text() else: drop_detection()`
This minimizes False Positives, ensuring only the most legible text is returned to the user.

## 5. Transfer Learning
We did not train a model from scratch. We leveraged **Transfer Learning** by utilizing Google's Tesseract engine, a pre-trained model that already understands universal visual concepts (edges, shapes, gradients) from millions of training images. We simply plugged our specific text-extraction task into its existing knowledge base.

## 6. Results
The pipeline successfully extracted the text "DECODELABS INVOICE" and "Total: $1,499.00" from the sample image, dropping the artificial noise and low-confidence artifacts. The generated `processed_image.png` confirmed that the thresholding successfully isolated the characters from the background.

## 7. Conclusion
By controlling the IPO model, image pre-processing, and confidence thresholds, we have dictated how the machine perceives reality. We have built the bridge between the physical world (pixels) and computational logic (text).