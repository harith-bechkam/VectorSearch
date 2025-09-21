import mimetypes
import fitz  # PyMuPDF
import docx
import pandas as pd
from PIL import Image
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
from striprtf.striprtf import rtf_to_text
from pptx import Presentation
from odf import text, teletype
from odf.opendocument import load
from sentence_transformers import SentenceTransformer
import numpy as np
import pytesseract
import cv2
import json

# CLIP model (512-dim)
clip_model = SentenceTransformer("clip-ViT-B-32")


def extract_text_from_pdf(file_path, ocr_threshold=20):
    """Extract text from PDF, automatically OCR if page is likely scanned."""
    try:
        doc = fitz.open(file_path)
        full_text = ""
        for page in doc:
            page_text = page.get_text()
            if len(page_text.strip()) < ocr_threshold:
                # Page likely scanned → use OCR
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img)
                full_text += ocr_text + "\n"
            else:
                full_text += page_text + "\n"
        return full_text
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return None


def extract_text(file_path, mime_type):
    """Extract text from various document types."""
    try:
        # PDF
        if mime_type == "application/pdf":
            return extract_text_from_pdf(file_path)

        # Word
        elif mime_type in [
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]:
            doc = docx.Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])

        # Excel
        elif mime_type in [
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
            "application/x-vnd.oasis.opendocument.spreadsheet"
        ]:
            text_data = ""
            if file_path.endswith(".xlsx"):
                df = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
            elif file_path.endswith(".xls"):
                df = pd.read_excel(file_path, sheet_name=None, engine="xlrd")
            elif file_path.endswith(".ods"):
                doc = load(file_path)
                sheets = doc.spreadsheet.getElementsByType(text.Table)
                for sheet in sheets:
                    rows = sheet.getElementsByType(text.TableRow)
                    for row in rows:
                        cells = row.getElementsByType(text.TableCell)
                        text_data += " | ".join([teletype.extractText(c) for c in cells]) + "\n"
                return text_data
            else:
                return ""
            for sheet_name, sheet in df.items():
                text_data += f"Sheet: {sheet_name}\n"
                text_data += "\n".join(sheet.astype(str).apply(lambda row: " | ".join(row), axis=1))
                text_data += "\n"
            return text_data

        # CSV
        elif mime_type == "text/csv":
            df = pd.read_csv(file_path)
            return "\n".join(df.astype(str).apply(lambda row: " | ".join(row), axis=1))

        # PowerPoint
        elif mime_type in ["application/vnd.ms-powerpoint",
                           "application/vnd.openxmlformats-officedocument.presentationml.presentation"]:
            prs = Presentation(file_path)
            text_data = ""
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text_data += shape.text + "\n"
            return text_data

        # Rich Text
        elif mime_type == "application/rtf":
            with open(file_path, "r", encoding="utf-8") as f:
                rtf_content = f.read()
            return rtf_to_text(rtf_content)

        # Text, HTML, XML, JSON, YAML, Markdown, log
        elif mime_type and (mime_type.startswith("text/") or
                            mime_type in ["application/json", "application/xml", "text/markdown", "text/yaml"]):
            if file_path.endswith(".json"):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return json.dumps(data)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()

    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return None


def audio_to_spectrogram_image(file_path):
    """Convert audio to spectrogram image for CLIP."""
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(3, 3), dpi=80)
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, ax=ax, cmap="magma")
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def extract_embedding(file_path):
    """
    Extract 512-dim CLIP embedding AND return extracted content.
    Returns: (embedding, extracted_content)
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    extracted_content = None

    # Text / Office / CSV / PPTX / JSON / YAML
    text = extract_text(file_path, mime_type)
    if text and text.strip():
        extracted_content = text
        emb = clip_model.encode(text, convert_to_numpy=True)
        return emb.tolist(), extracted_content

    # Image
    if mime_type and mime_type.startswith("image/"):
        img = Image.open(file_path).convert("RGB")
        emb = clip_model.encode(img, convert_to_numpy=True)
        extracted_content = ""
        # OCR if image has potential text
        try:
            ocr_text = pytesseract.image_to_string(img)
            if len(ocr_text.strip()) > 5:
                text_emb = clip_model.encode(ocr_text, convert_to_numpy=True)
                emb = (emb + text_emb) / 2
                extracted_content = ocr_text
        except Exception as e:
            print(f"OCR error for image {file_path}: {e}")
        return emb.tolist(), extracted_content

    # Audio
    if mime_type and mime_type.startswith("audio/"):
        img = audio_to_spectrogram_image(file_path)
        emb = clip_model.encode(img, convert_to_numpy=True)
        extracted_content = "[Audio spectrogram]"
        return emb.tolist(), extracted_content

    # Video
    if mime_type and mime_type.startswith("video/"):
        try:
            cap = cv2.VideoCapture(file_path)
            frame_embeddings = []
            ocr_text_accum = ""
            for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                ret, frame = cap.read()
                if not ret:
                    break
                img = Image.fromarray(frame).convert("RGB")
                emb_frame = clip_model.encode(img, convert_to_numpy=True)
                # OCR if frame has text
                try:
                    ocr_text = pytesseract.image_to_string(img)
                    if len(ocr_text.strip()) > 5:
                        text_emb = clip_model.encode(ocr_text, convert_to_numpy=True)
                        emb_frame = (emb_frame + text_emb) / 2
                        ocr_text_accum += ocr_text + "\n"
                except Exception as e:
                    print(f"OCR error in video frame: {e}")
                frame_embeddings.append(emb_frame)
            cap.release()
            if frame_embeddings:
                emb = np.mean(frame_embeddings, axis=0)
                extracted_content = ocr_text_accum.strip() or "[Video frames]"
                return emb.tolist(), extracted_content
        except Exception as e:
            print(f"Video processing error: {e}")
        return [0.0] * 512, None

    # Unsupported / binary formats → zero vector
    return [0.0] * 512, None
