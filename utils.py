import mimetypes
import os
import io
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import fitz  # PyMuPDF
import docx
from pptx import Presentation
from odf.opendocument import load
from odf import text, teletype
import json
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import easyocr
import ocrmypdf
import whisper
from moviepy import VideoFileClip
from PIL import Image
import tempfile
import torch

# -------------------------------
# Initialize CLIP model & EasyOCR
# -------------------------------
clip_model = SentenceTransformer("clip-ViT-B-32")
ocr_reader = easyocr.Reader(['en'], gpu=False)  # GPU=True if available
# Load Whisper model once (can be "small", "base", "medium", "large")
whisper_model = whisper.load_model("small")  # GPU=True if available

# -------------------------------
# Helper functions
# -------------------------------

def audio_to_spectrogram_image(file_path: str) -> Image.Image:
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


def ocr_text_from_image_easyocr(img: Image.Image, min_chars=5) -> str:
    """Run OCR using EasyOCR."""
    try:
        img_np = np.array(img)
        results = ocr_reader.readtext(img_np)
        text = " ".join([res[1] for res in results])
        if len(text.strip()) >= min_chars:
            return text
    except Exception as e:
        print(f"EasyOCR error: {e}")
    return ""


def extract_text(file_path: str, mime_type: str) -> str:
    """Extract text from various document types (with OCR for scanned PDFs)."""
    try:
        # PDF
        if mime_type == "application/pdf":
            # Check if PDF has little text → likely scanned
            doc = fitz.open(file_path)
            total_text_len = sum(len(page.get_text().strip()) for page in doc)
            if total_text_len < 50:
                # Use OCRmyPDF to make PDF searchable
                ocr_file_path = "/tmp/ocr_output.pdf"
                ocrmypdf.ocr(file_path, ocr_file_path, force_ocr=True, language='eng')
                file_path = ocr_file_path
                doc = fitz.open(file_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n"
            return full_text

        # Word
        elif mime_type in ["application/msword",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            doc = docx.Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])

        # Excel / ODS
        elif mime_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           "application/vnd.ms-excel",
                           "application/x-vnd.oasis.opendocument.spreadsheet"]:
            text_data = ""
            if file_path.endswith(".xlsx"):
                df = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
            elif file_path.endswith(".xls"):
                df = pd.read_excel(file_path, sheet_name=None, engine="xlrd")
            elif file_path.endswith(".ods"):
                doc = load(file_path)
                sheets = doc.spreadsheet.getElementsByType(text.Table)
                for sheet in sheets:
                    for row in sheet.getElementsByType(text.TableRow):
                        cells = row.getElementsByType(text.TableCell)
                        text_data += " | ".join([teletype.extractText(c) for c in cells]) + "\n"
                return text_data
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

        # Text / JSON / YAML / Markdown / log
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
        print(f"Text extraction error from {file_path}: {e}")
    return ""


# -------------------------------
# Main embedding extraction
# -------------------------------

def extract_embedding(file_path: str):
    """
    Extract 512-dim CLIP embedding AND return extracted content.
    Returns: (embedding, extracted_content)
    """
    mime_type, _ = mimetypes.guess_type(file_path)

    # First, try text extraction
    text = extract_text(file_path, mime_type)
    if text.strip():
        emb = clip_model.encode(text, convert_to_numpy=True)
        return emb.tolist(), text

    # Image
    if mime_type and mime_type.startswith("image/"):
        img = Image.open(file_path).convert("RGB")
        emb = clip_model.encode(img, convert_to_numpy=True)
        ocr_text = ocr_text_from_image_easyocr(img)
        if ocr_text:
            text_emb = clip_model.encode(ocr_text, convert_to_numpy=True)
            emb = (emb + text_emb) / 2
        return emb.tolist(), ocr_text if ocr_text else ""

    # Audio
    if mime_type and mime_type.startswith("audio/"):
        img = audio_to_spectrogram_image(file_path)
        emb = clip_model.encode(img, convert_to_numpy=True)
        return emb.tolist(), "[Audio spectrogram]"

    # Video
    if mime_type and mime_type.startswith("video/"):
        try:
            cap = cv2.VideoCapture(file_path)
            frame_embeddings = []
            extracted_content = ""

            # ----------------------
            # Step 1: Sample frames
            # ----------------------
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_every = max(1, frame_count // 10)

            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret or i % sample_every != 0:
                    continue

                img = Image.fromarray(frame).convert("RGB")
                emb = clip_model.encode(img, convert_to_numpy=True)

                # OCR on frames
                ocr_text = ocr_text_from_image_easyocr(img)
                if ocr_text:
                    text_emb = clip_model.encode(ocr_text, convert_to_numpy=True)
                    emb = (emb + text_emb) / 2
                    extracted_content += ocr_text + "\n"

                frame_embeddings.append(emb)

            cap.release()

            # ----------------------
            # Step 2: Extract audio
            # ----------------------
            video_clip = VideoFileClip(file_path)
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_audio:
                video_clip.audio.write_audiofile(tmp_audio.name, logger=None)
                # Transcribe audio using Whisper
                result = whisper_model.transcribe(tmp_audio.name)
                audio_text = result.get("text", "")
                if audio_text.strip():
                    audio_emb = clip_model.encode(audio_text, convert_to_numpy=True)
                    if frame_embeddings:
                        frame_embeddings = [(f + audio_emb) / 2 for f in frame_embeddings]
                    else:
                        frame_embeddings.append(audio_emb)
                    extracted_content += "\n" + audio_text

            if frame_embeddings:
                return np.mean(frame_embeddings, axis=0).tolist(), extracted_content.strip()

        except Exception as e:
            print(f"Video processing error: {e}")

        return [0.0] * 512, ""

    # Unsupported / binary → zero vector
    return [0.0] * 512, ""
