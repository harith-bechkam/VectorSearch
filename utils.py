import mimetypes
import fitz  # PyMuPDF
import docx
import numpy as np
from PIL import Image
import librosa

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # or any embedding model
emb = model.encode("hello world")
print(len(emb))

def extract_text(file_path, mime_type):
    """Extract text content from file."""
    if mime_type == "application/pdf":
        doc = fitz.open(file_path)
        text = "".join([page.get_text() for page in doc])
        return text
    elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif mime_type.startswith("text/"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return None

def extract_embedding(file_path):
    """Detect type and create embedding."""
    mime_type, _ = mimetypes.guess_type(file_path)

    # Text-based embedding
    text = extract_text(file_path, mime_type)
    if text:
        return model.encode(text).tolist()

    # Image embedding (placeholder: convert image to pixels)
    if mime_type.startswith("image/"):
        img = Image.open(file_path).resize((224, 224))
        arr = np.array(img).flatten()
        arr = arr / 255.0
        return arr[:768].tolist()  # truncate/pad to 768 dims

    # Audio embedding (placeholder: use mean MFCCs)
    if mime_type.startswith("audio/"):
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        vec = np.mean(mfcc, axis=1)
        return vec[:768].tolist()  # truncate/pad

    # Video embedding (placeholder: first frame as image)
    if mime_type.startswith("video/"):
        # For simplicity, treat as zeros
        return [0.0] * 768

    # Other unknown files
    return [0.0] * 768
