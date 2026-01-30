#!/usr/bin/env python3
"""
Dual-Stream OCR Pipeline for Construction Blueprints
=====================================================

Three parallel extraction streams for speed and accuracy:
  Stream 1: Python text extraction (PyMuPDF + pdfplumber + pdfminer.six)
  Stream 2: Mistral OCR API (mistral-ocr-latest)
  Stream 3: Google Document AI OCR

No vision model. No LLM synthesis. Raw results returned directly.
"""

import sys
import os
import io
import json
import re
import base64
import time
from typing import Dict, List, Any, Tuple
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

# PDF Processing
import fitz  # PyMuPDF

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("Warning: pdfplumber not installed", file=sys.stderr)

try:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTFigure
    HAS_PDFMINER = True
except ImportError:
    HAS_PDFMINER = False
    print("Warning: pdfminer.six not installed", file=sys.stderr)

try:
    from mistralai import Mistral
    HAS_MISTRAL = True
except ImportError:
    HAS_MISTRAL = False
    print("Warning: mistralai not installed, Stream 2 (OCR) will be disabled", file=sys.stderr)

try:
    from google.cloud import documentai_v1 as documentai
    from google.api_core.client_options import ClientOptions
    HAS_DOCUMENT_AI = True
except ImportError:
    HAS_DOCUMENT_AI = False
    print("Warning: google-cloud-documentai not installed, Stream 3 will be disabled", file=sys.stderr)

from openai import OpenAI as OpenAIClient

from dotenv import load_dotenv
load_dotenv()


def get_openai_client():
    """Create OpenAI client for vision fallback on images."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAIClient(api_key=api_key)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Content detection
    CONTENT_DETECTION_DPI = 72
    CONTENT_PADDING = 30
    BLANK_THRESHOLD = 0.995
    MIN_CONTENT_RATIO = 0.02

    # Classification
    VECTOR_COMPLEXITY_THRESHOLD = 500
    TEXT_DENSITY_THRESHOLD = 0.3

    # Mistral OCR
    MISTRAL_OCR_MODEL = "mistral-ocr-latest"

    # Retry
    MAX_RETRIES = 3
    INITIAL_RETRY_DELAY = 1.0


# =============================================================================
# DOCUMENT TYPES
# =============================================================================

class DocumentType(Enum):
    CAD_VECTOR = "cad_vector"
    CAD_RASTER = "cad_raster"
    SCANNED = "scanned"
    TEXT_HEAVY = "text_heavy"
    MIXED = "mixed"
    UNKNOWN = "unknown"


# =============================================================================
# CONTENT DETECTOR - Find actual drawing bounds, remove white space
# =============================================================================

class ContentDetector:
    """Detects actual content area, removing white space margins."""

    @staticmethod
    def detect_content_bounds(page_fitz: fitz.Page) -> Tuple[fitz.Rect, Dict[str, Any]]:
        rect = page_fitz.rect
        stats = {
            "original_size": (rect.width, rect.height),
            "content_ratio": 1.0,
            "method": "default"
        }

        # Method 1: Vector drawings bounding box
        drawings = page_fitz.get_drawings()
        if drawings:
            min_x, min_y = rect.width, rect.height
            max_x, max_y = 0, 0

            for d in drawings:
                if d.get("rect"):
                    try:
                        d_rect = fitz.Rect(d["rect"])
                        min_x = min(min_x, d_rect.x0)
                        min_y = min(min_y, d_rect.y0)
                        max_x = max(max_x, d_rect.x1)
                        max_y = max(max_y, d_rect.y1)
                    except Exception:
                        continue

            if max_x > min_x and max_y > min_y:
                padding = Config.CONTENT_PADDING
                content_rect = fitz.Rect(
                    max(0, min_x - padding),
                    max(0, min_y - padding),
                    min(rect.width, max_x + padding),
                    min(rect.height, max_y + padding)
                )
                content_area = content_rect.width * content_rect.height
                page_area = rect.width * rect.height
                stats["content_ratio"] = content_area / page_area if page_area > 0 else 1.0
                stats["method"] = "drawings"
                stats["content_size"] = (content_rect.width, content_rect.height)

                if stats["content_ratio"] < 0.85:
                    return content_rect, stats

        # Method 2: Pixel-based detection
        try:
            pix = page_fitz.get_pixmap(matrix=fitz.Matrix(Config.CONTENT_DETECTION_DPI / 72.0, Config.CONTENT_DETECTION_DPI / 72.0))
            samples = pix.samples
            n = pix.n
            width, height = pix.width, pix.height

            min_x, min_y = width, height
            max_x, max_y = 0, 0
            white_threshold = 250
            step = 3

            for y in range(0, height, step):
                for x in range(0, width, step):
                    idx = (y * width + x) * n
                    if idx + n <= len(samples):
                        if n >= 3:
                            avg = (samples[idx] + samples[idx + 1] + samples[idx + 2]) / 3
                        else:
                            avg = samples[idx]
                        if avg < white_threshold:
                            min_x = min(min_x, x)
                            min_y = min(min_y, y)
                            max_x = max(max_x, x)
                            max_y = max(max_y, y)

            if max_x > min_x and max_y > min_y:
                scale = 72.0 / Config.CONTENT_DETECTION_DPI
                padding = Config.CONTENT_PADDING
                content_rect = fitz.Rect(
                    max(0, min_x * scale - padding),
                    max(0, min_y * scale - padding),
                    min(rect.width, max_x * scale + padding),
                    min(rect.height, max_y * scale + padding)
                )
                content_area = content_rect.width * content_rect.height
                page_area = rect.width * rect.height
                stats["content_ratio"] = content_area / page_area if page_area > 0 else 1.0
                stats["method"] = "pixel_scan"
                stats["content_size"] = (content_rect.width, content_rect.height)

                if stats["content_ratio"] < 0.90:
                    return content_rect, stats

        except Exception as e:
            print(f"Pixel-based content detection failed: {e}", file=sys.stderr)

        stats["method"] = "full_page"
        return rect, stats


# =============================================================================
# DOCUMENT CLASSIFIER
# =============================================================================

class DocumentClassifier:

    @staticmethod
    def classify_page(page_fitz: fitz.Page, page_plumber=None) -> Tuple[DocumentType, Dict[str, Any]]:
        stats = {
            "text_char_count": 0,
            "vector_count": 0,
            "image_count": 0,
            "table_count": 0,
            "has_searchable_text": False,
            "text_coverage": 0.0
        }

        rect = page_fitz.rect
        page_area = rect.width * rect.height

        text = page_fitz.get_text()
        stats["text_char_count"] = len(text.strip())
        stats["has_searchable_text"] = stats["text_char_count"] > 50

        drawings = page_fitz.get_drawings()
        stats["vector_count"] = len(drawings) if drawings else 0

        images = page_fitz.get_images()
        stats["image_count"] = len(images)

        text_dict = page_fitz.get_text("dict")
        text_area = 0
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:
                bbox = block.get("bbox", [0, 0, 0, 0])
                text_area += (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        stats["text_coverage"] = text_area / page_area if page_area > 0 else 0

        if page_plumber:
            try:
                tables = page_plumber.find_tables()
                stats["table_count"] = len(tables) if tables else 0
            except Exception:
                pass

        if stats["vector_count"] > Config.VECTOR_COMPLEXITY_THRESHOLD:
            doc_type = DocumentType.CAD_VECTOR if stats["has_searchable_text"] else DocumentType.CAD_RASTER
        elif stats["image_count"] > 0 and not stats["has_searchable_text"]:
            doc_type = DocumentType.SCANNED
        elif stats["text_coverage"] > Config.TEXT_DENSITY_THRESHOLD:
            doc_type = DocumentType.TEXT_HEAVY
        elif stats["has_searchable_text"]:
            doc_type = DocumentType.MIXED
        else:
            doc_type = DocumentType.SCANNED

        return doc_type, stats

    @staticmethod
    def classify_document(pdf_fitz: fitz.Document, pdf_plumber=None) -> Tuple[DocumentType, List[Dict[str, Any]]]:
        page_stats = []
        type_counts = {}

        for i, page in enumerate(pdf_fitz):
            plumber_page = pdf_plumber.pages[i] if pdf_plumber and i < len(pdf_plumber.pages) else None
            page_type, stats = DocumentClassifier.classify_page(page, plumber_page)
            stats["page_num"] = i
            stats["page_type"] = page_type
            page_stats.append(stats)
            type_counts[page_type] = type_counts.get(page_type, 0) + 1

        if not type_counts:
            return DocumentType.UNKNOWN, page_stats

        dominant_type = max(type_counts, key=type_counts.get)
        return dominant_type, page_stats


# =============================================================================
# STREAM 1: PYTHON TEXT EXTRACTION
# =============================================================================

class PythonTextExtractor:
    """Extract ALL text using Python libraries only (no API calls)."""

    @staticmethod
    def extract_pymupdf(page: fitz.Page) -> Dict[str, Any]:
        result = {
            "text": "",
            "text_blocks": [],
            "drawings_count": 0
        }

        result["text"] = page.get_text()

        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:
                block_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "") + " "
                if block_text.strip():
                    result["text_blocks"].append({
                        "text": block_text.strip(),
                        "bbox": block.get("bbox"),
                    })

        drawings = page.get_drawings()
        result["drawings_count"] = len(drawings) if drawings else 0

        return result

    @staticmethod
    def extract_pdfplumber(page) -> Dict[str, Any]:
        result = {
            "text": "",
            "words": [],
            "tables": [],
        }

        if not page:
            return result

        try:
            result["text"] = page.extract_text() or ""

            words = page.extract_words(
                x_tolerance=3,
                y_tolerance=3,
                keep_blank_chars=True,
                extra_attrs=['fontname', 'size']
            )
            result["words"] = [
                {"text": w["text"], "x0": w["x0"], "y0": w["top"], "x1": w["x1"], "y1": w["bottom"]}
                for w in (words or [])
            ]

            tables = page.extract_tables()
            result["tables"] = tables if tables else []

        except Exception as e:
            print(f"pdfplumber extraction error: {e}", file=sys.stderr)

        return result

    @staticmethod
    def extract_pdfminer(pdf_bytes: bytes, page_num: int) -> Dict[str, Any]:
        """Extract text using pdfminer.six - best for overlapping/rotated text."""
        result = {
            "text": "",
            "elements": [],
        }

        if not HAS_PDFMINER:
            return result

        try:
            laparams = LAParams(
                detect_vertical=True,
                all_texts=True,
                line_margin=0.3,
                word_margin=0.1,
                char_margin=2.0,
                boxes_flow=None,  # No forced reading order - capture everything
            )

            pdf_file = io.BytesIO(pdf_bytes)
            for i, page_layout in enumerate(extract_pages(pdf_file, laparams=laparams)):
                if i != page_num:
                    continue

                page_texts = []

                def extract_from_layout(element, depth=0):
                    if isinstance(element, (LTTextBox, LTTextLine)):
                        text = element.get_text().strip()
                        if text:
                            page_texts.append(text)
                            result["elements"].append({
                                "text": text,
                                "bbox": [element.x0, element.y0, element.x1, element.y1],
                            })
                    elif isinstance(element, LTFigure):
                        for child in element:
                            extract_from_layout(child, depth + 1)

                for element in page_layout:
                    extract_from_layout(element)

                result["text"] = "\n".join(page_texts)
                break

        except Exception as e:
            print(f"pdfminer extraction error on page {page_num}: {e}", file=sys.stderr)

        return result

    @staticmethod
    def extract_all_pages(pdf_fitz: fitz.Document, pdf_plumber, pdf_bytes: bytes) -> Dict[str, Any]:
        """Extract text from all pages using all three libraries."""
        pages = []

        for i in range(len(pdf_fitz)):
            print(f"  Stream 1: Extracting page {i + 1}...", file=sys.stderr)

            # PyMuPDF
            pymupdf_data = PythonTextExtractor.extract_pymupdf(pdf_fitz[i])

            # pdfplumber
            plumber_page = pdf_plumber.pages[i] if pdf_plumber and i < len(pdf_plumber.pages) else None
            plumber_data = PythonTextExtractor.extract_pdfplumber(plumber_page) if HAS_PDFPLUMBER else {"text": "", "words": [], "tables": []}

            # pdfminer.six
            pdfminer_data = PythonTextExtractor.extract_pdfminer(pdf_bytes, i)

            pages.append({
                "page_num": i + 1,
                "pymupdf_text": pymupdf_data["text"],
                "pymupdf_blocks": pymupdf_data["text_blocks"],
                "pdfplumber_text": plumber_data["text"],
                "pdfplumber_words": plumber_data["words"],
                "pdfminer_text": pdfminer_data["text"],
                "pdfminer_elements": pdfminer_data["elements"],
                "tables": plumber_data["tables"],
                "drawings_count": pymupdf_data["drawings_count"],
            })

            # Log extraction summary
            py_len = len(pymupdf_data["text"].strip())
            pl_len = len(plumber_data["text"].strip())
            pm_len = len(pdfminer_data["text"].strip())
            tbl_count = len(plumber_data["tables"])
            print(f"    PyMuPDF: {py_len} chars | pdfplumber: {pl_len} chars | pdfminer: {pm_len} chars | {tbl_count} tables", file=sys.stderr)

        return {"pages": pages}


# =============================================================================
# STREAM 2: MISTRAL OCR
# =============================================================================

class MistralOCRProcessor:
    """Extract text using Mistral OCR API + GPT-4o-mini vision fallback for images."""

    # Regex to find image placeholders in markdown
    IMG_PATTERN = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

    def __init__(self):
        if not HAS_MISTRAL:
            self.client = None
        else:
            api_key = os.environ.get("MISTRAL_OCR_API_KEY") or os.environ.get("MISTRAL_API_KEY")
            if not api_key:
                print("Warning: No MISTRAL_OCR_API_KEY or MISTRAL_API_KEY found", file=sys.stderr)
                self.client = None
            else:
                self.client = Mistral(api_key=api_key)

        self.openai_client = get_openai_client()

    def _extract_text_from_image(self, image_b64: str, image_id: str) -> str:
        """Send an image to GPT-4o-mini vision to extract text from it."""
        if not self.openai_client:
            return ""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Extract ALL text visible in this construction drawing/blueprint image. "
                                    "Include every label, dimension, room name, note, number, annotation, "
                                    "equipment ID, grid reference, and any other readable text. "
                                    "Return ONLY the extracted text, one item per line. "
                                    "Do not describe the image - just extract the text."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_b64 if image_b64.startswith("data:") else f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=2048,
                temperature=0.1,
            )
            result = response.choices[0].message.content or ""
            print(f"      GPT-4o-mini extracted {len(result)} chars from {image_id}", file=sys.stderr)
            return result
        except Exception as e:
            print(f"      GPT-4o-mini vision error for {image_id}: {e}", file=sys.stderr)
            return ""

    def _process_page_images(self, raw_markdown: str, images: List[Dict[str, Any]]) -> str:
        """Replace image placeholders with GPT-4o-mini extracted text."""
        if not images or not self.openai_client:
            # No images or no OpenAI client — just strip image refs
            cleaned = self.IMG_PATTERN.sub('', raw_markdown)
            return re.sub(r'\n{3,}', '\n\n', cleaned).strip()

        # Build a map of image_id -> base64 data
        image_map: Dict[str, str] = {}
        for img in images:
            img_id = img.get("id", "") if isinstance(img, dict) else (img.id if hasattr(img, "id") else "")
            img_b64 = img.get("image_base64", "") if isinstance(img, dict) else (img.image_base64 if hasattr(img, "image_base64") else "")
            if img_id and img_b64:
                image_map[img_id] = img_b64

        if not image_map:
            cleaned = self.IMG_PATTERN.sub('', raw_markdown)
            return re.sub(r'\n{3,}', '\n\n', cleaned).strip()

        print(f"    Found {len(image_map)} embedded images, sending to GPT-4o-mini vision...", file=sys.stderr)

        # Extract text from each image in parallel
        extracted_texts: Dict[str, str] = {}

        def process_image(item: Tuple[str, str]) -> Tuple[str, str]:
            img_id, img_b64 = item
            text = self._extract_text_from_image(img_b64, img_id)
            return img_id, text

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_image, image_map.items()))
            for img_id, text in results:
                if text.strip():
                    extracted_texts[img_id] = text

        # Replace image placeholders with extracted text
        def replace_image(match: re.Match) -> str:
            img_ref = match.group(2)  # The URL/reference part
            # Try to find matching image by checking if ref contains any image_id
            for img_id, text in extracted_texts.items():
                if img_id in img_ref or img_ref in img_id:
                    return f"\n[Extracted from image {img_id}]:\n{text}\n"
            return ""  # Remove if no match found

        result = self.IMG_PATTERN.sub(replace_image, raw_markdown)
        return re.sub(r'\n{3,}', '\n\n', result).strip()

    def process_document(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Send entire PDF to Mistral OCR API, then process any images with GPT-4o-mini."""
        if not self.client:
            return {"pages": [], "error": "Mistral OCR client not available"}

        print("  Stream 2: Sending PDF to Mistral OCR API...", file=sys.stderr)

        pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')

        for attempt in range(Config.MAX_RETRIES):
            try:
                # Request image data so we can process images with GPT-4o-mini
                ocr_response = self.client.ocr.process(
                    model=Config.MISTRAL_OCR_MODEL,
                    document={
                        "type": "document_url",
                        "document_url": f"data:application/pdf;base64,{pdf_b64}",
                    },
                    include_image_base64=True,
                )

                pages = []
                if hasattr(ocr_response, 'pages') and ocr_response.pages:
                    for page in ocr_response.pages:
                        raw_md = page.markdown if hasattr(page, 'markdown') else str(page)
                        page_num = page.index + 1 if hasattr(page, 'index') else len(pages) + 1

                        # Get images for this page (if any)
                        page_images = page.images if hasattr(page, 'images') and page.images else []

                        # Check if there are image placeholders in the markdown
                        has_images = bool(self.IMG_PATTERN.search(raw_md))

                        if has_images and page_images:
                            print(f"    Page {page_num}: has {len(page_images)} image(s), processing with GPT-4o-mini...", file=sys.stderr)
                            processed_md = self._process_page_images(raw_md, page_images)
                        else:
                            # No images, just clean the markdown
                            processed_md = self.IMG_PATTERN.sub('', raw_md)
                            processed_md = re.sub(r'\n{3,}', '\n\n', processed_md).strip()

                        pages.append({
                            "page_num": page_num,
                            "markdown_text": processed_md,
                        })
                elif hasattr(ocr_response, 'text'):
                    pages.append({
                        "page_num": 1,
                        "markdown_text": re.sub(r'\n{3,}', '\n\n', self.IMG_PATTERN.sub('', ocr_response.text)).strip(),
                    })
                else:
                    resp_dict = ocr_response.model_dump() if hasattr(ocr_response, 'model_dump') else {}
                    if 'pages' in resp_dict:
                        for p in resp_dict['pages']:
                            raw = p.get('markdown', str(p))
                            pages.append({
                                "page_num": p.get('index', len(pages)) + 1,
                                "markdown_text": re.sub(r'\n{3,}', '\n\n', self.IMG_PATTERN.sub('', raw)).strip(),
                            })
                    else:
                        pages.append({
                            "page_num": 1,
                            "markdown_text": str(ocr_response),
                        })

                print(f"  Stream 2: Received {len(pages)} page(s) from Mistral OCR", file=sys.stderr)
                for p in pages:
                    text_len = len(p.get("markdown_text", ""))
                    print(f"    Page {p['page_num']}: {text_len} chars", file=sys.stderr)

                return {"pages": pages}

            except Exception as e:
                error_str = str(e).lower()
                if ("rate" in error_str or "429" in error_str) and attempt < Config.MAX_RETRIES - 1:
                    delay = Config.INITIAL_RETRY_DELAY * (2 ** attempt)
                    print(f"  Rate limited, retrying in {delay}s...", file=sys.stderr)
                    time.sleep(delay)
                    continue

                print(f"  Mistral OCR error: {e}", file=sys.stderr)
                return {"pages": [], "error": str(e)}

        return {"pages": [], "error": "Max retries exceeded"}


# =============================================================================
# STREAM 3: GOOGLE DOCUMENT AI
# =============================================================================

class GoogleDocAIProcessor:
    """Extract text using Google Document AI OCR."""

    def __init__(self):
        self.client = None
        self.processor_name = None

        if not HAS_DOCUMENT_AI:
            return

        project_id = os.environ.get("GCP_PROJECT_ID")
        location = os.environ.get("GCP_LOCATION", "us")
        processor_id = os.environ.get("GOOGLE_DOCUMENT_AI_PROCESSOR_ID")

        if not project_id or not processor_id:
            print("Warning: GCP_PROJECT_ID or GOOGLE_DOCUMENT_AI_PROCESSOR_ID not set", file=sys.stderr)
            return

        # Set credentials path
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path and not os.path.isabs(creds_path):
            # Resolve relative to project root
            creds_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), creds_path)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

        try:
            opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
            self.client = documentai.DocumentProcessorServiceClient(client_options=opts)
            self.processor_name = self.client.processor_path(project_id, location, processor_id)
        except Exception as e:
            print(f"Warning: Failed to initialize Document AI client: {e}", file=sys.stderr)

    def process_document(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Send PDF to Google Document AI for OCR."""
        if not self.client:
            return {"pages": [], "error": "Google Document AI client not available"}

        print("  Stream 3: Sending PDF to Google Document AI...", file=sys.stderr)

        try:
            raw_document = documentai.RawDocument(content=pdf_bytes, mime_type="application/pdf")
            request = documentai.ProcessRequest(name=self.processor_name, raw_document=raw_document)
            result = self.client.process_document(request=request)
            document = result.document

            pages = []
            if document.pages:
                for page in document.pages:
                    page_num = page.page_number if page.page_number else len(pages) + 1

                    # Extract text for this page using layout text anchors
                    page_text = ""
                    if document.text and page.layout and page.layout.text_anchor:
                        for segment in page.layout.text_anchor.text_segments:
                            start = int(segment.start_index) if segment.start_index else 0
                            end = int(segment.end_index)
                            page_text += document.text[start:end]

                    # If no text from layout, try paragraphs
                    if not page_text.strip() and page.paragraphs:
                        para_texts = []
                        for para in page.paragraphs:
                            if para.layout and para.layout.text_anchor:
                                for segment in para.layout.text_anchor.text_segments:
                                    start = int(segment.start_index) if segment.start_index else 0
                                    end = int(segment.end_index)
                                    para_texts.append(document.text[start:end])
                        page_text = "\n".join(para_texts)

                    # Extract tables
                    page_tables = []
                    if page.tables:
                        for table in page.tables:
                            table_data = []
                            for row in (table.header_rows or []) + (table.body_rows or []):
                                row_data = []
                                for cell in row.cells:
                                    cell_text = ""
                                    if cell.layout and cell.layout.text_anchor:
                                        for segment in cell.layout.text_anchor.text_segments:
                                            start = int(segment.start_index) if segment.start_index else 0
                                            end = int(segment.end_index)
                                            cell_text += document.text[start:end]
                                    row_data.append(cell_text.strip())
                                table_data.append(row_data)
                            if table_data:
                                page_tables.append(table_data)

                    pages.append({
                        "page_num": page_num,
                        "text": page_text.strip(),
                        "tables": page_tables,
                    })
            elif document.text:
                pages.append({
                    "page_num": 1,
                    "text": document.text.strip(),
                    "tables": [],
                })

            print(f"  Stream 3: Received {len(pages)} page(s) from Document AI", file=sys.stderr)
            for p in pages:
                print(f"    Page {p['page_num']}: {len(p['text'])} chars, {len(p['tables'])} tables", file=sys.stderr)

            return {"pages": pages}

        except Exception as e:
            print(f"  Document AI error: {e}", file=sys.stderr)
            return {"pages": [], "error": str(e)}


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class OCRPipeline:

    def __init__(self):
        self.python_extractor = PythonTextExtractor()
        self.ocr_processor = MistralOCRProcessor()
        self.docai_processor = GoogleDocAIProcessor()

    def process(self, pdf_bytes: bytes) -> Dict[str, Any]:
        start_time = time.time()

        # Open PDFs
        pdf_fitz = fitz.open(stream=pdf_bytes, filetype="pdf")
        pdf_plumber_doc = None
        if HAS_PDFPLUMBER:
            try:
                pdf_plumber_doc = pdfplumber.open(io.BytesIO(pdf_bytes))
            except Exception as e:
                print(f"pdfplumber failed to open: {e}", file=sys.stderr)

        total_pages = len(pdf_fitz)

        # Classify document
        print(f"Classifying document ({total_pages} pages)...", file=sys.stderr)
        doc_type, page_stats = DocumentClassifier.classify_document(pdf_fitz, pdf_plumber_doc)
        print(f"  Document type: {doc_type.value}", file=sys.stderr)

        # Run all three streams in parallel
        print("Running triple extraction streams...", file=sys.stderr)

        with ThreadPoolExecutor(max_workers=3) as executor:
            stream1_future = executor.submit(
                self.python_extractor.extract_all_pages,
                pdf_fitz, pdf_plumber_doc, pdf_bytes
            )
            stream2_future = executor.submit(
                self.ocr_processor.process_document,
                pdf_bytes
            )
            stream3_future = executor.submit(
                self.docai_processor.process_document,
                pdf_bytes
            )

            python_results = stream1_future.result()
            ocr_results = stream2_future.result()
            docai_results = stream3_future.result()

        processing_time = time.time() - start_time
        print(f"Pipeline complete in {processing_time:.2f}s", file=sys.stderr)

        return {
            "document_type": doc_type.value,
            "total_pages": total_pages,
            "processing_time_seconds": round(processing_time, 2),
            "python_extraction": python_results,
            "ocr_extraction": ocr_results,
            "docai_extraction": docai_results,
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
    else:
        pdf_bytes = sys.stdin.buffer.read()

    if not pdf_bytes:
        print(json.dumps({"error": "No PDF data provided"}))
        sys.exit(1)

    pipeline = OCRPipeline()
    result = pipeline.process(pdf_bytes)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
