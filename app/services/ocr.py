"""
OCR Service for text extraction and Indian license plate pattern matching in video frames.
"""

import re
import threading
from typing import Dict, Any, List

from loguru import logger


class OCRService:
    """
    Service utilizing EasyOCR to perform text extraction
    and identify license plates.
    """

    _reader: Any = None
    _lock = threading.Lock()

    @classmethod
    def get_reader(cls) -> Any:
        """
        Initializes and returns a singleton EasyOCR Reader.

        Thread-safe:
        Multiple concurrent OCR calls will not create
        multiple EasyOCR instances.
        """

        if cls._reader is not None:
            return cls._reader

        with cls._lock:

            # Double-check after acquiring lock
            if cls._reader is not None:
                return cls._reader

            try:
                import easyocr
                import torch

                gpu_available = torch.cuda.is_available()

                if gpu_available:
                    try:
                        gpu_name = torch.cuda.get_device_name(0)
                        logger.info(
                            f"Initializing EasyOCR Reader (GPU mode) "
                            f"using: {gpu_name}"
                        )
                    except Exception:
                        logger.info(
                            "Initializing EasyOCR Reader (GPU mode)"
                        )
                else:
                    logger.info(
                        "Initializing EasyOCR Reader (CPU mode)"
                    )

                cls._reader = easyocr.Reader(
                    ["en"],
                    gpu=gpu_available
                )

                logger.info(
                    f"EasyOCR Reader initialized successfully "
                    f"({'GPU' if gpu_available else 'CPU'} mode)"
                )

            except Exception:
                logger.exception(
                    "Failed to initialize EasyOCR Reader"
                )
                cls._reader = None

        return cls._reader

    @classmethod
    def extract_text(cls, image_path: Any) -> Dict[str, List[str]]:
        """
        Extract visible text and detect Indian license plates.

        Args:
            image_path: Path to image file

        Returns:
            {
                "detected_text": [...],
                "license_plates": [...]
            }
        """

        result = {
            "detected_text": [],
            "license_plates": [],
        }

        try:
            reader = cls.get_reader()

            if reader is None:
                logger.warning(
                    "OCR Reader unavailable. Returning empty OCR result."
                )
                return result

            ocr_results = reader.readtext(str(image_path))

            detected_texts = []

            for _bbox, text, _confidence in ocr_results:

                if not text:
                    continue

                cleaned_text = " ".join(text.split()).strip()

                if cleaned_text:
                    detected_texts.append(cleaned_text)

            # Preserve order while removing duplicates
            unique_texts = []
            seen = set()

            for text in detected_texts:
                if text not in seen:
                    seen.add(text)
                    unique_texts.append(text)

            result["detected_text"] = unique_texts

            # Indian vehicle registration pattern
            plate_regex = re.compile(
                r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$"
            )

            license_plates = []

            for text in unique_texts:

                normalized_text = re.sub(
                    r"[\s\-]",
                    "",
                    text
                ).upper()

                if plate_regex.match(normalized_text):
                    license_plates.append(normalized_text)

            result["license_plates"] = list(
                dict.fromkeys(license_plates)
            )

        except Exception:
            logger.exception(
                f"OCR extraction failed for image: {image_path}"
            )

        return result