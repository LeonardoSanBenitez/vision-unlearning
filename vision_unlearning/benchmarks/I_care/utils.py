"""Utility helpers for the I-CARE benchmark.

- Image encoding/decoding (base64 PNG)
- SHAP Explanation serialization (requires shap package — optional dep)
- Error classes used across result templates

Note: shap is imported lazily inside functions to avoid forcing all users
to install it. It is only needed for the SHAP-related result templates.
"""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


def _encode_image_file(img_path: str, max_dim: int = 1024) -> str:
    '''
    Downsample / reduce resolution to limit size before encoding
    '''
    assert os.path.exists(img_path), f"Image file not found at {img_path}"
    img: Image.Image = Image.open(img_path)
    with img:
        # Convert to RGB to ensure compatibility with JPEG
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if max(img.size) > max_dim:
            scale = max_dim / max(img.size)
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            #print(f"Resized image from {img.size} to {new_size} to limit size before encoding.")
        buf = io.BytesIO()
        img.save(buf, format='PNG', quality=85, optimize=True)
        image_bytes = buf.getvalue()
    return base64.b64encode(image_bytes).decode('ascii')


def _decode_image(image_data: str) -> io.BytesIO:
    assert isinstance(image_data, str), f"Expected image data to be a base64 string, but got {type(image_data)}"
    return io.BytesIO(base64.b64decode(image_data))


def explanation_to_dict(expl: Any) -> Dict[str, Any]:
    """Serialize a shap.Explanation to a plain dict (JSON-serializable)."""
    return {
        "values": expl.values.tolist() if expl.values is not None else None,
        "base_values": (
            expl.base_values.tolist()
            if isinstance(expl.base_values, np.ndarray)
            else expl.base_values
        ),
        "data": expl.data.tolist() if expl.data is not None else None,
        "feature_names": list(expl.feature_names) if expl.feature_names is not None else None,
        "output_names": list(expl.output_names) if expl.output_names is not None else None,
    }


def dict_to_explanation(d: Dict[str, Any]) -> Any:
    """Deserialize a plain dict back to a shap.Explanation.

    Requires 'shap' package (optional dependency — install with pip install shap
    or pip install vision-unlearning[testbed]).
    """
    import shap  # lazy import — only needed for SHAP result templates

    return shap.Explanation(
        values=np.array(d["values"]) if d["values"] is not None else None,
        base_values=np.array(d["base_values"]) if d["base_values"] is not None else None,
        data=np.array(d["data"]) if d["data"] is not None else None,
        feature_names=d["feature_names"],
        output_names=d["output_names"],
    )


class InvalidAttributeTypeError(ValueError):
    pass


class InsufficientSamplesError(ValueError):
    pass
