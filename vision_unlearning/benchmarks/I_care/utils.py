


def _encode_image_file(img_path: str, max_dim: int = 1024) -> str:
    '''
    Downsample / reduce resolution to limit size before encoding
    '''
    assert os.path.exists(img_path), f"Image file not found at {img_path}"
    with Image.open(img_path) as im:
        # Convert to RGB to ensure compatibility with JPEG
        if im.mode != 'RGB':
            im = im.convert('RGB')
        if max(im.size) > max_dim:

            scale = max_dim / max(im.size)
            new_size = (int(im.size[0] * scale), int(im.size[1] * scale))
            im = im.resize(new_size, Image.LANCZOS)
            #print(f"Resized image from {im.size} to {new_size} to limit size before encoding.")
        buf = io.BytesIO()
        im.save(buf, format='PNG', quality=85, optimize=True)
        image_bytes = buf.getvalue()
    return base64.b64encode(image_bytes).decode('ascii')

def _decode_image(image_data: str) -> io.BytesIO:
    assert isinstance(image_data, str), f"Expected image data to be a base64 string, but got {type(image_data)}"
    return io.BytesIO(base64.b64decode(image_data))

####

import json
import numpy as np
import pandas as pd
import shap

def explanation_to_dict(expl):
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

def dict_to_explanation(d):
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