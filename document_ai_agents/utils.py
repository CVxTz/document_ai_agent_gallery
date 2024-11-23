import base64
import io
from functools import lru_cache

import PIL.PpmImagePlugin


def ppm_to_base64_jpeg(ppm_image: PIL.PpmImagePlugin.PpmImageFile):
    try:
        # Convert to RGB mode if necessary
        rgb_image = ppm_image.convert("RGB")

        # In-memory buffer for the JPEG image
        buffered = io.BytesIO()

        # Save as JPEG
        rgb_image.save(buffered, format="JPEG")

        # Encode as base64
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return img_str

    except Exception as e:
        print(f"Error converting PPM to base64 JPEG: {e}")
        return None


def replace_value_in_dict(item, original_schema):
    # Source: https://github.com/pydantic/pydantic/issues/889
    if isinstance(item, list):
        return [replace_value_in_dict(i, original_schema) for i in item]
    elif isinstance(item, dict):
        if list(item.keys()) == ["$ref"]:
            definitions = item["$ref"][2:].split("/")
            res = original_schema.copy()
            for definition in definitions:
                res = res[definition]
            return res
        else:
            return {
                key: replace_value_in_dict(i, original_schema)
                for key, i in item.items()
            }
    else:
        return item

def delete_keys_recursive(d, key_to_delete):
    if isinstance(d, dict):
        # Delete the key if it exists
        if key_to_delete in d:
            del d[key_to_delete]
        # Recursively process all items in the dictionary
        for k, v in d.items():
            delete_keys_recursive(v, key_to_delete)
    elif isinstance(d, list):
        # Recursively process all items in the list
        for item in d:
            delete_keys_recursive(item, key_to_delete)