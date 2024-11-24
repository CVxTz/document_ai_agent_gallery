import base64
import io
import math

import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
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


def draw_bounding_box_on_image(
    image,
    ymin,
    xmin,
    ymax,
    xmax,
    color="red",
    thickness=4,
    display_str_list=(),
    use_normalized_coordinates=True,
):
    """
    Source: https://github.com/datitran/object_detector_app/blob/master/object_detection/utils/visualization_utils.py#L122
    Adds a bounding box to an image.

    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.

    Args:
      image: a PIL.Image object.
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box
                        (each to be shown on its own line).
      use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (
            xmin * im_width,
            xmax * im_width,
            ymin * im_height,
            ymax * im_height,
        )
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=thickness,
        fill=color,
    )
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    text_bottom = top
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        _, _, text_width, text_height = font.getbbox(display_str)

        margin = math.ceil(0.05 * text_height)
        draw.rectangle(
            [
                (left, text_bottom - text_height - 2 * margin),
                (left + text_width, text_bottom),
            ],
            fill=color,
        )
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill="black",
            font=font,
        )
        text_bottom -= text_height - 2 * margin
