import os
from PIL import Image, ImageChops, ImageEnhance

def convert_to_ela_image(path, quality=90):
    temp_path = "temp_ela.jpg"

    with Image.open(path).convert("RGB") as original:
        original.save(temp_path, "JPEG", quality=quality)

        with Image.open(temp_path) as compressed:
            ela_image = ImageChops.difference(original, compressed)

            extrema = ela_image.getextrema()
            max_diff = max([ex[1] for ex in extrema])

            scale = 255.0 / max_diff if max_diff != 0 else 1
            ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return ela_image
