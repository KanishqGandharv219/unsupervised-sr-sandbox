
from PIL import Image
import os

texture_dir = r"docs/assets/textures"
files_to_convert = ["normal.tif", "specular.tif"]

for filename in files_to_convert:
    path = os.path.join(texture_dir, filename)
    if os.path.exists(path):
        try:
            print(f"Converting {filename}...")
            img = Image.open(path)
            # Convert to RGB (to avoid issues with CMYK or alpha channels mostly)
            img = img.convert('RGB')
            new_path = path.replace(".tif", ".jpg")
            img.save(new_path, "JPEG", quality=90)
            print(f"Saved: {new_path}")
        except Exception as e:
            print(f"Error converting {filename}: {e}")
    else:
        print(f"File not found: {path}")
