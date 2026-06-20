import sys
from PIL import Image
from lib.gemini import edit_image

if __name__ == "__main__":
    img_path, save_path, prompt = sys.argv[1], sys.argv[2], sys.argv[3]

    output = edit_image(
        prompt=prompt,
        image=Image.open(img_path),
        resolution="2K",
        model="gemini-3-pro-image-preview",
    )

    if not output:
        print("error generating image")
    else:
        output.save(save_path)
