import time
from google import genai
from google.genai import types as genai_types
from PIL import Image
from io import BytesIO

_client = genai.Client()


def edit_image(
    prompt: str,
    image: Image.Image,
    aspect_ratio: str = "16:9",
    resolution: str = "2K",
    model: str = "gemini-3.1-flash-image-preview",
) -> Image.Image | None:
    response = _client.models.generate_content(
        model=model,
        contents=[prompt, image],
        config=genai_types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=genai_types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=resolution,
            ),
        ),
    )

    if not response.parts:
        return None

    for part in response.parts:
        if part.inline_data is not None and part.inline_data.data is not None:
            return Image.open(BytesIO(part.inline_data.data))

    return None


def generate_video(
    prompt: str,
    image_path: str,
    save_path: str,
    aspect_ratio: str = "16:9",
    resolution: str = "720p",
    duration: int = 6,
    model: str = "veo-3.1-generate-preview",
) -> None:
    operation = _client.models.generate_videos(
        model=model,
        source=genai_types.GenerateVideosSource(
            prompt=prompt,
            image=genai_types.Image.from_file(location=image_path),
        ),
        config=genai_types.GenerateVideosConfig(
            number_of_videos=1,
            duration_seconds=duration,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
        ),
    )

    while not operation.done:
        time.sleep(10)
        operation = _client.operations.get(operation)

    if operation.response and operation.response.generated_videos:
        video = operation.response.generated_videos[0]
        if video.video:
            _client.files.download(file=video.video)
            video.video.save(save_path)
