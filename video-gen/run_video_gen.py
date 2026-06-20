from lib.gemini import generate_video

if __name__ == "__main__":
    seed_path = "data/sample_0/seed_1.jpg"
    save_path = "data/sample_0/video_10.mp4"

    generate_video(
        prompt="the person walks forward and steps up on to the blocks.",
        image_path=seed_path,
        save_path=save_path,
    )
