from app_utils import censor_faces
from constants import VIDEOS_DIR


def main():
    # Define the source video path
    source_video = VIDEOS_DIR / "video.mp4"

    # Process the video to detect faces and save as AVI
    print("\n- Processing faces into AVI video ...\n")
    censor_faces(source_video)
    print("\n-- AVI video faces processing complete.\n")


if __name__ == "__main__":
    try:
        main()
        print(f'\n--- Output video with blurred faces generated successfully from the input video. ---\n')
    except Exception as err:
        raise err
