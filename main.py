import sys
from app_utils import censor_faces
from constants import VIDEOS_DIR, VIDEOS_EXTENSIONS, LINE_SEPARATOR, CENSORING_METHODS


def prompt_user_for_video(video_paths):
    """
    Prompts the user to select a video file from a list of video paths.

    Args:
        video_paths (list): List of paths to available video files.

    Returns:
        selected_video (Path): The path to the selected video file.
    """
    while True:
        # List the available video files
        print("\n- Please select a video file from the list below:\n")
        for idx, video in enumerate(video_paths, start=1):
            print(f"  {idx}. {video.name}")

        # Get user input
        choice = input(
            f"\n- Enter the number corresponding to your choice (1-{len(video_paths)}):  ")

        try:
            # Convert input to integer and validate
            choice = int(choice)
            if choice in range(1, len(video_paths) + 1):
                selected_video = video_paths[choice - 1]
                return selected_video
            else:
                print(
                    f"\n\n---> Invalid input. Please enter a number between 1 and {len(video_paths)}. <---\n")
        except ValueError:
            print("\n\n---> Invalid input. Please enter a number. <---\n")


def prompt_user_for_method(video_name):
    """
    Prompts the user to select a method for censoring faces in the video.

    Args:
        video_name (str): The name of the video for which the user is selecting the censoring method.

    Returns:
        str: The selected method for censoring faces ('1' for Blur, '2' for Box, '3' for Cat).
    """
    while True:
        # Prompt user for input
        print(
            f'\n- Please select a method for censoring faces in the video "{video_name}":\n')
        for idx, method in enumerate(CENSORING_METHODS, start=1):
            print(f"  {idx}. {method.title()}")

        # Get user input
        choice = input(
            f"\n- Enter the number corresponding to your choice (1-{len(CENSORING_METHODS)}):  ")

        try:
            # Convert input to integer and validate
            choice = int(choice)
            if choice in range(1, len(CENSORING_METHODS) + 1):
                selected_method = CENSORING_METHODS[choice - 1]
                return selected_method
            else:
                print(
                    f"\n\n---> Invalid input. Please enter a number between 1 and {len(CENSORING_METHODS)}. <---\n")
        except ValueError:
            print("\n\n---> Invalid input. Please enter a number. <---\n")


def main():
    """
    Main function to prompt user for video and method to censor faces, then process the selected video accordingly.

    Steps:
    1. Fetch the list of all video files in the current directory.
    2. Check if there are any video files in the directory.
    3. Prompt the user to select a video from the list.
    4. Prompt the user to select a method for censoring faces (blur, box, or cat).
    5. Process the selected video with the chosen method and generate an output AVI video.
    """

    # Get list of all video files in the current directory
    video_paths = [path for path in VIDEOS_DIR.iterdir()
                   if path.suffix.lower() in VIDEOS_EXTENSIONS]

    # Check if there are any video files in the directory
    if not video_paths:
        sys.exit("\n---> No video files found in the videos directory. <---\n")

    # Define the selected source video path
    source_video = prompt_user_for_video(video_paths)

    # Display a separator
    print(f"\n\n{LINE_SEPARATOR}\n")

    # Get user input for the method
    method = prompt_user_for_method(source_video.name)

    # Print a message indicating the start of the censoring process
    print(f'\n\n\n---> Censoring faces using "{method.title()}" method and generating output AVI video: "{
          source_video.stem}_censored_faces_{method.lower()}.avi" ...\n')

    try:
        # Apply selected method to censor faces in the video
        censor_faces(video_path=source_video, method=method)
    except Exception as error:
        print(f"\n--- An error occurred: {error} ---\n")
        raise

    # Print a message indicating successful completion of the face censoring process
    print("\n---> Face censoring completed successfully. Output AVI video created. <---\n")

    # Farewell message
    sys.exit('\n\n===== Thank you for using "CensorFace Video Tool" =====\n\n\n')


if __name__ == "__main__":
    """
    Entry point of the script. When the script is run directly, this block is executed.

    This block ensures that the `main` function is called, starting the process of 
    prompting the user for a video file and a face censoring method, and then 
    processing the selected video accordingly.
    """

    # Call the main function to start the process
    main()
