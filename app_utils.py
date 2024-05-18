import cv2
from pprint import pprint
from constants import *


def get_video_info(video):
    """
    Extracts information about the video from the provided VideoCapture object.

    Args:
        video: A VideoCapture object representing the input video.

    Returns:
        dict: A dictionary containing information about the video (all info data are of type Float).
    """
    # Get the video info in a dict
    video_info = {
        'frame_width': video.get(cv2.CAP_PROP_FRAME_WIDTH),
        'frame_height': video.get(cv2.CAP_PROP_FRAME_HEIGHT),
        'total_frames': video.get(cv2.CAP_PROP_FRAME_COUNT),
        'fps': video.get(cv2.CAP_PROP_FPS),
        'codec': video.get(cv2.CAP_PROP_FOURCC),
        'format': video.get(cv2.CAP_PROP_FORMAT),
        'brightness': video.get(cv2.CAP_PROP_BRIGHTNESS),
        'contrast': video.get(cv2.CAP_PROP_CONTRAST),
        'saturation': video.get(cv2.CAP_PROP_SATURATION),
        'hue': video.get(cv2.CAP_PROP_HUE),
        'gain': video.get(cv2.CAP_PROP_GAIN),
        'exposure': video.get(cv2.CAP_PROP_EXPOSURE),
        'is_rgb': video.get(cv2.CAP_PROP_CONVERT_RGB),
        'rectification_flag': video.get(cv2.CAP_PROP_RECTIFICATION)
    }

    # Return the dict
    return video_info


def blur_faces_in_frame(image):
    """
    Apply heavy Gaussian blur to detected faces in the input image.

    Args:
        image (numpy.ndarray): Input image (BGR format) containing faces to be blurred.

    Returns:
        numpy.ndarray: Image with detected faces blurred.

    Notes:
        - This function uses a pre-trained cascade classifier to detect faces in the input image.
        - Detected faces are blurred using a Gaussian blur filter with a kernel size of (45, 45).
        - The standard deviation of the Gaussian kernel is automatically calculated based on the kernel size.
        - Detected faces are replaced with heavily blurred regions in the input image.

    """
    # Load the pre-trained cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale for convenience and noice reduction and faster processing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image using the detectMultiScale function of the face cascade classifier
    # The parameters 1.1 and 4 control the scale factor and minimum number of neighbors respectively
    faces = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=4)

    # 'faces' is a numpy array holds information about the detected faces within the image.
    # Each element of 'faces' represents a rectangle (bounding box) that encloses a detected face.
    # The elements consist of coordinates (x, y, width, height) specifying the position and size of each detected face within the image.
    # These coordinates define rectangular regions within the image where faces are detected.
    # Uncomment the following two lines to display them for testing
    # print(type(faces), end=": ")
    # pprint(faces)

    # Iterate over each detected face (rectangle) in the 'faces' array (in the image)
    for (x, y, width, height) in faces:
        # Calculate the coordinates of the corners of the detected face rectangle
        start_y = y  # Starting y-coordinate of the detected face rectangle
        end_y = y + height  # Ending y-coordinate of the detected face rectangle
        start_x = x  # Starting x-coordinate of the detected face rectangle
        end_x = x + width  # Ending x-coordinate of the detected face rectangle

        # Extract the region of interest (ROI) within the rectangle which is the detected face region from the original image
        # In OpenCV, image slicing is performed as image[y:y+h, x:x+w].
        # Therefore, we start with the y-coordinate (vertical) followed by the x-coordinate (horizontal) to extract the desired region.
        # 'start_y' and 'end_y' represent the vertical range, and 'start_x' and 'end_x' represent the horizontal range.
        roi = image[start_y:end_y, start_x:end_x]

        # Apply heavy Gaussian blur to the ROI to obscure the face with a kernel size of (45, 45)
        # (45, 45): Size of the Gaussian kernel for blurring. Larger values result in heavier blurring.
        # 0: Standard deviation of the Gaussian kernel along the X and Y directions. A value of 0 indicates automatic calculation based on the kernel size. It controls the amount of blurring.
        blurred_roi = cv2.GaussianBlur(roi, (45, 45), 0)

        # Replace the original detected face region in the image with the heavily blurred face region
        image[start_y:end_y, start_x:end_x] = blurred_roi

    # Return the modified image object
    return image


def box_faces_in_frame(image):
    pass


def cat_faces_in_frame():
    # Load the pre-trained cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale for convenience and noice reduction and faster processing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image using the detectMultiScale function of the face cascade classifier
    # The parameters 1.1 and 4 control the scale factor and minimum number of neighbors respectively
    faces = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=4)

    # 'faces' is a numpy array holds information about the detected faces within the image.
    # Each element of 'faces' represents a rectangle (bounding box) that encloses a detected face.
    # The elements consist of coordinates (x, y, width, height) specifying the position and size of each detected face within the image.
    # These coordinates define rectangular regions within the image where faces are detected.
    # Uncomment the following two lines to display them for testing
    # print(type(faces), end=": ")
    # pprint(faces)

    # Iterate over each detected face (rectangle) in the 'faces' array (in the image)
    for (x, y, width, height) in faces:
        # Calculate the coordinates of the corners of the detected face rectangle
        start_y = y  # Starting y-coordinate of the detected face rectangle
        end_y = y + height  # Ending y-coordinate of the detected face rectangle
        start_x = x  # Starting x-coordinate of the detected face rectangle
        end_x = x + width  # Ending x-coordinate of the detected face rectangle

        # Extract the region of interest (ROI) within the rectangle which is the detected face region from the original image
        # In OpenCV, image slicing is performed as image[y:y+h, x:x+w].
        # Therefore, we start with the y-coordinate (vertical) followed by the x-coordinate (horizontal) to extract the desired region.
        # 'start_y' and 'end_y' represent the vertical range, and 'start_x' and 'end_x' represent the horizontal range.
        roi = image[start_y:end_y, start_x:end_x]

        # Apply heavy Gaussian blur to the ROI to obscure the face with a kernel size of (45, 45)
        # (45, 45): Size of the Gaussian kernel for blurring. Larger values result in heavier blurring.
        # 0: Standard deviation of the Gaussian kernel along the X and Y directions. A value of 0 indicates automatic calculation based on the kernel size. It controls the amount of blurring.
        blurred_roi = cv2.GaussianBlur(roi, (45, 45), 0)

        # Replace the original detected face region in the image with the heavily blurred face region
        image[start_y:end_y, start_x:end_x] = blurred_roi

    # Return the modified image object
    return image


def censor_faces(video_path, method="box"):
    """
    Anonymize faces in a video using specified method.

    Args:
        video_path (str): Path to the input video file.
        method (str, optional): Method for anonymizing faces. Options are 'blur' (default), 'box', or 'cat'.

    Raises:
        Exception: If the video file could not be opened or if an error occurs during video processing.

    Returns:
        None
    """
    try:
        # Create a VideoCapture object to read the video file
        video = cv2.VideoCapture(str(video_path))

        if not video.isOpened():
            # Exit program if the video file could not be opened
            raise Exception("\n--- Error: Could not open video. ---\n")

        # Prepare to write the modified video
        v_info = get_video_info(video)

        # Define the output video path (Create the output directory if it doesn't exist)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_video_path = OUTPUT_DIR / \
            f"{video_path.stem}_masked_faces.avi"

        # codec = int(v_info['codec'])  # Define the codec for the output video
        # Using same codec and format (type extension) as source video would give warnings and sometimes errors depending on source video format and codec (like FFMPEG related errors and warnings) so best solution is to use ".avi" type extension and codec for the output video (DIVX is a common choice for AVI format)
        # cv2.VideoWriter_fourcc() expects the format "cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')" to specify the codec so we use `*` to unpack the string "DIVX" (iterable)
        codec = cv2.VideoWriter_fourcc(*"DIVX")  # type: 'int'
        fps = int(v_info['fps'])
        width = int(v_info['frame_width'])
        height = int(v_info['frame_height'])

        # Specify the output video file path, codec, frames per second, and frame size
        # Arguments:
        # - output_video_path: Path to the output video file
        # - fourcc: Four-character code representing the codec (e.g., 'DIVX' for DivX codec)
        # - fps: Frames per second (FPS) of the output video
        # - (width, height): Tuple representing the frame size (width, height) of the output video
        output_video = cv2.VideoWriter(filename=str(
            output_video_path), fourcc=codec, fps=fps, frameSize=(width, height))

        while True:
            # Read a frame from the video
            success, frame = video.read()

            # Check if frame was successfully read
            if not success:
                break  # Break the loop if no frame is read or end of video

            # Match method ('blur', 'box', or 'cat'), this switch statement matches the method specified for face anonymization.
            match method.lower():
                case 'blur':
                    # Blur the faces in the current frame
                    new_frame = blur_faces_in_frame(frame)

                case 'box':
                    # Replace the faces with gray boxes in the current frame
                    new_frame = box_faces_in_frame(frame)

                case 'cat':
                    # Replace the faces with cat pic in the current frame
                    new_frame = cat_faces_in_frame(frame)

            # Write the modified frame to the output video
            output_video.write(new_frame)

        # Release resources
        video.release()
        output_video.release()
    except Exception as e:
        print(f"\n--- An error occurred during video processing: {e} ---\n")
        raise
