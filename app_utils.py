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
    pass


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
    pass