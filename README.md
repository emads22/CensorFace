# CensorFace Video Tool

![CensorFace_logo](./assets/images/CensorFace_logo.png)

## Overview
CensorFace Video Tool is a Python script designed to enhance privacy in videos by censoring faces. It offers various methods such as `blurring`, `boxing`, or even replacing faces with `cat face` images. This tool is particularly useful for scenarios where individuals in videos need to be anonymized for privacy reasons, such as public footage or sensitive content.

The tool provides an interactive user interface, guiding users through the process of selecting a video file and a preferred method for censoring faces. It supports a wide range of video formats and allows users to customize the censoring method based on their specific requirements.

## Features
- **Face Censoring**: Censor faces in a video using methods like blurring, boxing, or replacing them with cat faces.
- **Interactive User Interface**: The script prompts the user to select a video file and a method for censoring faces.
- **Output Generation**: Generates a new video file with censored faces based on the chosen method.

## Technologies Used
- **opencv-python**: A library for computer vision and image processing.

## Algorithms Used
The script utilizes the following main OpenCV algorithms:
- **Haar Cascade Classifier**: Used for face detection in images.
- **Gaussian Blur**: Applied to detected faces for blurring.
- **Rectangle Drawing**: Draws rectangles around detected faces for boxing.
- **Image Resizing and Replacement**: Replaces detected faces with cat images.

## Setup
1. Clone the repository.
2. Ensure Python 3.x is installed.
3. Install the required dependencies using `pip install -r requirements.txt`.
4. Configure the necessary parameters such as video paths, censoring methods, and additional video extensions in `constants.py`.
   - Users can add more video extensions to the `VIDEOS_EXTENSIONS` list to support additional video formats.
   - Place as many video files as needed in the `assets/videos` directory or change the directory path as per choice.
   - Sample videos for testing purposes can be obtained for free from [Freep!k](https://www.freepik.com/).
5. Run the script using `python main.py`.

## Usage
1. Run the script using `python main.py`.
2. Follow the prompts to select a video file and a method for censoring faces.
3. The script will process the video. Please note that depending on the video format and size, the censoring process may take several minutes. **`Kindly be patient during processing`**.
4. Upon completion, the output AVI video with censored faces will be saved in the `assets/output` directory.

 - **Note**: Some sample MP4 video files are included in `assets\videos` for testing and demonstration.

### Sample Videos
For testing purposes, sample videos have been provided along with their corresponding outputs after testing. You can make use of these sample videos to familiarize yourself with the tool and its capabilities.

## Contributing
Contributions are welcome! Here are some ways you can contribute to the project:
- Report bugs and issues
- Suggest new features or improvements
- Submit pull requests with bug fixes or enhancements

## Author
- Emad &nbsp; E>
  
  [<img src="https://img.shields.io/badge/GitHub-Profile-blue?logo=github" width="150">](https://github.com/emads22)

## License
This project is licensed under the MIT License, which grants permission for free use, modification, distribution, and sublicense of the code, provided that the copyright notice (attributed to [emads22](https://github.com/emads22)) and permission notice are included in all copies or substantial portions of the software. This license is permissive and allows users to utilize the code for both commercial and non-commercial purposes.

Please see the [LICENSE](LICENSE) file for more details.