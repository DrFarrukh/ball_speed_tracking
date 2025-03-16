# Ball Speed Tracking

This repository contains a Python script for tracking the speed of a cricket ball from video footage. The script can handle `.dav` video files by either opening them directly or converting them to `.mp4` format using FFmpeg.

## Features

- Convert `.dav` files to `.mp4` using FFmpeg
- Interactive selection of calibration points
- Track the ball across multiple frames
- Calculate the speed of the ball in meters per second (m/s) and kilometers per hour (km/h)
- Plot the ball trajectory and speed

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- SciPy
- FFmpeg (for `.dav` file conversion)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ball_speed_tracking.git
    cd ball_speed_tracking
    ```

2. Install the required Python packages:
    ```bash
    pip install opencv-python numpy matplotlib scipy
    ```

3. Install FFmpeg:
    - Follow the instructions on the [FFmpeg website](https://ffmpeg.org/download.html) to install FFmpeg on your system.

## Usage

1. Run the script:
    ```bash
    python ball_speed_tracking.py
    ```

2. Follow the on-screen instructions to provide the path to your `.dav` video file and a known distance in the video (e.g., the length of a cricket pitch, 20.12 meters).

3. Use the interactive interface to select calibration points and track the ball across frames.

4. The script will calculate and display the average and maximum speed of the ball, and save the results as `ball_speed_analysis.png`.

## Example

```bash
Enter the path to your .dav video file: /path/to/your/video.dav
Enter a known distance in meters (e.g., 20.12 for cricket pitch): 20.12
Enter starting frame number (default: 0): 0
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [SciPy](https://www.scipy.org/)
- [FFmpeg](https://ffmpeg.org/)
