import cv2
import numpy as np

# Load the video
video_path = "./data/cv02_hrnecek.mp4"
video = cv2.VideoCapture(video_path)

# Load the crop of the tracked object
crop_path = "./data/cv02_vzor_hrnecek.bmp"
crop = cv2.imread(crop_path, cv2.IMREAD_COLOR)

# Convert the crop to HSV color space
crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

# Calculate the histogram of the crop
crop_hist = cv2.calcHist([crop_hsv], [0], None, [180], [0, 180])

# Normalize the histogram
crop_hist = cv2.normalize(crop_hist, crop_hist, 0, 1, cv2.NORM_MINMAX)

# Set up the termination criteria for the algorithm
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# Initialize the window for tracking
x, y, w, h = 0, 0, crop.shape[1], crop.shape[0]

while True:
    # Read the next frame
    lze_cist_dal, frame = video.read()
    if not lze_cist_dal:
        break

    # Convert the frame to HSV color space
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Calculate the back projection of the frame
    dst = cv2.calcBackProject([frame_hsv], [0], crop_hist, [0, 180], 1)

    # Zeroth moment
    mass = np.sum(dst)

    # ROI size
    width = round(0.7 * np.sqrt(mass))
    height = round(0.9 * np.sqrt(mass))

    # Calculate the first moment along X-axis
    x_coords = np.arange(dst.shape[1])  # X coordinates
    x = round(np.sum(x_coords * np.sum(dst, axis=0)) / mass)

    # Calculate the first moment along Y-axis
    y_coords = np.arange(dst.shape[0])  # Y coordinates
    y = round(np.sum(y_coords * np.sum(dst, axis=1)) / mass)

    # Convert to integers
    x = int(x)
    y = int(y)
    width = int(width)
    height = int(height)

    # Apply CamShift to get the new window location
    # ret, track_window = cv2.CamShift(dst, (x, y, w, h), criteria)

    # Draw the new window on the frame
    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Check for key press
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # If ESC is pressed, exit
        break

# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()