import cv2
import numpy as np

def calculate_histogram(roi):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, (0, 60, 32), (180, 255, 255))
    hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

# Read the video
cap = cv2.VideoCapture('./data/cv02_hrnecek.mp4')

# Read the cropped region of interest (ROI)
roi = cv2.imread('./data/cv02_vzor_hrnecek.bmp')

# Set initial window to full ROI
x, y, w, h = 0, 0, roi.shape[1], roi.shape[0]
track_window = (x, y, w, h)

# Calculate histogram of the ROI
hist = calculate_histogram(roi)

# Define termination criteria
termination_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV color space
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 0]

    # Calculate the back projection of the frame
    dst = cv2.calcBackProject([frame_hsv], [0], hist, [0, 180], 1)

    # Zeroth moment
    mass = np.sum(dst)

    # ROI size
    width = round(0.7 * np.sqrt(mass))
    height = round(0.9 * np.sqrt(mass))

    #TODO: TOhle je totálně fucked, to bude zejtra dělat Pavel.
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

    print(x, y, width, height, mass)

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
# Exit if ESC pressed

input()
# Release resources
cap.release()
cv2.destroyAllWindows()