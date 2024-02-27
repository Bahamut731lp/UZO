import cv2
import numpy as np

def calculate_histogram(roi):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, (0, 60, 32), (180, 255, 255))
    hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist

def apply_camshift(frame, track_window, hist):
    x, y, w, h = track_window
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
    
    # Apply meanshift to get the new location
    ret, track_window = cv2.CamShift(dst, track_window, termination_criteria)
    
    # Draw the tracked region on the frame
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
    
    return track_window

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
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply CamShift to track the object
    track_window = apply_camshift(frame, track_window, hist)
    
    # Show the frame
    cv2.imshow('Frame', frame)
    
    # Exit if ESC pressed
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
