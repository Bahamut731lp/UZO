import cv2
import numpy as np
import matplotlib.pyplot as plt

def zero_frequency_shift(fft2):
    # Shift the zero-frequency component to the center of the spectrum
    rows, cols = fft2.shape
    half_rows, half_cols = rows//2, cols//2
    
    # Rearrange the quadrants of fft2
    fft2_shift = np.zeros_like(fft2)
    fft2_shift[:half_rows, :half_cols] = fft2[half_rows:, half_cols:]
    fft2_shift[:half_rows, half_cols:] = fft2[half_rows:, :half_cols]
    fft2_shift[half_rows:, :half_cols] = fft2[:half_rows, half_cols:]
    fft2_shift[half_rows:, half_cols:] = fft2[:half_rows, :half_cols]
    
    return fft2_shift

def D2_DFT(image):
    # 2D Discrete Fourier Transform
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fft2= np.fft.fft2(gray)
    # shift zero-frequency component to the center of the spectrum
    fft2_shift=zero_frequency_shift(fft2)
    return fft2_shift, fft2

cv4_robot = cv2.imread('./data/cv04c_robotC.bmp', cv2.IMREAD_GRAYSCALE)
fft2_robot_shift  = np.fft.fftshift(np.fft.fft2(cv4_robot))

# read filters and cast them to  1, 0 values
cv04_filtHP = cv2.cvtColor(cv2.imread('./data/cv04c_filtHP.bmp'), cv2.COLOR_BGR2GRAY) / 255
cv04_filtHP1 = cv2.cvtColor(cv2.imread('./data/cv04c_filtHP1.bmp'), cv2.COLOR_BGR2GRAY) / 255
cv04_filtDP = cv2.cvtColor(cv2.imread('./data/cv04c_filtDP.bmp'), cv2.COLOR_BGR2GRAY) / 255
cv04_filtDP1 = cv2.cvtColor(cv2.imread('./data/cv04c_filtDP1.bmp'), cv2.COLOR_BGR2GRAY) / 255


plt.figure(figsize=(5, 5))
# 4.1) High pass filter
fft2_robot_hp = fft2_robot_shift * cv04_filtHP
plt.subplot(2, 2, 1)
plt.imshow(np.log(np.abs(fft2_robot_hp)))

# 4.2) High pass filter 1
fft2_robot_hp1 = fft2_robot_shift * cv04_filtHP1
plt.subplot(2, 2, 2)
plt.imshow(np.log(np.abs(fft2_robot_hp1)))

# 4.3) Low pass filter
fft2_robot_dp = fft2_robot_shift * cv04_filtDP
plt.subplot(2, 2, 3)
plt.imshow(np.log(np.abs(fft2_robot_dp)))

# 4.4) Low pass filter 1
fft2_robot_dp1 = fft2_robot_shift * cv04_filtDP1
plt.subplot(2, 2, 4)
plt.imshow(np.log(np.abs(fft2_robot_dp1)))

plt.figure(figsize=(5, 5))
plt.subplot(2, 2, 1)
plt.imshow(np.abs(np.fft.ifft2(fft2_robot_hp)), cmap='gray')

plt.subplot(2, 2, 2)
plt.imshow(np.abs(np.fft.ifft2(fft2_robot_hp1)), cmap='gray')

plt.subplot(2, 2, 3)
plt.imshow(np.abs(np.fft.ifft2(fft2_robot_dp)), cmap='gray')

plt.subplot(2, 2, 4)
plt.imshow(np.abs(np.fft.ifft2(fft2_robot_dp1)), cmap='gray')

plt.show()