"""
Harris Corner Detection From Scratch"""



import numpy as np
import cv2

# Funcition to define Sobel Derivatives
def my_sobel(gray, kernel, w, h):

    res = np.zeros(gray.shape)
    for rows in range(kernel.shape[1]//2, w-kernel.shape[1]//2):
        for cols in range(kernel.shape[0]//2, h-kernel.shape[0]//2):
            for m in range(3):
                for n in range(3):
                    res[rows, cols] += kernel[m, n] * gray[rows - m - 1, cols - n - 1]
    return res

# Harris Corner Detection Code
def my_harris(img, limit=10**5):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    k = 0.04
    R = np.zeros(gray.shape)
    
    # Here we call sobel function to calculate sobel derivative
    I_x = my_sobel(gray, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), h, w)
    I_y = my_sobel(gray, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]), h, w)

    # Apply Gaussian blur of size (3x3) and Sigma = 1.5
    I_xx = cv2.GaussianBlur(np.multiply(I_x, I_x), (3, 3), 1.5)
    I_yy = cv2.GaussianBlur(np.multiply(I_y, I_y), (3, 3), 1.5)
    I_xy = cv2.GaussianBlur(np.multiply(I_x, I_y), (3, 3), 1.5)
    I_yx = cv2.GaussianBlur(np.multiply(I_y, I_x), (3, 3), 1.5)

    # Construct Matrix M and find R (using lambda1 and lambda2)
    for n in range(h):
        for m in range(w):

            M = np.array([[I_xx[n, m], I_xy[n, m]],
                          [I_yx[n, m], I_yy[n, m]]])

            R[n, m] = np.linalg.det(M) - k * np.square(np.trace(M))

    corner_points = []
    ksize = 7
    for row in range(h):
        for col in range(w):
            if R[row, col] > limit:
                # check in ksize
                # Here we use a clustring method to make sure that not many corners are formed in a very small space
                clustering = False
                for n in range(ksize):
                    for m in range(ksize):
                        if row + n - 2 < h and col + m - 2 < w and R[row + n - 2, col + m - 2] > R[row, col]:
                            clustering = True
                            break
                if not clustering:
                    cv2.circle(img, (col, row), 1, (0, 0, 255), 2)
                    corner_points.append([row, col])

    return img


# Enter Absolute Path of Image
path = "C:\\Users\\susha\\Desktop\\check1.PNG"
image = cv2.imread(path)
cv2.imshow('output', my_harris(image, limit=10**8))
cv2.waitKey(0)
cv2.destroyAllWindows()
