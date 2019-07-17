import cv2
import numpy as np


def make_coordinates(image, line_parameters):
    # print(line_parameters)
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/ slope)
    x2 = int((y2-intercept)/ slope)
    return np.array([x1,y1,x2,y2])


def average_slope_intercept(img, lines):
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # y = Mx + b
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0] #M
            intercept = parameters[1] #b
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        if len(left_fit) > 0:
            left_fit_average = np.average(left_fit, axis=0)
            left_line = make_coordinates(img, left_fit_average)

        if len(right_fit) > 0:
            right_fit_average = np.average(right_fit, axis=0)
            right_line = make_coordinates(img, right_fit_average)
        # try:
        # except Exception as e:
        #     print(e, '\n') # print error to console
        #     return None
        if len(left_fit) > 0 and len(right_fit) > 0:
            # rectangle = np.array( [[ (200, img_height), (1100, img_height), (550, 250) ]] )
            return np.array([left_line, right_line])


# func operation: colored image => bw edge marked image
def canny_func(img):
    # convert image to grayscale
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # gaussian blur => reduce noise (smoothened)
    # canny alg 1st step is gaussian blur, maybe this step is unnecessary
    blur_img = cv2.GaussianBlur(grayscale_img, (5, 5), 0)

    # use canny alg to find image's edges
    # canny = 1.gaussian blur, 2. find gradient intensity
    # 3. scan img then remove all unwanted pixel in order recive
    # only the edges, 4. thresholding, find which edge is eade for really
    # and filter who is not
    canny_img = cv2.Canny(blur_img, 10, 40)
    # original:     canny_img = cv2.Canny(blur_img, 50, 150)

    return canny_img


def display_rect(img, lines):
    rect = []
    mask = np.zeros_like(img)
    if lines is not None:
        a = lines[0][2]
        c = lines[0][3]
        b = lines[0][0]
        d = lines[0][1]
        rect.append((b, d))
        rect.append((a, c))
        a = lines[1][2]
        c = lines[1][3]
        b = lines[1][0]
        d = lines[1][1]
        rect.append((b, d))
        rect.append((a, c))
        # cv2.line(img, rect[0], rect[2], (255, 0, 0), 10)
        # cv2.line(img, rect[1], rect[3], (255, 0, 0), 10)

        recta = np.array([[rect[1], rect[0], rect[2], rect[3]]])

        cv2.fillPoly(mask, recta, 255)
    return mask

def display_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_img


# ROI
def region_of_interest(img):
    img_height = img.shape[0]
    triangle = np.array( [[ (200, img_height), (1100, img_height), (550, 250) ]] )
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, triangle, 255)
    masked_img = cv2.bitwise_and(img, mask) # 6_masked_img
    return masked_img