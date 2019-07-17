import cv2
import numpy as np
import utils


# ######################################### Video #########################################
cv2.namedWindow('road')
rect_mask = False
cap = cv2.VideoCapture('test2.mp4')
while cap.isOpened():
    isVideoRunning, frame = cap.read()
    if isVideoRunning is False:  # if video is over
        break
    canny_img = utils.canny_func(frame)
    roi_img = utils.region_of_interest(canny_img)
    lines = cv2.HoughLinesP(roi_img, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = utils.average_slope_intercept(frame, lines)
    if rect_mask:
        line_img = utils.display_rect(frame, averaged_lines)
    else:
        line_img = utils.display_lines(frame, averaged_lines)
    marked_lines_in_img = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
    cv2.imshow('road', marked_lines_in_img)
    key_pressed = cv2.waitKey(1)
    # press 'l' in order change the road marking
    if key_pressed == ord('l'):
        if rect_mask:
            rect_mask = False
        else:
            rect_mask = True
    if key_pressed == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
############################################################################################

# ################################################ Image ##########################################
# original_img = cv2.imread('test_image.jpg')
# lane_img = np.copy(original_img)
#
# canny_img = canny_func(lane_img)
#
# roi_img = region_of_interest(canny_img)
#
# lines = cv2.HoughLinesP(roi_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# # optimization
# averaged_lines = average_slope_intercept(lane_img, lines)
# line_img = display_lines(lane_img, averaged_lines)
#
#
# marked_lines_in_img = cv2.addWeighted(lane_img, 0.8, line_img, 1, 1)
#
# # cv2.imwrite('./images/10_marked_lines_in_img_optimization.jpg',marked_lines_in_img)
#
# cv2.imshow('road',roi_img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#####################################################################################################


# ############# calculate the ROI ###########
# import matplotlib.pyplot as plt
# plt.imshow(canny_img)
# plt.show(1)
#############################################