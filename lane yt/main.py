import cv2
import numpy as np

def detect_lanes(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using the Canny algorithm
    edges = cv2.Canny(blur, 50, 150)

    # Define a region of interest (ROI) mask
    height, width = image.shape[:2]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]], np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, polygons, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Perform Hough line transform to detect lines in the ROI
    lines = cv2.HoughLinesP(masked_edges, rho=6, theta=np.pi/60, threshold=160, minLineLength=40, maxLineGap=25)

    # Draw the detected lines on the image
    draw_lines(image, lines)

    return image

def draw_lines(image, lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

# Open the video file
video_capture = cv2.VideoCapture('test2.mp4')

while video_capture.isOpened():
    # Read the current frame from the video
    ret, frame = video_capture.read()

    if not ret:
        break

    # Perform lane detection on the frame
    lanes_image = detect_lanes(frame)

    # Display the result
    cv2.imshow('Lane Detection', lanes_image)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video_capture.release()
cv2.destroyAllWindows()
