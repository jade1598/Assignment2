import cv2

################################################# Step 1: Use an edge detector (Canny) to create an edge image #################################################

#initialize the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the frame.")
        break

    # grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use an edge detector (such as Canny) to create an edge image.
    edges = cv2.Canny(gray, 50, 150)  #edges = cv2.Canny(image, threshold1, threshold2, apertureSize=3, L2gradient=False) 
    # =>threshold1: The lower it is, the more it will detect weak edges. threshold2: The higher it is, the more it will detect prominent edges.


    #display the edge-detected image
    cv2.imshow("Edge Detection", edges)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


################################################# Step 2: Write the (x, y) coordinates of edge pixels into an array #################################################
import numpy as np


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the frame.")
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    #extract edge pixel coordinates
    edge_points = np.column_stack(np.where(edges > 0))  # (x, y) where is not black
    print("Edge Pixel Coordinates:", edge_points)

    cv2.imshow("Edge Detection", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

################################################# Step 3: Use RANSAC to fit a straight line to the edge pixels #################################################

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    edge_points = np.column_stack(np.where(edges > 0))

    #ensure there are enough points to fit a line
    if len(edge_points) > 2:
        #fit a line using RANSAC
        edge_points = np.array(edge_points, dtype=np.float32)
        [vx, vy, x0, y0] = cv2.fitLine(edge_points, cv2.DIST_L2, 0, 0.01, 0.01)

        #calculate the line's endpoints
        slope = vy / vx
        intercept = y0 - slope * x0
        y1 = 0  #point at the top of the image
        y2 = frame.shape[0] - 1  # point at the bottom of the image

        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        #draw the line on the frame
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
   
    cv2.imshow("Line Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
