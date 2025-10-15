import cv2
import numpy as np

# Start webcam
cap = cv2.VideoCapture(0)

# Kernel for morphological operations
kernel = np.ones((3,3), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame to avoid mirror effect
    frame = cv2.flip(frame, 1)

    # Expanded ROI for better hand detection
    x0, y0, x1, y1 = 50, 50, 450, 450
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
    roi = frame[y0:y1, x0:x1]

    # Convert ROI to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Tuned HSV skin color range (adjust if needed)
    lower_skin = np.array([0, 20, 40], dtype=np.uint8)
    upper_skin = np.array([25, 200, 255], dtype=np.uint8)

    # Create skin mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply Gaussian blur
    mask = cv2.GaussianBlur(mask, (5,5), 100)

    # Morphological operations to reduce noise
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        # Find largest contour (assume it's the hand)
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        # Ignore small contours (noise)
        if cv2.contourArea(contour) < 2000:
            raise Exception("Contour too small")

        # Draw hand contour
        cv2.drawContours(roi, [contour], -1, (255, 255, 0), 2)

        # Convex hull
        hull = cv2.convexHull(contour)
        cv2.drawContours(roi, [hull], -1, (0, 255, 255), 2)

        # Convexity defects (spaces between fingers)
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull_indices)

        count_defects = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                # Compute angle using cosine rule
                a = np.linalg.norm(np.array(end) - np.array(start))
                b = np.linalg.norm(np.array(far) - np.array(start))
                c = np.linalg.norm(np.array(end) - np.array(far))
                angle = np.arccos((b**2 + c**2 - a**2)/(2*b*c)) * 57

                # Ignore tiny defects by distance and angle
                if angle <= 90 and np.linalg.norm(np.array(far)-np.array(start)) > 20:
                    count_defects += 1
                    cv2.circle(roi, far, 5, [0, 0, 255], -1)

        # Number of fingers = defects + 1
        fingers = count_defects + 1

        # Display finger count
        cv2.putText(frame, f"Fingers: {fingers}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    except:
        cv2.putText(frame, "No Hand Detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Show windows
    cv2.imshow("Hand Gesture", frame)
    cv2.imshow("Mask", mask)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
