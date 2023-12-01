import cv2
import numpy as np
import random

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_and_count_objects(frame, min_area=20, max_distance=-250):
    preprocessed = preprocess_frame(frame)
    edged = cv2.Canny(preprocessed, 30, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    merged_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            if not merged_contours:
                merged_contours.append(contour)
            else:
                merged = False
                M = cv2.moments(contour)
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                for index, merged_contour in enumerate(merged_contours):
                    if cv2.pointPolygonTest(merged_contour, (cX, cY), False) < max_distance:
                        merged_contours[index] = np.concatenate((merged_contour, contour))
                        merged = True
                        break
                if not merged:
                    merged_contours.append(contour)
    
    count = 0
    for contour in merged_contours:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.drawContours(frame, [contour], -1, color, 2)
        
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        cv2.putText(frame, str(count), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        count += 1
            
    return frame, count

cap = cv2.VideoCapture(0)  # Use 0 for default camera, or specify the path to a video file

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame, object_count = detect_and_count_objects(frame)
    
    print(f"Number of objects: {object_count}")
    
    cv2.imshow("Processed Frame", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
