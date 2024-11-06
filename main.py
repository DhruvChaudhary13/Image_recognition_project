import tensorflow as tf
import cv2
import os

# Disable oneDNN optimizations in TensorFlow (optional, as TensorFlow is not being used here)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Start capturing video from the webcam
video_cap = cv2.VideoCapture(0)

while True:
    ret, vido_data = video_cap.read()  # Capture a video frame
    
    cv2.imshow("vido_live", vido_data)  # Display the frame in a window vdo_live si the frame and the data cature will got videodata which is then

    # Break the loop if the 'a' key is pressed
    if cv2.waitKey(5) == ord("e"):
        break

# Release the video capture object and close the display window
video_cap.release()
cv2.destroyAllWindows()
