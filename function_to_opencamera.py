import cv2

def cam_open(c):
    video_cap = cv2.VideoCapture(0)

    while True:
        ret, vido_data = video_cap.read()  # Capture a video frame
        
        cv2.imshow("vido_live", vido_data)  # Display the frame in a window vdo_live si the frame and the data cature will got videodata which is then readed

        # Break the loop if the 'a' key is pressed
        if cv2.waitKey(5) == ord("e"):
            break

    # Release the video capture object and close the display window
    video_cap.release()
    cv2.destroyAllWindows()   


