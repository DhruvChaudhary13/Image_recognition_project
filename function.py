# import os
# import cv2


# folder_path="C:/Users/Dhruv Chaudhary/Desktop/image_recognition_chatbot/scanned_images"

# def save_image(image,folder_path, file_name="captured_image.jpg"):
#                     full_path = os.path.join(folder_path, file_name)
#                     cv2.imwrite(folder_path, image)
#                     # Ensure the folder exists
#                     if not os.path.exists(folder_path):
#                         os.makedirs(folder_path)


count = 0

def increment():
    global count  # Use the global 'count' variable
    count += 1

increment()
print(count)  # Output: 1