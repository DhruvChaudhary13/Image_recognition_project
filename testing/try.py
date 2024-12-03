import tkinter as tk
from tkinter import filedialog
from tkinter import Label
from PIL import Image, ImageTk

def open_image():
    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
    if file_path:
        # Load and display the image
        img = Image.open(file_path)
        img = img.resize((300, 300), Image.ANTIALIAS)  # Resize image for display
        img_tk = ImageTk.PhotoImage(img)
        
        # Update the image label
        image_label.config(image=img_tk)
        image_label.image = img_tk
        path_label.config(text=f"Selected Image: {file_path}")

# Create the main application window
root = tk.Tk()
root.title("Image Input GUI")



# #adding bg image 
# bg_image = Image.open("C:/Users/Dhruv Chaudhary/Desktop/image_recognition_chatbot/space.png")  # Replace with your background image file path
# bg_image = bg_image.resize((2000,1000), Image.Resampling.LANCZOS)
# bg_photo = ImageTk.PhotoImage(bg_image)

# bg_label = tk.Label(root, image=bg_photo)
# bg_label.place(relwidth=1, relheight=1)


# Add a button to open the image
open_button = tk.Button(root, text="Select Image",fg="red",bg="white",font=("arial 30 bold"), command=open_image)
open_button.place(x=300,y=400)

# Add a label to display the selected image path
path_label = tk.Label(root, text="No image selected", wraplength=400)
path_label.pack(pady=10)

# Add a label to display the image
image_label = tk.Label(root)
image_label.pack(pady=20)

# Run the application
root.mainloop()
