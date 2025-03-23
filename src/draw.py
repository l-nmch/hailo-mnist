import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw

root = tk.Tk()
root.title("Draw a digit")
root.geometry("300x450")

canvas_width = 280
canvas_height = 280
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

image = Image.new("L", (canvas_width, canvas_height), color=255)
draw = ImageDraw.Draw(image)


def clear_canvas():
    """Clear the canvas and reset the drawing area."""
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill="white")


def paint(event):
    """Draw on the canvas and update the image."""
    x1, y1 = (event.x - 2), (event.y - 2)
    x2, y2 = (event.x + 2), (event.y + 2)
    canvas.create_rectangle(x1, y1, x2, y2, fill="black", width=30)
    draw.line([x1, y1, x2, y2], fill="black", width=30)


def save_image():
    """Save the current image to a file."""
    filename = image_name.get()
    if filename:
        image.save(f"{filename}.png")
        messagebox.showinfo("Image Saved", f"Image saved as {filename}.png")
    else:
        messagebox.showwarning("No Filename", "Please enter a filename.")


clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack(pady=10)

save_button = tk.Button(root, text="Save", command=save_image)
save_button.pack(pady=10)

image_name = tk.Entry(root)
image_name.pack(pady=10)

canvas.bind("<B1-Motion>", paint)

root.mainloop()
