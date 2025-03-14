import cv2
import os
import random
import matplotlib.pyplot as plt

def convert_to_grayscale(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Unable to load image {input_path}.")
        return
    
    if len(img.shape) == 3 and img.shape[2] != 1:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(output_path, img_gray)
        print(f"Converted {input_path} to grayscale and saved to {output_path}")
    else:
        print(f"{input_path} is already grayscale.")

def resize_image(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Unable to load image {input_path}.")
        return
    
    resized_img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, resized_img)
    print(f"Resized {input_path} and saved to {output_path}")

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        convert_to_grayscale(input_path, output_path)
        resize_image(output_path, output_path)

# Set your input and output directories here
input_dir = "/media/user/9c7eaef1-35fa-4210-889c-9e2b99342586/user/abul/dataset sdo patches"  # Replace with your input folder path
output_dir = "/media/user/9c7eaef1-35fa-4210-889c-9e2b99342586/user/abul/dataset sdo patches low res"  # Replace with your output folder path

# Process images
process_images(input_dir, output_dir)
