from __future__ import print_function
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os
import shutil
import time

# text to be generated
label_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '=', 11: '+', 12: '-', 13: '×', 14: '÷'}

# The folder corresponding to the text, create a file for each category
for value, char in label_dict.items():
    train_images_dir = "dataset/" + str(value)
    if os.path.isdir(train_images_dir):
        shutil.rmtree(train_images_dir)
    os.mkdir(train_images_dir)

# Generated images
def makeImage(label_dict, font_path, width=24, height=24, rotate=0):
    for value, char in label_dict.items():
        time_value = int(round(time.time() * 1000))
        img_path = "dataset/{}/img-{}_r-{}_{}.png".format(value, value, rotate, time_value)
        font = ImageFont.truetype(font_path, int(width*0.9))
        # print(font_path)
        # print(font)
        # print(char)
        # print(value)
        # print(label_dict[value])
        # print(label_dict[value] == char)
        # print(label_dict[value] == char)
        if label_dict[value] == char:
            image = Image.new('RGB', (width, height), "black")
            draw = ImageDraw.Draw(image)
            # Get the width and height of the font
            font_width, font_height = draw.textsize(char, font)
            # Calculate the x,y coordinates of the font drawing, so that the text is drawn in the center of the icon
            x = (width - font_width - font.getoffset(char)[0]) / 2
            y = (height - font_height - font.getoffset(char)[1]) / 2
            draw.text((x, y), char, (255, 255, 255), font=font)
            image = image.rotate(rotate)
            image.save(img_path)
            # image.show()
            # time.sleep(1)

font_dir = "./fonts"
for font_name in os.listdir(font_dir):
    # Take out each font and generate a batch of images for each font
    path_font_file = os.path.join(font_dir, font_name)
    # Tilt angle from -10 to 10 degrees, each angle generates a batch of pictures
    for k in range(-10, 10, 1):
        makeImage(label_dict, path_font_file, rotate=k)