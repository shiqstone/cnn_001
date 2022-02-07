import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import PIL
import matplotlib.pyplot as plt
import os
import shutil
from numpy.core.records import array
from numpy.core.shape_base import block
import time
from collections import Counter
import cnn


# The Y-axis projection of the entire image, passed in the image array, the image is binarized and inverted
import shutil

# Y-axis projection of the entire image
def img_y_projection(img):
    # Calculate the Y-axis projection of the entire image
    (h, w) = img.shape
    # Initialize an array with the same length as the image height to record the number of black points in each line
    arr = [0 for z in range(0, h)]
    # Traverse each column and record how many valid pixels this column contains
    for i in range(0,  h):
        for j in range(0, w):
            if img[i, j] == 255:
                arr[i] += 1
    return arr

# The X-axis projection of the one line in image
def img_x_projection(img):
    # Caculate the projection
    (h, w) = img.shape 
    # Initialize an array with the same length as the image width to record the number of black points in each line
    arr = [0 for z in range(0, w)]
    # Traverse each column and record how many valid pixels this column contains
    for i in range(0, w):
        for j in range(0, h):
            if img[j, i] == 255:
                arr[i] += 1
    return arr



# Show image
def image_show_array(a):
    plt.imshow()
    plt.show()

# Display the projection image, 
# the input parameter arr is the two-dimensional array of the image, and the direction is the x, y axis
def show_projection(arr, direction = 'x'):
    arr_max = max(arr)
    if direction == 'x': # Projection in the x-axis direction
        arr_projection = np.zeros((arr_max, len(arr)), dtype=int)
        for i in range(0, len(arr)):
            if arr[i] == 0:
                continue
            for j in range(0, arr[i]):
                arr_projection[j, i] = 255
    elif direction == 'y': # Projection in the y-axis direction
        arr_projection = np.zeros((len(arr), arr_max), dtype=int)
        for i in range(0, len(arr)):
            if arr[i] == 0:
                continue
            for j in range(0, arr[i]):
                arr_projection[i, j] = 255
    image_show_array(arr_projection)

# # The picture gets the text block, passes in the projection list, 
# and returns the coordinates of the marked array area [[left, top, right, bottom]]
def img2rows(arr, w, h):
    # Split tiles based on projection
    inLine = False
    start = 0 # The starting index of a split
    mark_boxs = []
    for i in range(0, len(arr)):
        if inLine == False and arr[i] > 10:
            inLine = True
            start = i
        # Record the selected area [left, top, right, bottom], 
        # up and down is the picture, left and right are start to current
        elif i - start > 5 and arr[i] < 10 and inLine:
            inLine = False
            if i - start > 10:
                top = max(start-1, 0)
                bottom = min(i+1, h)
                box = [0, top, w, bottom]
                mark_boxs.append(box)
    return mark_boxs

def row2blocks(arr, w, h):
    inLine = False
    start = 0
    block_mark_boxs = []

    for i in range(0, len(arr)):
        if inLine == False and arr[i] > 2:
            inLine = True
            start = i
        elif i - start > 10 and arr[i] < 2 and inLine:
            inLine = False
            left = max(start-1, 0)
            right = min(w, i+1)
            box = [left, 0, right, h]
            block_mark_boxs.append(box)
    return block_mark_boxs

def block2chars(arr, w, h, row_top, block_left):
    inLine = False
    start = 0
    char_mark_boxs = []
    abs_char_mark_boxs = []

    for i in range(0, len(arr)):
        if inLine == False and arr[i] > 0:
            inLine = True
            start = i
        elif i - start > 5 and arr[i] < 1 and inLine:
            inLine = False
            left = max(start-1, 0)
            right = min(w, i+1)
            box = [left, 0, right, h]
            char_mark_boxs.append(box)
            abs_box = [block_left + left, row_top, block_left + right, row_top + h]
            abs_char_mark_boxs.append(abs_box)
    return char_mark_boxs, abs_char_mark_boxs


# Crop image, 'img' image array, 'mark_boxs' area mark
def crop_image(img, mark_boxs, is_square = False):
    img_items = []
    for i in range(0, len(mark_boxs)):
        img_org = img.copy()
        box = mark_boxs[i]
        # corp image
        img_crop = img_org[box[1]:box[3], box[0]:box[2]]
        if is_square:
            img_crop = get_square_img(img_crop)
        img_items.append(img_crop)
    return img_items

# Save image
def save_images(dir_name, imgs):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    img_paths = []
    for i in range(0, len(imgs)):
        img_path = os.path.join(dir_name, 'part_{}.png'.format(i))
        cv2.imwrite(img_path, imgs[i])
        img_paths.append(img_path)

    return img_paths

# show image by path way
def img_show_path(img_path):
    pil_im = Image.open(img_path)
    plt.imshow(pil_im)
    plt.show()

# show image by array way
def img_show_array(img_array):
    plt.imshow(img_array)
    plt.show()

def get_square_img(image):
    x, y, w, h = cv2.boundingRect(image)
    image = image[y:y+h, x:x+w]

    max_size = 18
    max_size_and_border = 24

    if w > max_size or h > max_size:
        if w >= h:
            times = max_size / w
            w = max_size
            h = int(h * times)
        else:
            times = max_size / h
            h = max_size
            w = int(w * times)
        image = cv2.resize(image, (w, h))

    xw = image.shape[0]
    xh = image.shape[1]

    xwLeftNum = int((max_size_and_border - xw) / 2)
    xwRightNum = max_size_and_border - xw - xwLeftNum

    xhTopNum = int((max_size_and_border - xh) / 2)
    xhBottomNum = max_size_and_border - xh - xhTopNum

    img_large = np.pad(image, ((xwLeftNum, xwRightNum), (xhTopNum, xhBottomNum)), 'constant', constant_values=(0,0))
    return img_large

def div_img(img_path, save_file = False):
    thresh = 200
    img_o = cv2.imread(img_path, 1)

    img = cv2.imread(img_path, 0)
    (img_h, img_w) = img.shape
    # Binarize the entire graph, for branching
    ret, img_bin = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)

    # Calculate the projection, and intercept the lines of the entire image
    img_y_projection_arr = img_y_projection(img_bin)
    row_mark_boxs = img2rows(img_y_projection_arr, img_w, img_h)

    row_imgs = crop_image(img, row_mark_boxs)
    all_mark_boxs = []
    all_char_boxs = []

    # crop row image
    for i in range(0, len(row_imgs)):
        row_img = row_imgs[i]
        (row_img_h, row_img_w) = row_img.shape
        # Binarize the entire graph, for branching
        ret, row_img_bin = cv2.threshold(row_img, thresh, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        # dilate image 6 times
        row_img_bin_dilate = cv2.dilate(row_img_bin, kernel, iterations=6)
        img_x_projection_arr = img_x_projection(row_img_bin_dilate)
        block_mark_boxs = row2blocks(img_x_projection_arr, row_img_w, row_img_h)
        row_char_boxs = []
        row_char_imgs = []
        block_imgs = crop_image(row_img, block_mark_boxs)
        if save_file:
            bin_imgs = save_images('imgs/cuts/row_{}'.format(i), block_imgs)
        # crop block for char
        for j in range(0, len(block_imgs)):
            block_img = block_imgs[j]
            (block_img_h, block_img_w) = block_img.shape
            # Binarize the entire graph, for branching
            ret, block_img_bin = cv2.threshold(block_img, thresh, 255, cv2.THRESH_BINARY_INV)
            block_img_x_projection_arr = img_x_projection(block_img_bin)
            row_top = row_mark_boxs[i][1]
            block_left = block_mark_boxs[j][0]
            char_mark_boxs, abs_char_mark_boxs = block2chars(block_img_x_projection_arr, block_img_w, block_img_h, row_top, block_left)
            row_char_boxs.append(abs_char_mark_boxs)
            char_imgs = crop_image(block_img_bin, char_mark_boxs, True)
            row_char_imgs.append(char_imgs)
            if save_file:
                c_imgs = save_images('imgs/cuts/row_{}/blocks_{}'.format(i, j), char_imgs)
        all_mark_boxs.append(row_char_boxs)
        all_char_boxs.append(row_char_imgs)
    return all_mark_boxs, all_char_boxs, img_o

def calculation(chars):
    cstr = ''.join(chars)
    result = ''

    if("=" in cstr):
        str_arr = cstr.split("=")
        c_str = str_arr[0]
        r_str = str_arr[1]
        c_str = c_str.replace("×", "*")
        c_str = c_str.replace("÷", "/")
        try:
            print("eval:", c_str)
            c_r = int(eval(c_str))
        except Exception as e:
            print("Exception:", e)

        if r_str == "":
            result = c_r
        else:
            if str(c_r) == str(r_str):
                result = "√"
            else:
                result = "×"
    return result

# Draw text
def cv2ImgAddText(img, text, left, top, text_color=(255, 0, 0), text_size=20):
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype("fonts/STHeiti Medium.ttc", text_size, encoding="utf-8")
    draw.text((left, top), text, text_color, font=fontStyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def check(path, save = False):
    all_mark_boxs, all_char_imgs, img_o = div_img(path, save)
    model = cnn.create_model()
    model.load_weights('checkpoint/my_checkpoint')
    class_name = np.load('checkpoint/class_names.npy')

    # Traverse rows
    for i in range(0, len(all_char_imgs)):
        row_imgs = all_char_imgs[i]
        # Traverse blocks
        for j in range(0, len(row_imgs)):
            block_imgs = row_imgs[j]
            block_imgs = np.array(block_imgs)
            # image recognition
            results = cnn.predict(model, block_imgs, class_name)
            print('recognize result:',results)
            # calculate
            result = calculation(results)
            print('calculate result:',result)
            # Get the coordinates of the block
            block_mark = all_mark_boxs[i][j]
            answer_box = block_mark[-1]
            x = answer_box[2]
            y = answer_box[3]
            iw = answer_box[2] - answer_box[0]
            ih = answer_box[3] - answer_box[1]
            textSize = max(iw, ih)
            if str(result) == "√":
                color = (0, 255, 0)
            elif str(result) == "×":
                color = (255, 0, 0)
            else:
                color = (192, 192,192)
            img_o = cv2ImgAddText(img_o, str(result), answer_box[2], answer_box[1], color, textSize)
    output_path = 'imgs/question_result.png'
    cv2.imwrite(output_path, img_o)



if __name__ == '__main__':
    t = time.time()
    input_path = 'imgs/question.png'
    check(input_path, True)
    print(f'all take time:{time.time() - t:.4f}s')