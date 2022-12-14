import re

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import cv2

def find_object(one_image):
    img = cv2.imread(one_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # do some noise reduction
    d = 3
    g_sigmaColor = 10
    g_sigmaSpace = 10
    gray = cv2.bilateralFilter(gray, d, g_sigmaColor, g_sigmaSpace)

    # now detect edges and floodfill the binary image, dilation allows joining edges with holes
    edges = cv2.Canny(gray, 100, 200)
    im_th = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    im_floodfill = im_th.copy()

    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    kernel = np.ones((5, 5), np.uint8)
    im_out = cv2.erode(im_out, kernel, iterations=1)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(im_out, connectivity=4)

    # Create false color image
    colors = np.arange(0, 200, 200 / n_labels).astype(np.uint8)
    colors[0] = 0  # for cosmetic reason we want the background black
    false_colors = colors[labels]

    max_br = 0
    num_objects = 0
    area_thr = 0.3
    whole_area = im_out.shape[0] * im_out.shape[1]
    brightest_color = 0
    for i in range(len(colors)):
        object_area = np.sum(false_colors == colors[i])
        # print(object_area / whole_area * 100, "Remove:", object_area < whole_area * area_thr / 100)
        if object_area < whole_area * area_thr / 100:
            false_colors[false_colors == colors[i]] = 0
        else:
            num_objects += 1
            # print("area brightness:", np.mean(gray[false_colors == colors[i]]))
            cur_br = np.mean(gray[false_colors == colors[i]])
            if colors[i] != 0 and cur_br > max_br:
                max_br = cur_br
                brightest_color = colors[i]

    false_colors[false_colors == brightest_color] = 255  # colorize drawn area with the brightest color
    false_colors[np.logical_and(false_colors != 255, false_colors > 0)] = 128  # colorize objects to fit into area

    plt.figure(figsize=(20, 20))
    plt.subplot(2, 2, 1), plt.imshow(gray, cmap="gray"), plt.title(f"Input image")
    plt.subplot(2, 2, 2), plt.imshow(false_colors), plt.title(f"Final object")
    plt.savefig(f"result_{os.path.basename(one_image)}")
    return false_colors

def predict(False_colors):
    marked_area_coords = np.argwhere(False_colors == 255)
    area_crop_coords = (np.min(marked_area_coords[:, 0]),
                        np.min(marked_area_coords[:, 1]),
                        np.max(marked_area_coords[:, 0]),
                        np.max(marked_area_coords[:, 1]))

    object_area_coords = np.argwhere(False_colors == 128)
    object_crop_coords = (np.min(object_area_coords[:, 0]),
                          np.min(object_area_coords[:, 1]),
                          np.max(object_area_coords[:, 0]),
                          np.max(object_area_coords[:, 1]))

    area = False_colors[area_crop_coords[0]:area_crop_coords[2], area_crop_coords[1]:area_crop_coords[3]]
    object = False_colors[object_crop_coords[0]:object_crop_coords[2], object_crop_coords[1]:object_crop_coords[3]]

    object_fits_in_area = False
    source_area = np.sum(area > 0)
    if np.all(np.less(object.shape, area.shape)):
        for i in range(area.shape[0] - object.shape[0]):
            for j in range(area.shape[1] - object.shape[1]):
                # put object in coords (i,j) over area
                tmp_area = area.copy()
                tmp_area[i:i + object.shape[0], j:j + object.shape[1]][object > 0] = object[object > 0]
                new_area = np.sum(tmp_area > 0)
                if new_area <= source_area:
                    object_fits_in_area = True
                    break
            if object_fits_in_area:
                break
    return object_fits_in_area

if __name__ == "__main__":

    num = 1
    with open("result.txt", "w",encoding='utf-8') as file:
        file.write("результат: \n")
    image_list = glob.glob("./input/*.jpg")
    new_image_list= sorted(image_list, key=lambda info: (info[0], int(info[8:10])))

    for one_image in new_image_list:
        false_colors = find_object(one_image)
        predicted_value = predict(false_colors)

        actual_value = False if "False" in one_image else True

        with open ("result.txt","a",encoding='utf-8') as file:
            file.write("picture ")
            file.write(str(num))
            file.write("\n Predicted value:")
            file.write(str(predicted_value))
            file.write("\n actual value:")
            file.write(str(actual_value))
            file.write("\n Is correct:")
            file.write(str(predicted_value == actual_value))
            file.write("\n")
        print("picture",num, "\n","Predicted value:", predicted_value,"\n","actual value:", actual_value,"\n"
              ,"Is correct:", predicted_value == actual_value, "\n")
        num = num + 1
    file.close()

