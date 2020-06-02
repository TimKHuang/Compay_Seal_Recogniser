from cv2 import cv2
import numpy as np
import argparse

def get_path():
    '''
    Get the path of the image.
    Use a paramter from command line or Use the input()
    '''
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=False)
    args = vars(ap.parse_args())
    if args['image']:
        return args['image']
    return input('Please enter the file path\n')

def preprocess(img_path):
    '''
    Change the image to gray scale image to reduce calculation
    '''
    img = cv2.imread(img_path)
    origin = img.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (origin, gray_img)

def resize(img):
    '''
    Resize the image to fit the screen. Should only be called before showing.
    '''
    (height, width) = img.shape[:2]
    resize_factor = 1080.0 / height
    if resize_factor > 1:
        resize_factor = 1
    height = height * resize_factor
    width = width * resize_factor
    return cv2.resize(img, (int(width), int(height)))

def edge(img):
    '''
    Edge Detection.
    '''
    binary_image= cv2.threshold(img, np.average(img) - 20, 255, cv2.THRESH_BINARY)[1]
    edged_image= cv2.Canny(binary_image, 50, 200)
    return edged_image

def circle_detection(img, min_r, max_r, step_r, step_theta, accept_ratio):
    '''
    Find the circles in the images
    '''
    (height, width) =img.shape
    r_number = int((max_r - min_r)/ step_r)
    theta_number = int((2 * np.pi - 0)/step_theta)
    hough_space = np.zeros((height, width,max_r), dtype=int)

    for y in range(height):
        for x in range(width):
            if img[y][x] == 255:
                for r in range(r_number):
                    for theta in range(theta_number):
                        radius = int(min_r + r * step_r)
                        angle = 0 + theta * step_theta
                        para_x = int(x - radius * np.cos(angle))
                        para_y = int(y - radius * np.sin(angle))
                        if para_x < 0 or para_x >= width or para_y < 0 or para_y >= height or r >= max_r:
                            continue
                        hough_space[para_y][para_x][radius] += 1
    max_value = max([max([max(z) for z in x]) for x in hough_space])
    accept_value = max_value * accept_ratio
    return [ (a, b , r) for a in range(width) for b in range(height) for r in range(max_r) if hough_space[b][a][r] >= accept_value]

def draw_circle(img, circles):
    '''
    draw all the circles on a image
    '''
    for (x, y, r) in circles:
        cv2.circle(img, (x, y), r, (255, 255, 255), 4)

def show(img):
    '''
    Show the img in a proper size
    '''
    cv2.imshow('Result', resize(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    (image, gray_image) = preprocess(get_path())
    edged = edge(gray_image)
    # show(edged)
    draw_circle(edged, circle_detection(edged, 30, 100, .5, .5, 1))
    show(image)
    show(edged)