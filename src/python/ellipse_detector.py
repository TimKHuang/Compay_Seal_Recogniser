from cv2 import cv2
import numpy as np
import math
import subprocess

class EllipseDetector():
    
    # image
    image = None
    # The List of Ellipse detected
    ellipse_collection = []
    # HSV threshold
    hue_low = 30
    hue_high = 170
    saturation_low = 10
    saturation_high = 255
    value_low = 5
    value_high = 255
    # The minimum randian of an arc
    min_radian = math.pi * 0.1
    # The minimum radius relative to the biggest ellipse detected
    min_relative_radius = 0.5
    # path
    output_img = ''
    intermediate_path = 'images/intermediate/in.pgm'
    elsdc_path = 'src/ELSDC/elsdc'
    # output size
    max_height = 900
    max_width = 1200

    def __call__(self, input_img, output_img):
        self.image = cv2.imread(input_img)
        self.output_img = output_img
        self.generate_pgm()
        self.elsdc()
        self.read_result()

    def generate_pgm(self):
        image = self.resize(self.image, self.max_height, self.max_width)
        mask = self.red_mask(image)
        gray_image = cv2.cvtColor(self.copy(image, mask=mask), cv2.COLOR_BGR2GRAY)
        cv2.imwrite(self.intermediate_path, self.erode(gray_image))
    
    def elsdc(self):
        command = subprocess.run([self.elsdc_path, self.intermediate_path], stdout=subprocess.PIPE, text=True)
        if command.returncode:
            raise Exception('Faild to run elsdc.')

    def read_result(self):
        file = open('out_ellipse.txt')
        collection = [ [float(number) for number in ellipse.split()] for ellipse in file.read().splitlines() ]
        for ellipse in collection:
            self.ellipse_collection.append(Ellipse(ellipse))
        subprocess.run(['rm', 'out_ellipse.txt'])

    def red_mask(self, image):
        '''
        It will choose the range that is larger than the high_threshold
        and smaller than the low_threshold so it is only suitable for red extraction.
        '''
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        (height, width, _) = hsv_image.shape

        mask = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                hsv = hsv_image[y, x]
                if (hsv[0] <= self.hue_low or hsv[0] >= self.hue_high) and \
                    (hsv[1] <= self.saturation_high and hsv[1] >= self.saturation_low) and \
                    (hsv[2] <= self.value_high and hsv[2] >= self.value_low):
                    mask[y, x] = 255
        return mask

    def copy(self, image, mask = None):
        copy = image.copy()
        if mask.any():
            (height, width, _) = image.shape
            for y in range(height):
                for x in range(width):
                    if mask[y, x]:
                        continue
                    copy[y, x] = [0, 0, 0]
        return copy

    def erode(self, image):
        kerenal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.erode(image, kerenal, iterations=2)
            
    def resize(self, image, max_height, max_width, keep_ratio = True):
        (height, width, _) = image.shape
        if keep_ratio:
            if height > width:
                return cv2.resize(image, (int(width * max_height / height), int(max_height)))
            return cv2.resize(image, (int( max_width), int(height * max_width / width)))
        return cv2.resize(image, (int(max_width), int(max_height)))

class Ellipse():
    def __init__(self, data_list):
        self.x1 = data_list[0]
        self.y1 = data_list[1]
        self.x2 = data_list[2]
        self.y2 = data_list[3]
        self.center_x = data_list[4]
        self.center_y = data_list[5]
        self.ax = data_list[6]
        self.bx = data_list[7]
        self.theta = data_list[8]
        self.angle_start = data_list[9]
        self.angle_end = data_list[10]
    