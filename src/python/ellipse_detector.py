from cv2 import cv2
import numpy as np
import subprocess

class EllipseDetector():
    
    # image
    image = None
    resized_image = None
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
    min_radian = np.pi * 0.1
    # The minimum radius relative to the biggest ellipse detected
    min_relative_radius = 0.8
    # path
    output_path = ''
    intermediate_path = 'images/intermediate/in.pgm'
    elsdc_path = 'src/ELSDC/elsdc'
    # process size
    max_height = 900
    max_width = 1200

    def __call__(self, input_img, output_path):
        self.image = cv2.imread(input_img)
        self.output_path = output_path
        self.generate_pgm()
        self.elsdc()
        self.read_result()
        self.select()
        self.output()

    def generate_pgm(self):
        self.resized_image = self.resize(self.image, self.max_height, self.max_width)
        gray_image = cv2.cvtColor(self.resized_image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(self.intermediate_path, gray_image)
    
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

    def select(self):
        max_radius = 0

        # remove the ellipse with too small radian
        index = 0
        while index < len(self.ellipse_collection):
            ellipse = self.ellipse_collection[index]
            randian = ellipse.angle_end - ellipse.angle_start
            if randian < 0:
                randian += np.pi*2
            if randian < self.min_radian:
                self.ellipse_collection.pop(index)
                continue
            max_radius = max([max_radius, ellipse.major_axis])
            index += 1
        
        # remove the ellipse with too small radius
        index = 0
        while index < len(self.ellipse_collection):
            ellipse = self.ellipse_collection[index]
            if ellipse.major_axis < max_radius * self.min_relative_radius:
                self.ellipse_collection.pop(index)
                continue
            index += 1
        
        # if two ellipse center are close to each other, remove the one with shorter major radius
        index = 0
        while index < len(self.ellipse_collection):
            index2 = index + 1
            while index2 < len(self.ellipse_collection):
                ellipse1 = self.ellipse_collection[index]
                ellipse2 = self.ellipse_collection[index2]
                distance = (ellipse1.center_x - ellipse2.center_x)**2 + (ellipse1.center_y - ellipse2.center_y)**2
                parameter = (min(ellipse1.minor_axis, ellipse2.minor_axis))**2
                if distance > parameter:
                    index2 += 1
                    continue
                if ellipse1.major_axis > ellipse2.major_axis:
                    self.ellipse_collection.pop(index2)
                    continue
                self.ellipse_collection.pop(index)
                index -= 1
                break
            index += 1
                
    def output(self):
        scale = self.image.shape[0] / self.resized_image.shape[0]
        # save each ellipse
        for i in range(len(self.ellipse_collection)):
            self.save_image(self.ellipse_collection[i], scale, self.output_path + str(i) + '.png')
        
        # save the labled graph
        circled_image = self.resized_image.copy()
        for ellipse in self.ellipse_collection:
            circled_image = cv2.ellipse(circled_image,\
                (int(ellipse.center_x), int(ellipse.center_y)), \
                (int(ellipse.major_axis), int(ellipse.minor_axis)), \
                ellipse.theta, 0, 360, \
                (255, 0,0), thickness=2, lineType= cv2.LINE_8)
        cv2.imwrite(self.output_path + 'labled.png', circled_image)

    def save_image(self, ellipse, resize_scale, name):
        # get the orginal shape to prevent overflow
        (height, width, _) = self.image.shape
        # get the coordinate of the center at the original graph
        center_x = ellipse.center_x * resize_scale
        center_y = ellipse.center_y * resize_scale
        # get the area to cut
        edge_length = int(ellipse.major_axis * 2 * resize_scale * 1.5)
        output = np.zeros((edge_length, edge_length, 3), dtype=np.uint8)
        for x in range(edge_length):
            for y in range(edge_length):
                origin_x = int(center_x - edge_length / 2 + x)
                origin_y = int(center_y - edge_length / 2 + y)
                if origin_x >= 0 and origin_y >=0 and origin_x < width and origin_y < height: 
                    output[y, x] = self.image[origin_y, origin_x]
        cv2.imwrite(name, output)

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
        self.major_axis = data_list[6]
        self.minor_axis = data_list[7]
        self.theta = data_list[8]
        self.angle_start = data_list[9]
        self.angle_end = data_list[10]