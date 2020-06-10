from cv2 import cv2
import numpy as np

class SealComparator():

    # HSV threshold
    hue_low = 175
    hue_high = 5
    saturation_low = 5
    saturation_high = 255
    value_low = 5
    value_high = 255

    def __call__(self, seal):
        binary_image = self.threshold(seal)
        noise_reduced_image = self.reduce_noise(binary_image)
        fixed_image = self.close_calculate(noise_reduced_image)

    def threshold(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        (height, width, _) = hsv_image.shape
        output = np.zeros((height, width), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                hsv = hsv_image[y, x]
                if (hsv[0] >= self.hue_low or hsv[0] <= self.hue_high) and \
                    (hsv[1] >= self.saturation_low and hsv[1] <= self.saturation_high) and \
                    (hsv[2] >= self.value_low and hsv[2] <= self.value_high):
                    output[y, x] = 255
        return output
    
    def reduce_noise(self, image):
        median_blur = cv2.medianBlur(image, 5)
        return cv2.GaussianBlur(median_blur, (5, 5), 0)
    
    def close_calculate(self, image):
        kerenal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kerenal)