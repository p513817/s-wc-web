import os

import cv2
import numpy as np
from functools import lru_cache
import time

def crop_frame(frame: np.ndarray, region: list = [55, 116, 669, 428]) -> np.ndarray:
    """
    - Arguments
        - frame
            - type: np.ndarry
            - descr: the image buffer
        - region
            - type: list[x1, y1, x2, y2]
            - descr: the region you want to crop.
    """
    return frame[region[1] : region[3], region[0] : region[2]]


def gray_frame(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.merge((gray, gray, gray))

@lru_cache(maxsize=None)
def get_gray_background():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backg_dir = os.path.join(script_dir, "background")
    backg_path = os.path.join(backg_dir, "aida64background_bw.png")
    return gray_frame(cv2.imread(backg_path))

def process(input_frame: np.ndarray) -> np.ndarray:
    """Prcoess image function
    - Arguments
        - input_frame
            - type: np.ndarray
            - descr: the image buffer with numpy.ndarray format.
    - Return
        - output_frame
            - type: np.ndarray
            - descr: the image buffer after process.
    """

    # Crop Image
    croped = crop_frame(input_frame)

    # GrayTestImg
    grayimg = gray_frame(croped)

    # Load Background, Convert to Grayscale
    graybg = get_gray_background()
    
    # Do Subtraction
    substracted = grayimg - graybg

    # Filter Threhold
    threshold, limit = 20, 255
    ret, result_image = cv2.threshold(substracted, threshold, limit, cv2.THRESH_BINARY)

    # Return result
    return result_image


if __name__ == "__main__":
    path = r"./s-wc.png"
    frame = cv2.imread(path)
    data = process(frame)
    cv2.imwrite("cv.jpg", data)
