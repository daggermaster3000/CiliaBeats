"""

"""

import cv2
import tiffcapture as tc
import numpy as np
from scipy.signal import wiener, savgol_filter
import matplotlib.pyplot as plt
import os
from skimage.measure import label, regionprops
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button, TextBox
import time
import copy

class Cilia:
    """
    Class Constructor: Kymo
    -----------------------
    Initializes the Kymo class object.

    INPUTS:
    - path (str): Path to the image file.
    - pixel_size (float): Pixel size in micrometers.
    - frame_time (float): Frame time in seconds.
    - filter_size (tuple or None): Optional filter size for image filtering.

    OUTPUTS:
    - None
    """
    def __init__(self,path, pixel_size, frame_time, filter_size=None):
        np.seterr(all="ignore")
        self.path = path
        self.pixel_size = pixel_size    # in um
        self.frame_time = frame_time    # in s
        self.rect = {}  # rectangles for kept blobs
        self.images = []
        
        self.filtered_images = []
        self.kymo = []
        self.raw_kymo = []
        self.mean_velocities = []
        self.se_velocities = []
        self.binary_kymo = []
        # open the data
        self.data, self.name = self.open_tiff()
        # get some information about the data
        _, self.first_img = self.data.retrieve()
        self.dv_pos,self.width = np.shape(self.first_img)
        self.N_images = self.data.length
        # check the if a .npy was created if not create one
        self.init_bin()
        # convert to numpy array
        self.images = np.array(self.images,dtype='>u2')

        if filter_size != None:

            # if filter size is passed, filter images
            self.filtered_images = np.zeros_like(self.images)   # pre allocate to be faster
            self.filtered_images = self.filter_wiener(self.images, filter_size)

            # generate kymograph
            self.kymo = self.swap_axes(self.filtered_images)

        else:

            # generate kymograph
            self.kymo = self.swap_axes(self.images)


    # helper functions
    def open_tiff(self):
        """
        Opens a TIFF image file using the tiffcapture library.

        INPUTS:
        -------
        None

        OUTPUTS:
        --------
        tiff: tc.TiffCapture
            TiffCapture object for the image file.
        name: str
            Name of the image file.
        """
        tiff = tc.opentiff(self.path) #open img
        name = self.path.split("\\")[-1]
        
        return tiff,name

    def cache(self):
        """
        A function to save an np.array as .npy binary (not really a cache)
        """ 
        np.save("cache\\"+self.name.split(".")[0],self.images)

    def init_bin(self):
        """
        init_bin
        Initializes image data by processing time-lapse images or loading from cache if previously processed.

        INPUTS:
        -------
        None

        OUTPUTS:
        --------
        None
        """
        # check the cache to see if we haven't already processed the image
        # create cache if non existent
        if "cache" not in os.listdir():
            os.mkdir("cache")
        print("Input image: ",self.name)
        # process the time lapse to array if it hasnt previously been processed
        if self.name.split(".")[0]+".npy" not in os.listdir("cache"):    
            for ind,im in enumerate(self.data):
                print(f"Processing images {np.round(ind/self.N_images*100,1)}%",end = "\r")
                self.images.append(im)
            self.cache()
        else:
            # if it already has been processed load it from the cache
            print("Loading from previous processing!")
            self.images = np.load("cache\\"+self.name.split(".")[0]+".npy",allow_pickle=True)

    def rescale(self,array,min,max):
        """
        Performs min-max scaling on the input array.

        INPUTS:
        -------
        array: ndarray
            Input array for scaling.
        min: float
            Minimum value after scaling.
        max: float
            Maximum value after scaling.

        OUTPUTS:
        --------
        scaled_matrix: ndarray
            Scaled array.
        """
        # Perform min-max scaling
        min_val = np.min(array)
        max_val = np.max(array)
        scaled_matrix = (max-min)*(array - min_val) / (max_val - min_val) + max
        
        return scaled_matrix

    

if __name__ == "__main__":
