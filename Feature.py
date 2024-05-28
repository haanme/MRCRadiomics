

from abc import ABC, abstractmethod
from features.Utils import *


"""
Radiomic feature base class
"""


class Feature(ABC):

    """
    Initialization

    @param name: short name of the feature
    @param params: parameter list for the feature instance
    """
    def __init__(self, name, params):
        self.name = name
        self.params = params

    """
    Return short name of this feature 

    @return string without spaces
    """

    def get_name(self):
        return self.name


    """
    Executes the feature
    
    @param intensity_images: intensity values images
    @param foreground_mask_images: foreground mask images
    @param background_mask_images: background mask images
    @param resolution: image resolution mm x mm x mm
    @return number of return values matching get_return_value_descriptions
    """
    @abstractmethod
    def fun(self, intensity_images, foreground_mask_images, background_mask_images, resolution, **kwargs):
        pass

    """
    Returns list of output value short names 
    
    @return list of return value short names, without spaces
    """
    @abstractmethod
    def get_return_value_short_names(self):
        pass

    """
    Returns list of input value descriptions 

    @return list of stings, or None
    """

    @abstractmethod
    def get_input_descriptions(self):
        pass

    """
    Returns list of strings decsribing boilerplate information about feature, including citations, if nay
    """

    @staticmethod
    @abstractmethod
    def get_boilerplate():
        pass

    """
    Returns true if background region is needed for this feature to be executed
    """
    @abstractmethod
    def need_background(self):
        pass

    """
    Returns true if foreground region is needed for this feature to be executed
    """
    @abstractmethod
    def need_foreground(self):
        pass

    """
    Returns number of required intensity images
    """
    @abstractmethod
    def number_of_intensity_images_required(self):
        pass

    """
    Returns number of required foreground mask images
    """
    @abstractmethod
    def number_of_foreground_mask_images_required(self):
        pass

    """
    Returns number of required background mask images
    """
    @abstractmethod
    def number_of_background_mask_images_required(self):
        pass


"""
Radiomic feature base class for methods using one index lesion and background mask
"""


class FeatureIndexandBackground(Feature):
    """
    Initialization

    @param name: name of the feature class, without spaces
    """

    def __init__(self, name, params):
        super(FeatureIndexandBackground, self).__init__(name, params)


    """
    Executes the feature

    @param intensity_images: intensity values images
    @param foreground_mask_images: foreground mask images
    @param background_mask_images: background mask images
    @param resolution: image resolution mm x mm x mm
    @return number of return values matching get_return_value_descriptions
    """

    @abstractmethod
    def fun(self, intensity_images, foreground_mask_images, background_mask_images, resolution, **kwargs):
        pass

    """
    Returns list of output value short names 

    @return list of return value short names, without spaces
    """

    @abstractmethod
    def get_return_value_short_names(self):
        pass


    """
    Returns true if background region is needed for this feature to be executed
    """

    def need_background(self):
        return self.number_of_background_mask_images_required() > 0

    """
    Returns true if foreground region is needed for this feature to be executed
    """

    def need_foreground(self):
        return self.number_of_foreground_mask_images_required() > 0

    """
    Returns number of required intensity images
    """

    def number_of_intensity_images_required(self):
        return 1

    """
    Returns number of required foreground mask images
    """

    def number_of_foreground_mask_images_required(self):
        return 1

    """
    Returns number of required background mask images
    """

    def number_of_background_mask_images_required(self):
        return 1
