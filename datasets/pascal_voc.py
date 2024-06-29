import os
import math
import xml.etree.ElementTree as ET
import numpy as np

from .imageDB import imageDB

class pascal_voc(imageDB):
    def __init__(self, image_set, year)