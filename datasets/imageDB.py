import os
import PIL

class imageDB(object):
    """
    Image database object
    """

    def __init__(self, name, classes=None):
        self.name = name
        self.num_classes = 0
        if not classes:
            self.classes = []
        else:
            self.classes = classes
        self.image_index = []
    
    @property
    def name(self):
        return self.name
    
    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def classes(self):
        return self.classes

    @property
    def image_index(self):
        return self.image_index