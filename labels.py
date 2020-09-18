# -*- coding: utf-8 -*-
"""SEMGen - Labelling handler.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import csv


class LabelFile(object):
    """Generated image label file handling object.

    Args:
        filename (str): Name and extension of the label file.

    Attributes:
        columns (list): List of data columns to be used in the label file.
        labels (list): List of rows containing data corresponding to the
            columns.
        filename

    """

    def __init__(self, filename='labels.csv'):
        self.filename = filename
        self.columns = []
        self.files = []
        self.labels = []

    @property
    def columns(self):
        """List of data columns to be used in the label file.

        Returns:
            list

        """
        return self.__c

    @columns.setter
    def columns(self, lst):
        if not isinstance(lst, list):
            raise ValueError("Columns must be set as a list.")

        self.__c = ['img'] + lst

    def empty(self):
        """Deallocate the lists containing all label data within the object.

        Returns:
            None

        """
        self.columns = []
        self.files = []
        self.labels = []

    def add_file(self, image_file):
        """Add new item to the filename list.

        Args:
            image_file (str): File name of the corresponding image file.

        Returns:
            None

        """
        self.files.append(image_file)

    def add_label(self, label_data):
        """Add new item to the label list.

        Args:
            label_data (list): List of label parameters.

        Returns:
            None

        """
        if not isinstance(label_data, list):
            raise ValueError("Columns must be set as a list.")
        self.labels.append(label_data)

    def read(self, path):
        """Read an existing label file and populate the object with its data.

        Args:
            path (str): Path to folder containing the label file.

        Returns:
            bool: True if file was found and contents were read,
                False otherwise.

        """
        self.empty()
        fpath = os.path.join(path, self.filename)

        if not os.path.exists(fpath) or not os.path.isfile(fpath):
            return False

        with open(fpath, 'r') as f:
            r = csv.reader(f, delimiter=',')
            line = 0
            for row in r:
                if line == 0:
                    self.columns = row[1:]
                else:
                    self.files.append(row[0])
                    self.labels.append(row[1:])
                line += 1

        return True

    def save(self, path):
        """Save object data as label file.

        Args:
            path (str): Path to folder containing the label file.

        Returns:
            None

        """
        fpath = os.path.join(path, self.filename)
        with open(fpath, mode='w') as f:
            w = csv.writer(f,
                delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL)
            w.writerow(self.columns)
            for i, row in enumerate(self.labels):
                w.writerow([self.files[i]] + row)
