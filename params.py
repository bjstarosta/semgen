# -*- coding: utf-8 -*-
"""SEMGen - Generator parameter handler.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import json
import click
import semgen


class ParamFile(object):
    """Generated image param file handling object.

    Param files allow for reproducibility of generated datasets and can be
    used to recreate generator conditions.

    Args:
        gen_id (str): Generator ID string to save to the param file.
        filename (str): Filename of the param file.

    Attributes:
        data (dict): Dictionary representing the data hierarchy in the
            param file.
        filename

    """

    def __init__(self, gen_id, filename='params.json'):
        self.filename = filename
        self.data = {
            'generator': gen_id,
            'global': {},
            'params': []
        }
        self._i = 0

    def __iter__(self):
        while self._i < len(self.data['params']):
            yield self.data['params'][self._i]
            self._i += 1

    def __next__(self):
        if self._i < len(self.data['params']):
            ret = self.data['params'][self._i]
            self._i += 1
            return ret
        else:
            return

    def append(self, d, filename):
        """Append an object to the 'params' key.

        Method will also write a 'image_file' key in the passed dictionary
        containing the filename argument.

        Args:
            d (dict): Dictionary to append.
            filename (str): Filename of the corresponding image.

        """
        d['image_file'] = filename
        self.data['params'].append(d)

    def get(self):
        """Return the 'params' key.

        Returns:
            list: List of dicts containing individual image processing
                parameters.

        """
        return self.data['params']

    def pack_obj(self, obj):
        """Pack object properties into 'global' key.

        Args:
            obj (object): Object to populate.

        """
        d = {}
        g = obj.get_globals()
        for k in g.keys():
            d[k] = g[k]
        self.data['global'] = d

    def unpack_obj(self, obj):
        """Unpack variables from 'global' key into passed object as properties.

        Args:
            obj (object): Object to populate.

        """
        for k in self.data['global']:
            obj.__dict__[k] = self.data['global'][k]

    def read(self, path):
        """Read a simulation parameter file and transform it to dict format.

        Args:
            path (str): Directory to find the parameter file in.

        """
        with click.open_file(path, 'r') as f:
            self.data = json.load(f)

    def save(self, path):
        """Save a dictionary of simulation parameters to a JSON file.

        Args:
            path (str): Directory to save the parameter file in.

        """
        self.data['params'] = sorted(self.data['params'],
            key=lambda k: k['image_file'])

        self.data['_version'] = semgen.VERSION
        with click.open_file(os.path.join(path, self.filename), 'w') as f:
            json.dump(self.data, f, sort_keys=True, indent=4)
