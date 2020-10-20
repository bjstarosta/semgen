# -*- coding: utf-8 -*-
"""SEMGen - Dipole machine learning labels generator class.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import logging
import numpy as np
import cv2

from generators.dipole import DipoleGenerator
import utils


class DipoleLabelsGenerator(DipoleGenerator):

    def __init__(self):
        """Init function."""
        super().__init__()

        self.data = None
        self.dipole_params = None
        self.point_size = 5

    def generate_params(self):
        return {}

    def process(self, task):
        params = self.dipole_params[task.index]
        im = np.zeros(self.dim)

        points = []
        subdiv = cv2.Subdiv2D((0, 0, self.dim[1], self.dim[0]))
        for p in params['dipoles']:
            xy = [
                int((p['offset'][0] + 1) * (self.dim[0] / 2)),
                int(((p['offset'][1] + 1) * (self.dim[1] / 2)))
            ]
            self._draw_point(im, xy)
            subdiv.insert((xy[0], xy[1]))
            points.append(xy)
        self._draw_voronoi(im, subdiv)

        task.image = utils.feature_scale(im, 0, 255, 0., 1., 'uint8')
        return task

    def _draw_point(self, im, xy):
        cv2.circle(im, (xy[0], xy[1]), self.point_size, 1, -1)

    def _draw_delaunay(self, im, subdiv):
        trilist = subdiv.getTriangleList()
        r = (0, 0, im.shape[1], im.shape[0])
        c = 1

        for t in trilist:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            if (self._in_rect(r, pt1)
            and self._in_rect(r, pt2)
            and self._in_rect(r, pt3)):
                cv2.line(im, pt1, pt2, c, 1, cv2.LINE_AA, 0)
                cv2.line(im, pt2, pt3, c, 1, cv2.LINE_AA, 0)
                cv2.line(im, pt3, pt1, c, 1, cv2.LINE_AA, 0)

    def _draw_voronoi(self, im, subdiv):
        (facets, centers) = subdiv.getVoronoiFacetList([])
        c = 1

        for i in range(0, len(facets)):
            ifacet_arr = []
            for f in facets[i]:
                ifacet_arr.append(f)

            ifacet = np.array(ifacet_arr, np.int)
            ifacets = np.array([ifacet])
            cv2.polylines(im, ifacets, True, c, 1, cv2.LINE_AA, 0)

    def _in_rect(self, rect, p):
        if (p[0] < rect[0] or p[0] > rect[2]
        or p[1] < rect[1] or p[1] > rect[3]):
            return False
        else:
            return True
