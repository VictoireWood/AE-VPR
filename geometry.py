import math
import cv2
import numpy as np
from scipy.spatial import cKDTree

class Footprints:

    def __init__(self, xy):
        self.xy = np.asarray(xy, dtype=np.float64)
        self.tree = cKDTree(self.xy)
        self.corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64)
        self.canonical = self.corners * [128.0, 96.0]
        self.area = 256.0 * 192.0

    def overlap(self, index, height, angle):
        t = math.radians(-angle)
        rotation = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        polygon = (self.corners * [height * 2048 / 2400, height * 1536 / 2400] @ rotation.T).astype(np.float32)
        area = height ** 2 * 2048 * 1536 / 1200 ** 2
        radius = math.hypot(height * 2048 / 2400, height * 1536 / 2400) + 160.0
        ids = self.tree.query_ball_point(self.xy[index], radius)
        output = np.zeros(len(self.xy), np.float32)
        for j in ids:
            other = (self.canonical + (self.xy[j] - self.xy[index])).astype(np.float32)
            intersection, _ = cv2.intersectConvexConvex(polygon, other)
            output[j] = np.clip(intersection / (area + self.area - intersection), 0, 1)
        return output
