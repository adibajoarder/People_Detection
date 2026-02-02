import cv2
import numpy as np

class Heatmap:
    def __init__(self, h, w, decay=0.96, intensity=25, radius=25):
        self.h = h
        self.w = w
        self.decay = decay
        self.intensity = intensity
        self.radius = radius
        self.map = np.zeros((h, w), dtype=np.float32)

    def update(self, points):
        self.map *= self.decay
        for (x, y) in points:
            cv2.circle(self.map, (int(x), int(y)), self.radius, float(self.intensity), -1)

    def render_box(self, box_w=240, box_h=240):
        norm = cv2.normalize(self.map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        small = cv2.resize(colored, (box_w, box_h))
        return small
