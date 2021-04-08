import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np

class Interface(object):
    SPEED = {'up': 1, 'down': -1}
    ANGLE = {'left': 1, 'right': -1}

    def __init__(self):
        plt.close()
        self.fig = plt.figure(1, figsize=(15, 6), dpi=90)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        pos = self.ax.get_position()
        box = Bbox(pos)
        self.ax.set_position(box)

    def show(self, block=True):
        plt.ion()
        plt.show(block=block)

    def plot(self, cars, circuit):
        circuit.plot(self.ax)
        cars.plot(self.ax)

    def update(self, cars, circuit):
        cars.update_plot(self.ax)
        circuit.update_plot(self.ax, cars)
        self.fig.canvas.draw()
        plt.pause(1/90)

    def close(self):
        plt.close()
