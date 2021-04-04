import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np

class Interface(object):
    SPEED = {'up': 1, 'down': -1}
    ANGLE = {'left': 1, 'right': -1}

    def __init__(self, circuit, cars):
        plt.close()
        self.circuit = circuit
        self.cars = cars

        self.fig = plt.figure(1, figsize=(15, 6), dpi=90)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        pos = self.ax.get_position()
        box = Bbox(pos)
        self.ax.set_position(box)

        self.circuit.plot(self.ax)
        self.cars.plot(self.ax)

    def show(self, block=True):
        plt.ion()
        plt.show(block=block)

    def update(self):
        self.cars.update_plot(self.ax)
        self.circuit.update_plot(self.ax, self.cars)
        self.fig.canvas.draw()
        plt.pause(1/120)

    def close(self):
        plt.close()
