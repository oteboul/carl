import imageio
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class Interface(object):
    SPEED = {'up': 1, 'down': -1}
    ANGLE = {'left': 1, 'right': -1}

    def __init__(self, circuit, car, save_frames=False):
        self.circuit = circuit
        self.car = car

        self.fig = plt.figure(1, figsize=(12, 6), dpi=90)
        self.ax = self.fig.add_subplot(111)

        self.circuit.plot(self.ax)
        self.car.plot(self.ax)
        self.fig.canvas.mpl_connect('key_press_event', self.onpress)

        self.save_frames = save_frames
        self.frames = []

    def addFrame(self):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf)
        self.frames.append(np.array(
            im.getdata(), np.uint8).reshape(im.size[1], im.size[0], 4))
        buf.close()

    def toGIF(self, filename):
        writer = imageio.get_writer(filename, fps=10)

        for frame in self.frames:
            writer.append_data(frame)
        writer.close()

    def show(self, block=True):
        plt.ion()
        if block:
            plt.show(block=block)
        else:
            plt.pause(0.001)
            plt.show()
            plt.close()

    def setTitle(self, title):
        self.ax.set_title(title)

    def update(self):
        self.car.update_plot(self.ax)
        self.fig.canvas.draw()
        if self.save_frames:
            self.addFrame()

    def onpress(self, event):
        speed = self.SPEED.get(event.key, 0)
        theta = self.ANGLE.get(event.key, 0)
        self.car.action(speed, theta)
        self.update()
        self.setTitle(self.car.getTitle())
