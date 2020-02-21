import matplotlib.pyplot as plt


class Interface(object):
    SPEED = {'up': 1, 'down': -1}
    ANGLE = {'left': 1, 'right': -1}

    def __init__(self, circuit, car):
        plt.close()
        self.circuit = circuit
        self.car = car

        self.fig = plt.figure(1, figsize=(12, 6), dpi=90)
        self.ax = self.fig.add_subplot(111)

        self.circuit.plot(self.ax)
        self.car.plot(self.ax)
        self.fig.canvas.mpl_connect('key_press_event', self.onpress)

    def show(self, block=True):
        plt.ion()
        plt.show(block=block)

    def setTitle(self, title):
        self.ax.set_title(title)

    def update(self):
        self.car.update_plot(self.ax)
        self.fig.canvas.draw()
        plt.pause(0.001)

    def onpress(self, event):
        speed = self.SPEED.get(event.key, 0)
        theta = self.ANGLE.get(event.key, 0)
        self.car.action(speed, theta)
        self.update()
        self.setTitle(self.car.getTitle())
    
    def close(self):
        plt.close()
