import numpy as np
import shapely.geometry as geom
from descartes import PolygonPatch


class Car(object):

    ANGLE_UNIT = np.pi / 16
    SPEED_UNIT = 0.05

    def __init__(self, circuit, num_sensors=5):
        self.circuit = circuit
        self.reset()
        self.w = circuit.width * 2 / 3
        self.h = 2 * self.w
        self.computeBox()

        self.patch = None
        self.sensor_lines = None
        self.num_sensors = num_sensors
        self.color = '#44dafb'

    def reset(self):
        self.x = self.circuit.start.x
        self.y = self.circuit.start.y
        self.theta = 0.0
        self.speed = 0.0

    def action(self, speed=0, theta=0):
        """Change the speed of the car and / or its direction.
        Both can be negative."""
        self.speed = max(0.0, self.speed + speed * self.SPEED_UNIT)
        self.theta += theta * self.ANGLE_UNIT
        self.move()

    def in_circuit(self) -> bool:
        """returns True if the car is entirely in its circuit, False otherwise.
        """
        return self.car in self.circuit

    def move(self):
        """Based on the current speed and position of the car, make it move."""
        trajectory = [(self.x, self.y)]
        self.x += self.speed * np.cos(self.theta)
        self.y += self.speed * np.sin(self.theta)
        self.computeBox()
        trajectory.append((self.x, self.y))
        self.circuit.updateCheckpoints(geom.LineString(trajectory))

    def getCoords(self, i, j):
        """From car coordinates to world coordinates, (0, 0) being the center of
        the car. And considering the car is a square [-1, +1]^2"""
        a = i * self.h / 2
        b = j * self.w / 2
        cos = np.cos(self.theta)
        sin = np.sin(self.theta)
        return a * cos - b * sin + self.x, a * sin + b * cos + self.y

    def computeBox(self):
        points = []
        for i, j in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
            points.append(self.getCoords(i, j))
        self.car = geom.Polygon(points)

    def intersection(self, phi):
        """Computes the distance between the front of the car and
        the border of the circuit in the direction phi."""
        # Builds the half line
        origin = self.getCoords(1, 0)
        line = geom.LineString(
            [origin, self.getCoords(1000 * np.cos(phi), 1000 * np.sin(phi))])

        # Compute intersection with circuit that lies inside the circuit
        origin = geom.Point(origin)
        try:
            p = line.intersection(self.circuit.circuit)
            end = p if isinstance(p, geom.LineString) else p[0]
            end = end.boundary[1]
        except Exception as e:
            return line.boundary[1]

        seg = geom.LineString([origin, end])
        if seg not in self.circuit:
            return origin

        return end

    def angles(self):
        result = []
        for i in range(self.num_sensors):
            result.append(-np.pi / 2 + i * np.pi / (self.num_sensors - 1))
        return result

    def getTitle(self):
        return "Speed: {:.2f}".format(self.speed)

    def distances(self):
        result = []
        origin = geom.Point(self.getCoords(1, 0))
        for phi in self.angles():
            point = self.intersection(phi)
            result.append(origin.distance(point))
        return result

    def update_plot(self, ax):
        # Plot the car
        other = PolygonPatch(
            self.car, fc=self.color, ec='black', alpha=1.0, zorder=4)
        if self.patch is None:
            self.patch = other
        else:
            self.patch._path._vertices = other._path._vertices
            self.patch.set_fc(self.color)

        # Plot the distances to the circuit
        sensor_lines = []
        origin = self.getCoords(1, 0)
        for phi in self.angles():
            p = self.intersection(phi)
            sensor_lines.append((
                [origin[0], p.xy[0][0]], [origin[1], p.xy[1][0]]))

        if self.sensor_lines is None:
            self.sensor_lines = []
            for curr_x, curr_y in sensor_lines:
                line = ax.plot(
                    curr_x, curr_y, color='#df5a65', linestyle=':', lw=2,
                    zorder=5)
                self.sensor_lines.append(line[0])
        else:
            for k, (curr_x, curr_y) in enumerate(sensor_lines):
                line = self.sensor_lines[k]
                line.set_xdata(curr_x)
                line.set_ydata(curr_y)

    def plot(self, ax):
        self.update_plot(ax)
        ax.add_patch(self.patch)
