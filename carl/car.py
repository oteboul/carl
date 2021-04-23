import numpy as np
import shapely.geometry as geom
from descartes import PolygonPatch
import matplotlib.patheffects as PathEffects
from copy import deepcopy

from carl.utils import make_color

class Cars(object):

    def __init__(self, circuit, n_cars=1, names=None, colors=None,
        n_sensors=5, render_sensors=True, fov=np.pi,
        car_width=0.2, angle_unit=np.pi/16, speed_unit=0.02):

        self.ANGLE_UNIT = angle_unit
        self.SPEED_UNIT = speed_unit

        self.n_cars = n_cars
        self.num_sensors = n_sensors

        ones = np.ones(self.n_cars)
        self.anchors = (
            (-ones, -ones),
            (-ones, ones),
            (ones, ones),
            (ones, -ones),
        )

        start = circuit.start.x, circuit.start.y
        self.reset(start)
        self.w = car_width
        self.h = 2 * self.w
        self.compute_box()
        self.reset_render()

        if colors is None:
            self.colors = np.array([
                make_color(np.random.random())
                for _ in range(self.n_cars)
            ])
        else:
            self.colors = colors

        if names is None:
            self.names = np.array([
                f'Car n°{i+1}'
                for i in range(self.n_cars)
            ])
        elif isinstance(names, str):
            self.names = np.array([names for i in range(self.n_cars)])
        else:
            self.names = names

        self.render_sensors = render_sensors
        self.angles = [
            -fov / 2 + i * fov / (self.num_sensors - 1)
            for i in range(self.num_sensors)
        ]

    def reset(self, start_coords):
        self.xs = np.array([start_coords[0] for _ in range(self.n_cars)])
        self.ys = np.array([start_coords[1] for _ in range(self.n_cars)])
        self.thetas = np.array([0. for _ in range(self.n_cars)])
        self.speeds = np.array([0. for _ in range(self.n_cars)])
        self.in_circuit = np.ones(self.n_cars, dtype=np.bool)
        self.render_locked = np.zeros(self.n_cars, np.bool)
        self.sensor_lines_data = np.zeros((self.n_cars, self.num_sensors, 2, 2))
        self.time = 0
    
    def reset_render(self, ax=None):
        if hasattr(self, 'patch'):
            for patch in getattr(self, 'patch'):
                patch.set_alpha(0)
        if hasattr(self, 'sensor_lines'):
            for sensor_lines in getattr(self, 'sensor_lines'):
                if sensor_lines is not None:
                    for line in sensor_lines:
                        line.set_alpha(0)
        else:
            self.hover_text = [None for _ in range(self.n_cars)]
            self.patch = [None for _ in range(self.n_cars)]
            self.sensor_lines = [None for _ in range(self.n_cars)]

    def action(self, actions, circuit):
        """Change the speed of the car and / or its direction.
        Both can be negative."""
        speeds = np.clip(actions[:, 0], -1, 1)
        thetas = 2 * np.clip(actions[:, 1], -1 , 1)

        self.speeds = np.minimum(self.h/2, np.maximum(0.0, self.speeds + speeds * self.SPEED_UNIT))
        self.thetas += self.in_circuit * thetas * self.ANGLE_UNIT

        start = deepcopy((self.xs, self.ys))
        self.move()
        stop = (self.xs, self.ys)
        circuit.update_checkpoints(start, stop)

        for car_id, car in enumerate(self.cars):
            self.in_circuit[car_id] = car in circuit

    def move(self):
        """Based on the current speed and position of the car, make it move."""
        self.xs += self.in_circuit * self.speeds * np.cos(self.thetas)
        self.ys += self.in_circuit * self.speeds * np.sin(self.thetas)
        self.compute_box()

    def coords(self, i, j):
        """From car coordinates to world coordinates, (0, 0) being the center of
        the car. And considering the car is a square [-1, +1]^2"""
        a = i * self.h / 2
        b = j * self.w / 2
        cos = np.cos(self.thetas)
        sin = np.sin(self.thetas)
        return np.stack((a * cos - b * sin + self.xs, a * sin + b * cos + self.ys), axis=-1)

    def compute_box(self):
        points = np.stack([self.coords(i, j) for i, j in self.anchors], axis=-2)
        self.cars = [geom.Polygon(points[k]) for k in range(self.n_cars)]

    def intersection(self, i, phi, circuit):
        """Computes the intersection coords between the front of the car and
        the border of the circuit in the direction phi."""
        intersections = []
        origins = self.coords(1, 0)
        half_line_coords = self.coords(1000 * np.cos(phi), 1000 * np.sin(phi))
        for org, half in zip(origins, half_line_coords):
            # Builds the half line
            line = geom.LineString([org, half])

            # Compute intersection with circuit that lies inside the circuit
            origin = geom.Point(org)
            try:
                p = line.intersection(circuit.circuit)
                end = p if isinstance(p, geom.LineString) else p[0]
                end = end.boundary[1]
                seg = geom.LineString([origin, end])
                if seg not in circuit:
                    intersections.append(origin.xy)
                else:
                    intersections.append(end.xy)
            except Exception as e:
                intersections.append(line.boundary[1].xy)
        intersections = np.array(intersections)[..., 0]
        self.sensor_lines_data[:, i] = np.stack((origins, intersections), axis=-1)
        return intersections

    def get_distances(self, circuit):
        distances = []
        origin = np.array([1, 0])
        for i, phi in enumerate(self.angles):
            intersections = self.intersection(i, phi, circuit)
            distance = np.sqrt(np.sum(np.square(intersections - origin), axis=-1))
            distances.append(distance)
        return np.stack(distances, axis=-1)

    def update_plot(self, ax):
        # Plot the car
        for car_id, car in enumerate(self.cars):
            if not self.render_locked[car_id]:
                other = PolygonPatch(car, fc=self.colors[car_id], ec='black', alpha=1.0, zorder=4)
                if self.patch[car_id] is None:
                    self.patch[car_id] = other
                else:
                    self.patch[car_id]._path._vertices = other._path._vertices
                    self.patch[car_id].set_facecolor(self.colors[car_id])

                car_x, cay_y = car.centroid.x, car.centroid.y,
                if self.hover_text[car_id] is None:
                    self.hover_text[car_id] = ax.text(
                        car_x, cay_y,
                        self.names[car_id],
                        color=self.colors[car_id],
                        fontname='Lucida Console',
                        ha='center',
                        zorder=10
                    )
                    self.hover_text[car_id].set_path_effects(
                        [PathEffects.withStroke(linewidth=3, foreground='w')]
                    )
                else:
                    self.hover_text[car_id].set_position((car_x, cay_y))

                if not self.in_circuit[car_id]:
                    self.patch[car_id].set_alpha(0.1)
                    self.hover_text[car_id].set_alpha(0.1)
                elif self.time == 0:
                    self.patch[car_id].set_alpha(1.)
                    self.hover_text[car_id].set_alpha(1.)

        if self.render_sensors:
            self._plot_sensors(ax)

        self.render_locked = np.logical_not(self.in_circuit)
        self.time += 1
    
    def _plot_sensors(self, ax):
        for car_id in range(self.n_cars):
            if not self.render_locked[car_id]:
                sensor_lines = self.sensor_lines_data[car_id]
                if self.sensor_lines[car_id] is None:
                    self.sensor_lines[car_id] = []
                    for curr_x, curr_y in sensor_lines:
                        line = ax.plot(
                            curr_x, curr_y, color='#df5a65', linestyle=':', lw=2,
                            zorder=5)
                        self.sensor_lines[car_id].append(line[0])
                else:
                    for k, (curr_x, curr_y) in enumerate(sensor_lines):
                        line = self.sensor_lines[car_id][k]
                        line.set_alpha(1)
                        line.set_xdata(curr_x)
                        line.set_ydata(curr_y)

    def plot(self, ax):
        self.update_plot(ax)
        for k in range(self.n_cars):
            ax.add_patch(self.patch[k])

    @property
    def crashed(self):
        return np.logical_or(np.logical_not(self.in_circuit), self.speeds <= self.h / 20)
