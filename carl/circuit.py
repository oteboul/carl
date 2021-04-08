import shapely.geometry as geom
from descartes import PolygonPatch
import numpy as np

class Circuit(object):

    def __init__(self, points, n_cars, width=0.3, num_checkpoints=100):
        self.n_cars = n_cars

        if isinstance(points, np.ndarray):
            points = points.tolist()

        self.points = points
        if self.points[0] != self.points[-1]:
            self.points.append(points[0])

        # Compute circuit's geometry
        self.line = geom.LineString(self.points)
        self.width = width
        self.circuit = self.line.buffer(self.width, cap_style=1)

        # For numerical stabilities when checking if something is inside the
        # circuit.
        self.dilated = self.line.buffer(self.width * 1.01, cap_style=1)

        # Where the start line is
        self.define_start()

        # Define the checkpoints
        self.make_checkpoints(n=num_checkpoints)

    def define_start(self):
        """The start line is in the middle of the longest horizontal segment."""
        last = geom.Point(*self.line.coords[0])
        self.start = last
        maxDistance = 0
        for x, y in self.line.coords[1:]:
            curr = geom.Point((x, y))
            if curr.distance(last) > maxDistance and curr.y == last.y:
                maxDistance = curr.distance(last)
                self.start = geom.Point((0.5 * (x + last.x)), 0.5 * (y + last.y))
            last = curr

        self.start_line = geom.LineString([
            (self.start.x, self.start.y - self.width),
            (self.start.x, self.start.y + self.width)
        ])

    def make_checkpoints(self, n):
        step_ext = self.circuit.exterior.length / n
        step_int = self.circuit.interiors[0].length / n
        self.checklines = []
        for i in range(n):
            self.checklines.append(geom.LineString([
                self.circuit.exterior.interpolate(step_ext * (n - i)),
                self.circuit.interiors[0].interpolate(step_int * i)],
            ))
        self.reset()

    def reset(self):
        self.checkpoints = np.zeros((self.n_cars, len(self.checklines)), dtype=np.bool)
        self.laps = np.zeros(self.n_cars, dtype=np.int32)
        self.progression = np.zeros(self.n_cars, dtype=np.float)

        self.chicken_dinner = False
        self.half_chicken_dinner = False

    def update_checkpoints(self, start, stop):
        x_start, y_start = start
        x_stop, y_stop = stop

        for k in range(self.n_cars):
            x_str, y_str = x_start[k], y_start[k]
            x_stp, y_stp = x_stop[k], y_stop[k]
            traj = geom.LineString([(x_str, y_str), (x_stp, y_stp)])

            checkpoints = self.checkpoints[k]
            if not np.all(checkpoints):
                for idx, line in enumerate(self.checklines):
                    if line.intersects(traj):
                        checkpoints[idx] = True

            if np.all(checkpoints):
                if self.start_line.intersects(traj):
                    checkpoints = np.zeros(len(self.checklines), dtype=np.bool)
                    self.laps[k] += 1
            
            self.progression[k] = np.sum(checkpoints) / len(checkpoints)

    def debug(self):
        return "laps {}: {:.0f}%".format(self.laps, self.progression * 100)

    def __contains__(self, shape):
        return self.dilated.contains(shape)

    def plot(self, ax, color='gray', skeleton=True):
        if skeleton:
            self.skeleton_patch = ax.plot(
                self.line.xy[0], self.line.xy[1],
                color='white', linewidth=3, solid_capstyle='round', zorder=3,
                linestyle='--'
            )

        self.start_line_patch = ax.plot(
            self.start_line.xy[0], self.start_line.xy[1],
            color='black', linewidth=3, linestyle='-', zorder=3
        )

        self.circuit_patch = PolygonPatch(
            self.circuit, fc=color, ec='black', alpha=0.5, zorder=2
        )
        ax.add_patch(self.circuit_patch)

        offset_x = (self.circuit.bounds[2] - self.circuit.bounds[0]) * 0.35
        offset_y = (self.circuit.bounds[3] - self.circuit.bounds[1]) * 0.2

        self.x_min, self.x_max = self.circuit.bounds[0], self.circuit.bounds[2] + offset_x
        self.y_min, self.y_max = self.circuit.bounds[1], self.circuit.bounds[3] + offset_y

        self.scoreboard_text = ax.text(self.x_max, self.y_max, 'SCOREBOARD',
            fontname='Lucida Console', fontsize=20, ha='right', va='top')

        self.texts = []
        for rank in range(self.n_cars):
            x_text_pos = self.x_max
            y_text_pos = self.y_min + 0.8 * (self.y_max - self.y_min) * (1 - rank / self.n_cars)
            text = ax.text(x_text_pos, y_text_pos, " ", fontname='Lucida Console', ha='right')
            self.texts.append(text)

        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_aspect(1)

    def update_plot(self, ax, cars):
        crashed = cars.render_locked
        names = cars.names
        prog_total = self.progression + self.laps
        ranks = np.argsort(prog_total)

        for k, car_id in enumerate(ranks):
            progress = self.progression[car_id]
            lap = self.laps[car_id]

            true_name = names[car_id][:12].capitalize()
            name_len = len(true_name)
            if name_len < 8:
                name = ' ' * (8-name_len) + true_name
            else:
                name = true_name
            
            rank = self.n_cars - k

            if rank == 1:
                rank_text = '1er'
            elif rank == 2:
                rank_text = '2nd'
            elif rank == 3:
                rank_text = '3rd'
            else:
                rank_text = f'{rank}e'

            if len(rank_text) < 3:
                rank_text += ' '

            if len(ranks) == 1:
                rank_text = ''

            text = f'{rank_text} {name} - Lap {lap+1} - {progress:2.0%} - {car_id}'
            text_patch = self.texts[rank - 1]
            text_patch.set_visible(True)
            text_patch.set_text(text)
            text_patch.set_color(cars.colors[car_id])
            bbox = dict(facecolor='none', edgecolor='none')

            if lap > 1:
                bbox = dict(facecolor=(1, 1, 0, 0.3), edgecolor='none', boxstyle='round,pad=1')
                if not self.half_chicken_dinner:
                    ax.set_title(f'{true_name} is on fire !',
                                 fontname='Lucida Console',
                                 fontsize=32)
                    self.half_chicken_dinner = True
            if lap >= 2:
                bbox = dict(facecolor=(0, 1, 0, 0.3), edgecolor='none', boxstyle='round,pad=1')
                if not self.chicken_dinner:
                    ax.set_title(f'A winner is {true_name} ({car_id})!',
                                 fontname='Lucida Console',
                                 fontsize=32)
                    self.chicken_dinner = True
            if crashed[car_id]:
                bbox = dict(facecolor=(1, 0, 0, 0.3), edgecolor='none', boxstyle='round,pad=1')

            if bbox is not None:
                text_patch.set_bbox(bbox)

        if not self.chicken_dinner:
            ax.set_title(f'Let the best AI win !',
                            fontname='Lucida Console',
                            fontsize=32)

    def remove_plot(self, ax):
        self.start_line_patch[0].remove()
        if hasattr(self, 'skeleton_patch'):
            self.skeleton_patch[0].remove()
        self.circuit_patch.remove()
        self.scoreboard_text.remove()
        for text in self.texts:
            text.remove()
