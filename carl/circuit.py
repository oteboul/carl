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

        # Reset display
        self.reset_render()

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

    def reset_render(self):
        if not hasattr(self, 'texts'):
            self.texts = [None for _ in range(self.n_cars)]
        else:
            for text in self.texts:
                text.set_text('')
                text.set_color('black')

                bbox = dict(facecolor='none', edgecolor='none', boxstyle='round,pad=1')
                text.set_bbox(bbox)

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
            ax.plot(
                self.line.xy[0], self.line.xy[1],
                color='white', linewidth=3, solid_capstyle='round', zorder=3,
                linestyle='--'
            )

        ax.plot(
            self.start_line.xy[0], self.start_line.xy[1],
            color='black', linewidth=3, linestyle='-', zorder=3
        )

        patch = PolygonPatch(
            self.circuit, fc=color, ec='black', alpha=0.5, zorder=2
        )
        ax.add_patch(patch)

        offset_x = (self.circuit.bounds[2] - self.circuit.bounds[0]) * 0.35
        offset_y = (self.circuit.bounds[3] - self.circuit.bounds[1]) * 0.2

        self.x_min, self.x_max = self.circuit.bounds[0], self.circuit.bounds[2] + offset_x
        self.y_min, self.y_max = self.circuit.bounds[1], self.circuit.bounds[3] + offset_y
        ax.text(self.x_max, self.y_max, 'SCOREBOARD',
            fontname='Lucida Console', fontsize=20, ha='right', va='top')
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
            x_text_pos = self.x_max
            y_text_pos = self.y_min + self.circuit.bounds[3] *  (1 - (rank-1)/self.n_cars)

            if self.texts[car_id] is None:
                self.texts[car_id] = ax.text(x_text_pos, y_text_pos, text,
                    fontname='Lucida Console', ha='right')
            else:
                self.texts[car_id].set_text(text)
                if lap < 2:
                    self.texts[car_id].set_position((x_text_pos, y_text_pos))
                    self.texts[car_id].set_color(cars.colors[car_id])
            if lap > 1:
                bbox = dict(facecolor=(1, 1, 0, 0.3), edgecolor='none', boxstyle='round,pad=1')
                self.texts[car_id].set_bbox(bbox)
                if not self.half_chicken_dinner:
                    ax.set_title(f'{true_name} is on fire !',
                                 fontname='Lucida Console',
                                 fontsize=32)
                    self.half_chicken_dinner = True
            if lap >= 2:
                bbox = dict(facecolor=(0, 1, 0, 0.3), edgecolor='none', boxstyle='round,pad=1')
                self.texts[car_id].set_bbox(bbox)
                if not self.chicken_dinner:
                    ax.set_title(f'A winner is {true_name} ({car_id})!',
                                 fontname='Lucida Console',
                                 fontsize=32)
                    self.chicken_dinner = True
            if crashed[car_id]:
                bbox = dict(facecolor=(1, 0, 0, 0.3), edgecolor='none', boxstyle='round,pad=1')
                self.texts[car_id].set_bbox(bbox)
        if not self.chicken_dinner:
            ax.set_title(f'Let the best AI win !',
                            fontname='Lucida Console',
                            fontsize=32)

def generate_circuit(n_points=16, difficulty=0, circuit_size=(5, 2)):
    n_points = min(25, n_points)
    angles = [-np.pi/4 + 2*np.pi*k/n_points for k in range(3*n_points//4)]
    points = [(circuit_size[0]/2, 0.5), (3*circuit_size[0]/2, 0.5)]
    points += [(circuit_size[0]*(1+np.cos(angle)), circuit_size[1]*(1+np.sin(angle))) for angle in angles]
    for i, angle in zip(range(n_points), angles):
        rd_dist = 0
        if difficulty > 0:
            rd_dist = min(circuit_size) * np.random.vonmises(mu=0, kappa=32/difficulty)/np.pi
        points[i+2] = tuple(np.array(points[i+2]) + rd_dist*np.array([np.cos(angle), np.sin(angle)]))
    return np.array(points) - np.array([5, .6])
