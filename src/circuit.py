import shapely.geometry as geom
from descartes import PolygonPatch


class Circuit(object):

    def __init__(self, points, width):
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

        self.num_checkpoints = 30

    def __contains__(self, shape):
        return self.dilated.contains(shape)

    def plot(self, ax, color='gray', skeleton=True):
        if skeleton:
            ax.plot(
                self.line.xy[0], self.line.xy[1],
                color='white', linewidth=3, solid_capstyle='round', zorder=3,
                linestyle='--')

        patch = PolygonPatch(
            self.circuit, fc=color, ec='black', alpha=0.5, zorder=2)
        ax.add_patch(patch)

        bounds = self.circuit.bounds
        offset_x = (bounds[2] - bounds[0]) * 0.1
        offset_y = (bounds[3] - bounds[1]) * 0.1
        ax.set_xlim(bounds[0] - offset_x, bounds[2] + offset_x)
        ax.set_ylim(bounds[1] - offset_y, bounds[3] + offset_y)
        ax.set_aspect(1)
