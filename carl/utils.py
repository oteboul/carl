import colorsys
import numpy as np

def make_color(h):
    return "#{:02x}{:02x}{:02x}".format(
        *map(lambda x: int(255*x), colorsys.hsv_to_rgb(h, 0.8, 0.8)))

def generate_circuit(n_points=20, difficulty=0, circuit_size=(5, 3)):
    rx, ry = circuit_size
    angles = np.linspace(0.05*2*np.pi, 0.95*2*np.pi, num=n_points) - np.pi/2
    points = [(rx*np.cos(angle), ry*np.sin(angle)) for angle in angles]
    for i, (point, angle) in enumerate(zip(points, angles)):
        if i not in (0, n_points-1) and difficulty > 0:
            rd_dist = min(circuit_size)/np.pi * np.random.vonmises(mu=0, kappa=32/difficulty)
            px, py = point
            points[i] = (px + rd_dist * np.cos(angle), py + rd_dist * np.sin(angle))
    return points
