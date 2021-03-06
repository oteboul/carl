"""
To run the UI mode, from the carl/ directory run

python3 -m scripts.run_ui
"""

from carl.car import Car
from carl.circuit import Circuit
from carl.ui import Interface


if __name__ == '__main__':
    coords = [(0, 0), (0.5, 1), (0, 2), (2, 2), (3, 1), (6, 2), (6, 0)]
    circuit = Circuit(coords, width=0.3)
    car = Car(circuit)
    ui = Interface(circuit, car)
    ui.show()
