from src.car import Car
from src.circuit import Circuit
from src.ui import Interface


if __name__ == '__main__':
    coords = [(0, 0), (0.5, 1), (0, 2), (2, 2), (3, 1), (6, 2), (6, 0)]
    circuit = Circuit(coords, width=0.3)
    car = Car(3, 0, 0.2, 0.4, 0, circuit=circuit)
    ui = Interface(circuit, car)
    ui.show()
