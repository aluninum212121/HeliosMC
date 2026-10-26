"""
БЛОК 1 — НЕЙТРОН.

Класс Neutron — представление одного нейтрона (одиночный трек).

Примечание: движок Монте-Карло (physics.py) работает с массивами NumPy
и напрямую этот класс не использует; ветка оставлена как модель отдельной
частицы (соответствует БЛОКУ 1 исходного монолита).
"""

import numpy as np


class Neutron:
    def __init__(self, position, direction, energy=1.0):
        self.position  = np.array(position,  dtype=float)
        self.direction = np.array(direction, dtype=float)
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction /= norm
        self.energy = energy
        self.alive  = True
        self.path   = [self.position.copy()]   # история позиций

    def move(self, distance):
        self.position += self.direction * distance
        self.path.append(self.position.copy())

    def scatter(self, mu, phi):
        sin_theta = np.sqrt(max(0.0, 1.0 - mu**2))
        u, v, w = self.direction
        if abs(w) < 0.9999:
            denom = np.sqrt(1.0 - w**2)
            new_dir = np.array([
                u*mu + sin_theta*(u*w*np.cos(phi) - v*np.sin(phi)) / denom,
                v*mu + sin_theta*(v*w*np.cos(phi) + u*np.sin(phi)) / denom,
                w*mu - sin_theta*denom*np.cos(phi)
            ])
        else:
            new_dir = np.array([
                sin_theta * np.cos(phi),
                sin_theta * np.sin(phi),
                mu * np.sign(w)
            ])
        norm = np.linalg.norm(new_dir)
        if norm > 0:
            self.direction = new_dir / norm
