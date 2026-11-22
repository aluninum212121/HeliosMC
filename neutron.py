"""Standalone particle and trajectory model.

The batch transport engine in physics.py operates on NumPy arrays.
"""

import numpy as np


class Neutron:
    def __init__(self, position, direction, energy=1.0):
        self.position  = np.array(position,  dtype=float)
        self.direction = np.array(direction, dtype=float)
        if self.position.shape != (3,) or self.direction.shape != (3,):
            raise ValueError("position and direction must have shape (3,)")
        if not np.isfinite(self.position).all() or not np.isfinite(self.direction).all():
            raise ValueError("position and direction must be finite")
        if not np.isfinite(energy) or energy < 0:
            raise ValueError("energy must be finite and nonnegative")
        norm = np.linalg.norm(self.direction)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("direction must have nonzero finite norm")
        if norm > 0:
            self.direction /= norm
        self.energy = energy
        self.alive  = True
        self.path   = [self.position.copy()]

    def move(self, distance):
        if not np.isfinite(distance) or distance < 0:
            raise ValueError("distance must be finite and nonnegative")
        self.position += self.direction * distance
        self.path.append(self.position.copy())

    def scatter(self, mu, phi):
        if not np.isfinite(mu) or not -1 <= mu <= 1 or not np.isfinite(phi):
            raise ValueError("mu must be in [-1, 1] and phi must be finite")
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
