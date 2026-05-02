"""
Minimal CartPole simulator (classic control equations, Gym-compatible constants).
State: [x, x_dot, theta, theta_dot]. theta = 0 means pole upright.
Actions: 0 = push left, 1 = push right (same convention as Gym).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

Action = Literal[0, 1]


@dataclass
class CartPoleParams:
    gravity: float = 9.8
    mass_cart: float = 1.0
    mass_pole: float = 0.1
    pole_half_length: float = 0.5  # Gym calls this `length`; it's pivot-to-COM distance
    force_mag: float = 10.0
    dt: float = 0.02
    x_threshold: float = 2.4
    theta_threshold_rad: float = 12 * math.pi / 180


class CartPoleEnv:
    def __init__(self, params: CartPoleParams | None = None, seed: int | None = None):
        self.p = params or CartPoleParams()
        self.rng = np.random.default_rng(seed)
        self.state: np.ndarray = np.zeros(4, dtype=np.float64)

    def reset(self) -> np.ndarray:
        """Small random start near upright (same spirit as Gym)."""
        self.state = self.rng.uniform(low=-0.05, high=0.05, size=4).astype(np.float64)
        return self.state.copy()

    def _terminated(self) -> bool:
        x, _, theta, _ = self.state
        return bool(
            abs(x) > self.p.x_threshold or abs(theta) > self.p.theta_threshold_rad
        )

    def step(self, action: Action) -> tuple[np.ndarray, float, bool, dict]:
        x, x_dot, theta, theta_dot = self.state
        force = self.p.force_mag if action == 1 else -self.p.force_mag

        mp, mc = self.p.mass_pole, self.p.mass_cart
        L = self.p.pole_half_length
        g = self.p.gravity
        dt = self.p.dt
        total = mp + mc
        pole_mass_length = mp * L

        cost = math.cos(theta)
        sint = math.sin(theta)

        temp = (force + pole_mass_length * theta_dot**2 * sint) / total
        theta_acc = (g * sint - cost * temp) / (
            L * (4.0 / 3.0 - mp * cost**2 / total)
        )
        x_acc = temp - pole_mass_length * theta_acc * cost / total

        x += dt * x_dot
        x_dot += dt * x_acc
        theta += dt * theta_dot
        theta_dot += dt * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)

        terminated = self._terminated()
        # Match Gym: +1 each timestep (including the step that ends the episode).
        reward = 1.0
        return self.state.copy(), reward, terminated, {}
