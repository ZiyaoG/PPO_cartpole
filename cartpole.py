"""
Minimal CartPole simulator (classic control equations).
State: [x, x_dot, theta, theta_dot]. theta = 0 means pole upright.
Action: horizontal force on the cart (Newtons), clipped to ±force_mag.
Cart friction: viscous drag ~ -cart_viscous * x_dot (N). Pole: ~ -pole_viscous * theta_dot (rad/s²).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CartPoleParams:
    gravity: float = 9.8
    mass_cart: float = 1.0
    mass_pole: float = 0.1
    pole_half_length: float = 0.5  # pivot-to-COM distance
    force_mag: float = 10.0  # max |horizontal force| applied each step
    cart_viscous: float = 0.5  # N/(m/s); ~none matches classic Gym; 1.0 makes balancing much harder
    pole_viscous: float = 0.1  # scale on joint damping in θ̈
    dt: float = 0.02
    x_threshold: float = 2.4
    theta_threshold_rad: float = 20 * math.pi / 180


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

    def step(self, action: float | np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        x, x_dot, theta, theta_dot = self.state
        force = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        force = float(np.clip(force, -self.p.force_mag, self.p.force_mag))

        mp, mc = self.p.mass_pole, self.p.mass_cart
        L = self.p.pole_half_length
        g = self.p.gravity
        dt = self.p.dt
        total = mp + mc
        pole_mass_length = mp * L

        cost = math.cos(theta)
        sint = math.sin(theta)

        friction = -self.p.cart_viscous * x_dot
        temp = (force + friction + pole_mass_length * theta_dot**2 * sint) / total
        theta_acc = (g * sint - cost * temp) / (
            L * (4.0 / 3.0 - mp * cost**2 / total)
        )
        theta_acc -= self.p.pole_viscous * theta_dot
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
