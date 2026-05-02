"""
Plot / animate rollouts so you can sanity-check a policy before training PPO.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from cartpole import CartPoleEnv

Obs = np.ndarray
PolicyFn = Callable[[Obs], float]


def run_episode(
    env: CartPoleEnv,
    policy: PolicyFn,
    max_steps: int = 500,
    spin_down_steps: int = 0,
):
    """
    Collect a trajectory. After the MDP terminates (or max_steps), optionally keep
    integrating with force=0 so animations can show the pole coast/spin without control.
    """
    obs = env.reset()
    xs, thetas, forces, rewards = [], [], [], []
    total = 0.0
    controlled_steps = 0

    for _ in range(max_steps):
        xs.append(obs[0])
        thetas.append(obs[2])
        f = float(policy(obs))
        f = float(np.clip(f, -env.p.force_mag, env.p.force_mag))
        forces.append(f)
        obs, r, done, _ = env.step(f)
        rewards.append(r)
        total += r
        controlled_steps += 1
        if done:
            break

    for _ in range(spin_down_steps):
        xs.append(obs[0])
        thetas.append(obs[2])
        forces.append(0.0)
        obs, _, _, _ = env.step(0.0)
        rewards.append(0.0)

    return {
        "x": np.array(xs),
        "theta": np.array(thetas),
        "force": np.array(forces),
        "reward_per_step": np.array(rewards),
        "total_reward": total,
        "steps": controlled_steps,
    }


def plot_rollout(history: dict, title: str = "Rollout") -> None:
    t = np.arange(len(history["theta"]))
    fig, axes = plt.subplots(4, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(t, history["theta"])
    axes[0].set_ylabel("theta (rad)")
    axes[1].plot(t, history["x"])
    axes[1].set_ylabel("cart x")
    axes[2].plot(t, history["force"])
    axes[2].set_ylabel("force (N)")
    axes[3].step(t, history["reward_per_step"], where="post")
    axes[3].set_ylabel("reward")
    axes[3].set_xlabel("step")
    fig.suptitle(f"{title} — return={history['total_reward']:.0f}, steps={history['steps']}")
    plt.tight_layout()
    plt.show()


def animate_rollout(history: dict, interval_ms: int = 30) -> None:
    """Crude side-view stick figure; good enough to see failures."""
    fig, ax = plt.subplots(figsize=(6, 3))
    cart_w = 0.3
    x_abs_max = float(np.max(np.abs(history["x"]))) + cart_w / 2 + 0.2
    lim = max(2.6, x_abs_max)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=1)

    cart_h = 0.15
    L = 0.5  # match CartPoleParams default pole_half_length for drawing

    cart_rect = plt.Rectangle((0, 0), cart_w, cart_h, fc="tab:blue")
    pole_line, = ax.plot([0, 0], [cart_h, cart_h + L], lw=3, color="tab:orange")

    def init():
        ax.add_patch(cart_rect)
        return cart_rect, pole_line

    def update(frame):
        x = history["x"][frame]
        theta = history["theta"][frame]
        cart_rect.set_xy((x - cart_w / 2, 0))
        px = x + L * np.sin(theta)
        py = cart_h + L * np.cos(theta)
        pole_line.set_data([x, px], [cart_h, py])
        ctrl_n = history["steps"]
        phase = "control" if frame < ctrl_n else "spin-down"
        ax.set_title(f"{phase}  frame {frame}  theta={theta:.2f}")
        return cart_rect, pole_line

    anim = animation.FuncAnimation(
        fig, update, frames=len(history["x"]), init_func=init, interval=interval_ms, blit=True
    )
    plt.show()
    # keep reference so animation isn't garbage-collected on some backends
    _ = anim


if __name__ == "__main__":
    env = CartPoleEnv(seed=0)
    fm = env.p.force_mag
    SPIN_DOWN = 600  # physics-only tail after episode ends (force = 0)

    def random_policy(obs: Obs) -> float:
        return float(env.rng.uniform(-fm, fm))

    h_ctrl = run_episode(env, random_policy, spin_down_steps=0)
    plot_rollout(h_ctrl, "Random policy")

    h_anim = run_episode(env, random_policy, spin_down_steps=SPIN_DOWN)
    animate_rollout(h_anim)
