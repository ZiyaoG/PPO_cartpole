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
PolicyFn = Callable[[Obs], int]


def run_episode(env: CartPoleEnv, policy: PolicyFn, max_steps: int = 500):
    obs = env.reset()
    xs, thetas, rewards = [], [], []
    total = 0.0
    for _ in range(max_steps):
        xs.append(obs[0])
        thetas.append(obs[2])
        a = int(policy(obs))
        obs, r, done, _ = env.step(a)
        rewards.append(r)
        total += r
        if done:
            break
    return {
        "x": np.array(xs),
        "theta": np.array(thetas),
        "reward_per_step": np.array(rewards),
        "total_reward": total,
        "steps": len(rewards),
    }


def plot_rollout(history: dict, title: str = "Rollout") -> None:
    t = np.arange(len(history["theta"]))
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, history["theta"])
    axes[0].set_ylabel("theta (rad)")
    axes[1].plot(t, history["x"])
    axes[1].set_ylabel("cart x")
    axes[2].step(t, history["reward_per_step"], where="post")
    axes[2].set_ylabel("reward")
    axes[2].set_xlabel("step")
    fig.suptitle(f"{title} — return={history['total_reward']:.0f}, steps={history['steps']}")
    plt.tight_layout()
    plt.show()


def animate_rollout(history: dict, interval_ms: int = 30) -> None:
    """Crude side-view stick figure; good enough to see failures."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=1)

    cart_w, cart_h = 0.3, 0.15
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
        ax.set_title(f"step {frame}  theta={theta:.2f}")
        return cart_rect, pole_line

    anim = animation.FuncAnimation(
        fig, update, frames=len(history["x"]), init_func=init, interval=interval_ms, blit=True
    )
    plt.show()
    # keep reference so animation isn't garbage-collected on some backends
    _ = anim


if __name__ == "__main__":
    env = CartPoleEnv(seed=0)

    def random_policy(obs: Obs) -> int:
        return int(np.random.randint(0, 2))

    h = run_episode(env, random_policy)
    plot_rollout(h, "Random policy")
    # animate_rollout(h)
