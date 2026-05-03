"""
Load a checkpoint from training (policy.pt) and visualize behavior.

  python play.py
  python play.py --weights policy.pt --spin-down 600 --animate
"""

from __future__ import annotations

import argparse

import torch

from cartpole import CartPoleEnv
from ppo import ActorCritic
from visualize import Obs, animate_rollout, plot_rollout, run_episode


def load_policy(path: str, device: torch.device) -> ActorCritic:
    ckpt = torch.load(path, map_location=device)
    net = ActorCritic().to(device)
    net.load_state_dict(ckpt["policy_state_dict"])
    net.eval()
    return net


def make_mean_policy(net: ActorCritic, env: CartPoleEnv, device: torch.device):
    """Same deterministic control as eval_mean_return in ppo.py."""
    fm = env.p.force_mag

    def policy(obs: Obs) -> float:
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mu, _, _ = net(o)
            a = fm * torch.tanh(mu.squeeze())
        return float(a.cpu())

    return policy


def main():
    p = argparse.ArgumentParser(description="Visualize trained PPO policy on CartPole.")
    p.add_argument("--weights", default="policy.pt", help="Checkpoint from ppo.py")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spin-down", type=int, default=600, help="Extra steps at force=0 after failure")
    p.add_argument("--animate", action="store_true", help="Open stick-figure animation")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CartPoleEnv(seed=args.seed)
    policy_net = load_policy(args.weights, device)
    policy_fn = make_mean_policy(policy_net, env, device)

    h = run_episode(env, policy_fn, spin_down_steps=args.spin_down)
    plot_rollout(h, title=f"Trained policy ({args.weights})")
    if args.animate:
        env2 = CartPoleEnv(seed=args.seed)
        h2 = run_episode(env2, policy_fn, spin_down_steps=args.spin_down)
        animate_rollout(h2)


if __name__ == "__main__":
    main()
