"""
PPO homework scaffold — continuous-action CartPole (Gaussian policy).

Fill in the functions marked HOMEWORK below. Tips reference Schulman et al., "Proximal Policy
Optimization Algorithms" (2017) and the GAE paper (Schulman et al., 2018).

Run:  python ppo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from cartpole import CartPoleEnv


# -----------------------------------------------------------------------------
# Hyperparameters (tweak freely while debugging your formulas)
# -----------------------------------------------------------------------------


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95  # GAE lambda
    clip_eps: float = 0.2
    lr: float = 6e-4
    rollout_steps: int = 2048
    minibatch_size: int = 256
    epochs_per_rollout: int = 10
    vf_coef: float = 0.7  # critic loss scale
    ent_coef: float = 1e-2  # raise slightly (e.g. 1e-3) if you want entropy bonus


cfg = PPOConfig()


# -----------------------------------------------------------------------------
# Actor–critic (filled in — continuous 1-D Gaussian over raw force before env clip)
# -----------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """Gaussian over pre-tanh latent u; env force = force_mag * tanh(u)."""

    def __init__(self, obs_dim: int = 4, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(1))
        self.v_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        """obs shape [..., obs_dim]; returns mu [..., 1], log_std scalar broadcast, V [..., 1]."""
        h = self.body(obs)
        mu = self.mu_head(h)
        log_std = self.log_std.expand_as(mu)
        v = self.v_head(h)
        return mu, log_std, v


def evaluate_actions(policy: ActorCritic, obs: torch.Tensor, actions: torch.Tensor):
    """actions are pre-tanh latents u; log π(a) = log π(u) - log(1 - tanh²u) (scalar Jacobian)."""
    mu, log_std, v = policy(obs)
    std = log_std.exp().clamp(1e-6, 10)
    dist = Normal(mu.squeeze(-1), std.squeeze(-1))
    u = actions.squeeze(-1)
    tanh_u = torch.tanh(u)
    log_probs = dist.log_prob(u) - torch.log(1.0 - tanh_u.pow(2) + 1e-6)
    entropy = dist.entropy()
    return log_probs, v.squeeze(-1), entropy


# -----------------------------------------------------------------------------
# HOMEWORK 1 — Generalized Advantage Estimation + targets for the critic
# -----------------------------------------------------------------------------


def compute_gae_and_returns(
    rewards: torch.Tensor,  # [T]
    values: torch.Tensor,  # [T]   V(s_t) at rollout collection time (detached ok)
    dones: torch.Tensor,  # [T]   1 if episode ended after this step else 0 (float)
    last_value: torch.Tensor,  # scalar V(s_T), bootstrap when rollout truncates mid-episode
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build advantages A_t and value targets for supervised regression on V.

    Tips (GAE):
      • TD residual:  δ_t = r_t + γ * V_{t+1} * (1 - done_t) - V_t
        Here V_{t+1} means value at the *next* state along the rollout; use `last_value`
        only when indexing past the final stored transition (standard bootstrap).

      • Backward pass (t = T-1 … 0):
            A_t = δ_t + γ λ (1 - done_t) A_{t+1}

      • Often returns / critic targets are R_t = A_t + V_t   (what you regress V toward).

    Shapes: advantages and returns should both be [T], same device/dtype as rewards.
    """

    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    delta = rewards[T-1] + gamma * (1 - dones[T-1]) * last_value - values[T-1]
    advantages[T-1] = delta 
    returns[T-1] = advantages[T-1] + values[T-1]
    for t in reversed(range(T-1)):
        delta = rewards[t] + gamma * (1 - dones[t]) * values[t+1] - values[t]
        advantages[t] = delta + gamma * lam * (1 - dones[t]) * advantages[t+1]
        returns[t] = advantages[t] + values[t]
    

    return advantages, returns


# -----------------------------------------------------------------------------
# HOMEWORK 2 — clipped surrogate objective (policy loss as minimization term)
# -----------------------------------------------------------------------------


def clipped_policy_loss(
    old_log_probs: torch.Tensor,  # [N]
    new_log_probs: torch.Tensor,  # [N]
    advantages: torch.Tensor,  # [N]
    clip_eps: float,
) -> torch.Tensor:
    """
    PPO clipped surrogate — scalar loss to MINIMIZE (optimizer minimizes).

    Tips:
      • probability ratio: r_t(θ) = exp( log π_θ(a|s) - log π_old(a|s) )
      • unclipped objective per sample: r_t * Â_t
      • clipped version: clip(r_t, 1-ε, 1+ε) * Â_t
      • combine (negative sign matters — we maximize surrogate ⇒ minimize negative mean):
            L_CLIP = - mean_t  min( r_tÂ_t , clip(r_t, ...) Â_t )

    Optional course polish: normalize advantages batch-wise (mean/std) once before this call.
    """


    ratio = torch.exp(new_log_probs - old_log_probs)
    loss = torch.mean(torch.min(ratio * advantages, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages))

    return -loss


# -----------------------------------------------------------------------------
# HOMEWORK 3 — value-function regression loss
# -----------------------------------------------------------------------------


def value_loss(pred_values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Typical choice: mean squared error between predicted V(s) and your targets.

    Tip: torch.nn.functional.mse_loss(pred_values, targets)
    """


    loss = torch.nn.functional.mse_loss(pred_values, targets)

    return loss

# -----------------------------------------------------------------------------
# Rollout collection (filled in — uses env clipping on force magnitude)
# -----------------------------------------------------------------------------


@torch.no_grad()
def collect_rollout(env: CartPoleEnv, policy: ActorCritic, steps: int, device: torch.device):
    fm = env.p.force_mag
    obs_list, act_list, logp_list, rew_list, done_list, val_list = [], [], [], [], [], []

    obs = env.reset()
    policy.eval()

    for _ in range(steps):
        o_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        mu, log_std, v = policy(o_t.unsqueeze(0))
        mu = mu.squeeze(0).squeeze(-1)
        std = log_std.exp().clamp(1e-6, 10).squeeze(-1)
        dist = Normal(mu, std)
        u = dist.sample()
        tanh_u = torch.tanh(u)
        log_prob = (dist.log_prob(u) - torch.log(1.0 - tanh_u.pow(2) + 1e-6)).sum()

        action_np = float((fm * tanh_u).cpu())
        next_obs, rew, terminated, _ = env.step(action_np)

        obs_list.append(obs.astype(np.float32))
        act_list.append(u.cpu().squeeze())
        logp_list.append(log_prob.cpu())
        rew_list.append(rew)
        done_list.append(float(terminated))
        val_list.append(v.squeeze().cpu())

        obs = next_obs if not terminated else env.reset()

    # Bootstrap value V(s_last) for GAE tail (still detached — fixed baseline from old policy)
    with torch.no_grad():
        last_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        _, _, last_v = policy(last_obs_t)
        last_value = last_v.squeeze()

    batch = {
        "obs": torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=device),
        "actions": torch.stack(act_list).to(device).unsqueeze(-1),
        "old_log_probs": torch.stack(logp_list).to(device),
        "rewards": torch.as_tensor(rew_list, dtype=torch.float32, device=device),
        "dones": torch.as_tensor(done_list, dtype=torch.float32, device=device),
        "values": torch.stack(val_list).to(device),
        "last_value": last_value,
    }
    return batch


# -----------------------------------------------------------------------------
# Training loop (mostly wired — depends on your HOMEWORK functions)
# -----------------------------------------------------------------------------


def train_ppo(env: CartPoleEnv, policy: ActorCritic, opt: optim.Optimizer, device: torch.device):
    rollout = collect_rollout(env, policy, cfg.rollout_steps, device)

    rewards = rollout["rewards"]
    values_old = rollout["values"].detach()
    dones = rollout["dones"]
    last_value = rollout["last_value"].detach()

    advantages, returns = compute_gae_and_returns(
        rewards, values_old, dones, last_value, cfg.gamma, cfg.lam
    )

    # Tip from papers: normalize advantages across the rollout for numeric stability
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs_all = rollout["obs"]
    actions_all = rollout["actions"]
    old_log_probs_all = rollout["old_log_probs"]

    policy.train()
    T = obs_all.shape[0]
    idx = torch.arange(T, device=device)
    total_kl = 0.0
    total_clipped = 0
    total_samples = 0

    for _ in range(cfg.epochs_per_rollout):
        perm = idx[torch.randperm(T)]
        for start in range(0, T, cfg.minibatch_size):
            mb = perm[start : start + cfg.minibatch_size]
            obs_mb = obs_all[mb]
            act_mb = actions_all[mb]
            old_lp_mb = old_log_probs_all[mb]
            adv_mb = advantages[mb]
            ret_mb = returns[mb]

            new_lp, vals_mb, entropy = evaluate_actions(policy, obs_mb, act_mb)
            kl = torch.mean(old_lp_mb - new_lp)
            total_kl += float(kl.detach()) * obs_mb.shape[0]

            ratio = torch.exp(new_lp - old_lp_mb)
            clipped = (ratio > 1.0 + cfg.clip_eps) | (ratio < 1.0 - cfg.clip_eps)
            total_clipped += int(clipped.sum().item())
            total_samples += obs_mb.shape[0]

            pi_loss = clipped_policy_loss(old_lp_mb, new_lp, adv_mb, cfg.clip_eps)
            v_loss = value_loss(vals_mb, ret_mb)
            ent_bonus = entropy.mean()

            loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent_bonus

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            opt.step()

    avg_kl = total_kl / total_samples if total_samples > 0 else 0.0
    clip_frac = total_clipped / total_samples if total_samples > 0 else 0.0
    assert 0.0 <= clip_frac <= 1.0, f"clip_frac out of bounds: {clip_frac}"
    return avg_kl, clip_frac


def eval_mean_return(env: CartPoleEnv, policy: ActorCritic, episodes: int = 5, device=None):
    """Greedy-ish eval: use policy mean (no sampling)."""
    policy.eval()
    fm = env.p.force_mag
    totals = []
    with torch.no_grad():
        for _ in range(episodes):
            obs = env.reset()
            total = 0.0
            for _ in range(500):
                o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                mu, _, _ = policy(o)
                a = (fm * torch.tanh(mu.squeeze())).item()
                obs, r, done, _ = env.step(a)
                total += r
                if done:
                    break
            totals.append(total)
    return float(np.mean(totals))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CartPoleEnv(seed=42)
    torch.manual_seed(42)

    policy = ActorCritic().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)

    print(f"Device: {device}")
    print("Checkpoint written each iter to policy.pt (latest weights). Ctrl+C saves then exits.")

    try:
        for it in range(200):
            avg_kl, clip_frac = train_ppo(env, policy, optimizer, device)
            mean_ret = eval_mean_return(env, policy, episodes=5, device=device)
            print(
                f"iter {it:03d}  eval_mean_return≈ {mean_ret:.1f}  avg_kl≈ {avg_kl:.6f}  clip_frac≈ {clip_frac:.4f}  clip_pct≈ {clip_frac * 100:.1f}%"
            )

            torch.save(
                {"policy_state_dict": policy.state_dict(), "iter": it, "eval_mean_return": mean_ret},
                "policy.pt",
            )

            # Cart with friction is harder than classic Gym — tune this threshold if you like.
            if mean_ret >= 400:
                print("Looks pretty good — tune threshold / friction as needed.")
                break
    except KeyboardInterrupt:
        torch.save(
            {"policy_state_dict": policy.state_dict(), "iter": it, "note": "keyboard_interrupt"},
            "policy.pt",
        )
        print("\nSaved policy.pt after interrupt.")
        raise


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        print("Implement the HOMEWORK stubs in ppo.py, then run again.")
        print(f"Missing: {exc}")
