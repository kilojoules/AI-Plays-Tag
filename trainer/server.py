#!/usr/bin/env python3
"""
Production-ready-ish WebSocket bridge with a background Trainer.
Key changes:
- Dynamic obs_dim derived from first action request (vision-based obs).
- Trainer class manages buffers and updates outside the network I/O path.
"""
import asyncio
import json
import os
import random
import threading
from typing import Any, Dict, List, Tuple

import numpy as np
import websockets

try:
    import torch
    from ppo import PPOAgent, PPOConfig
except Exception:
    torch = None
    PPOAgent = None
    PPOConfig = None


class Trainer:
    def __init__(self, batch_target: int = 2048) -> None:
        self.policy = None  # type: ignore
        self.cfg = None     # type: ignore
        self.batch_target = batch_target
        self.buf_obs: List[List[float]] = []
        self.buf_act: List[List[float]] = []
        self.buf_rew: List[float] = []
        self.buf_done: List[bool] = []
        self.updates_done = 0
        self.current_ep_rewards: List[float] = []
        self.episodes_logged = 0
        self.log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics_csv = os.path.join(self.log_dir, "metrics.csv")
        self.metrics_columns = [
            "episode",
            "reward_mean",
            "reward_sum",
            "steps",
            "updates",
            "advantage_mean",
            "advantage_std",
            "policy_loss",
            "value_loss",
        ]
        self._prepare_metrics_csv()
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            self.writer = SummaryWriter(self.log_dir)
        except Exception:
            self.writer = None
        self.lock = threading.Lock()
        self.last_adv_mean = 0.0
        self.last_adv_std = 0.0
        self.last_policy_loss = 0.0
        self.last_value_loss = 0.0

    def ensure_policy(self, obs_dim: int, act_dim: int = 3):
        if torch is None or PPOConfig is None or PPOAgent is None:
            return
        if self.cfg is not None:
            if self.cfg.obs_dim != obs_dim or self.cfg.act_dim != act_dim:
                print(
                    "Observation/action space changed from"
                    f" ({self.cfg.obs_dim}, {self.cfg.act_dim}) to ({obs_dim}, {act_dim});"
                    " reinitialising policy"
                )
                self.policy = None
                self.cfg = None
                self.buf_obs, self.buf_act, self.buf_rew, self.buf_done = [], [], [], []
                self.current_ep_rewards = []
        if self.policy is not None:
            return
        self.cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
        self.policy = PPOAgent(self.cfg)
        if os.environ.get("DISABLE_POLICY_LOAD", "0").lower() in ("1", "true", "yes"):
            return
        policy_path = os.path.join(os.path.dirname(__file__), "policy.pt")
        try:
            if os.path.exists(policy_path):
                self.policy.load_policy(policy_path)
                print("Loaded policy.pt for action inference")
        except Exception as e:
            print("Policy load failed:", e)

    def act(self, obs: List[float]) -> List[float]:
        action, _, _ = self.act_with_cache("__single__", obs)
        return action

    def act_with_cache(self, agent_name: str, obs: List[float]) -> Tuple[List[float], float, float]:
        self.ensure_policy(len(obs))
        role = self._role_from_obs(obs)
        if role not in self.policies:
            return [random.uniform(-1, 1), random.uniform(-1, 1), 0.0], 0.0, 0.0
        action_np, logp, value = self.policies[role].act(np.array(obs, dtype=np.float32))
        action = [float(action_np[0]), float(action_np[1]), float(action_np[2] if action_np.size > 2 else 0.0)]
        cache_key = agent_name or "__single__"
        self.act_cache[cache_key] = {
            "logp": logp,
            "value": value,
            "role": role,
        }
        return action, logp, value

    def add_transition(self, obs: List[float], act: List[float], rew: float, done: bool):
        role = self._role_from_obs(obs)
        self._store_transition(role, obs, act, rew, done, next_obs=None, agent_name=None)

    def add_transition_batch(self, transitions: List[Dict[str, Any]]):
        for tr in transitions:
            try:
                obs = tr.get("obs")
                action = tr.get("action")
                reward = float(tr.get("reward", 0.0))
                done = bool(tr.get("done", False))
                next_obs = tr.get("next_obs")
                agent_name = ""
                info = tr.get("info")
                if isinstance(info, dict):
                    agent_name = str(info.get("agent", ""))
                if isinstance(obs, list) and isinstance(action, list):
                    role = self._role_from_obs(obs)
                    self._store_transition(role, obs, action, reward, done, next_obs=next_obs, agent_name=agent_name)
            except Exception:
                continue

    def maybe_update(self):
        if not self.policies:
            return
        updates: List[Tuple[str, Dict[str, float]]] = []
        total_samples = 0
        with self.lock:
            snapshot = {r: {k: v[:] for k, v in buf.items()} for r, buf in self.buffers.items()}
            for buf in self.buffers.values():
                for key in buf.keys():
                    buf[key].clear()
        for role, data in snapshot.items():
            if len(data["obs"]) < self.batch_target:
                with self.lock:
                    for key, values in data.items():
                        self.buffers[role][key].extend(values)
                continue
            info = self._update_role(role, data)
            updates.append((role, info))
            total_samples += info.get("batch_size", 0)
        if not updates:
            return
        self.global_updates += 1
        if total_samples > 0:
            self.last_adv_mean = sum(i[1]["adv_mean"] * i[1]["batch_size"] for i in updates) / total_samples
            self.last_adv_std = sum(i[1]["adv_std"] * i[1]["batch_size"] for i in updates) / total_samples
            self.last_policy_loss = sum(i[1]["policy_loss"] * i[1]["batch_size"] for i in updates) / total_samples
            self.last_value_loss = sum(i[1]["value_loss"] * i[1]["batch_size"] for i in updates) / total_samples
        base_dir = os.path.dirname(__file__)
        seeker_path = os.path.join(base_dir, "policy_seeker.pt")
        hider_path = os.path.join(base_dir, "policy_hider.pt")
        legacy_path = os.path.join(base_dir, "policy.pt")
        for role, info in updates:
            self.last_metrics[role] = info
            self.updates_done[role] += 1
            path = seeker_path if role == "seeker" else hider_path
            try:
                self.policies[role].save_policy(path)
                if role == "seeker":
                    self.policies[role].save_policy(legacy_path)
            except Exception as e:
                print(f"Policy save failed for {role}:", e)
            if self.writer is not None:
                step = self.updates_done[role]
                self.writer.add_scalar(f"{role}/updates", step, self.global_updates)
                self.writer.add_scalar(f"{role}/policy_loss", info["policy_loss"], step)
                self.writer.add_scalar(f"{role}/value_loss", info["value_loss"], step)
                self.writer.add_scalar(f"{role}/entropy", info["entropy"], step)

    def log_episode(self):
        if len(self.current_ep_rewards) == 0:
            return
        ep_sum = float(np.sum(self.current_ep_rewards))
        ep_mean = float(np.mean(self.current_ep_rewards))
        ep_steps = int(len(self.current_ep_rewards))
        self.episodes_logged += 1
        updates_total = sum(self.updates_done.values())
        with open(self.metrics_csv, "a") as f:
            f.write(
                ",".join(
                    [
                        str(self.episodes_logged),
                        f"{ep_mean}",
                        f"{ep_sum}",
                        f"{ep_steps}",
                        f"{updates_total}",
                        f"{self.last_adv_mean}",
                        f"{self.last_adv_std}",
                        f"{self.last_policy_loss}",
                        f"{self.last_value_loss}",
                    ]
                )
                + "\n"
            )
        if self.writer is not None:
            self.writer.add_scalar("episode/reward_mean", ep_mean, self.episodes_logged)
            self.writer.add_scalar("episode/reward_sum", ep_sum, self.episodes_logged)
            self.writer.add_scalar("episode/steps", ep_steps, self.episodes_logged)
        self.current_ep_rewards = []

    def _prepare_metrics_csv(self) -> None:
        header = ",".join(self.metrics_columns)
        if not os.path.exists(self.metrics_csv):
            with open(self.metrics_csv, "w") as f:
                f.write(header + "\n")
            return
        try:
            with open(self.metrics_csv, "r") as f:
                existing = f.readline().strip()
        except Exception:
            existing = ""
        if existing != header:
            backup = self.metrics_csv + ".legacy"
            try:
                os.replace(self.metrics_csv, backup)
                print(f"Metrics header changed; moved legacy log to {backup}")
            except Exception:
                pass
            with open(self.metrics_csv, "w") as f:
                f.write(header + "\n")

    def _store_transition(self, role: str, obs: List[float], act: List[float], rew: float, done: bool, next_obs: Any, agent_name: Any) -> None:
        cache = self._pull_cached_act(str(agent_name) if agent_name else "")
        if cache is not None:
            logp = cache.get("logp", 0.0)
            value = cache.get("value", 0.0)
        else:
            logp, value = self._evaluate_offline(role, obs, act)
        next_value = 0.0
        if not done and isinstance(next_obs, list) and role in self.policies:
            next_value = self.policies[role].value(np.array(next_obs, dtype=np.float32))
        with self.lock:
            buf = self.buffers.setdefault(role, self._make_buffer())
            buf["obs"].append(list(obs))
            buf["act"].append([float(act[0]), float(act[1]), float(act[2] if len(act) > 2 else 0.0)])
            buf["rew"].append(float(rew))
            buf["done"].append(bool(done))
            buf["logp"].append(float(logp))
            buf["val"].append(float(value))
            buf["next_val"].append(float(0.0 if done else next_value))
            self.current_ep_rewards.append(float(rew))
        if done:
            self.log_episode()

    def _pull_cached_act(self, agent_name: str) -> Dict[str, Any]:
        key = agent_name or "__single__"
        return self.act_cache.pop(key, None)

    def _evaluate_offline(self, role: str, obs: List[float], act: List[float]) -> Tuple[float, float]:
        if role not in self.policies:
            return 0.0, 0.0
        obs_tensor = torch.as_tensor(np.array(obs, dtype=np.float32)).unsqueeze(0)
        act_tensor = torch.as_tensor(np.array(act, dtype=np.float32)).unsqueeze(0)
        logp, _, value = self.policies[role].evaluate_actions(obs_tensor, act_tensor)
        return float(logp.item()), float(value.item())

    def _role_from_obs(self, obs: List[float]) -> str:
        if len(obs) > 8 and float(obs[8]) >= 0.5:
            return "seeker"
        return "hider"

    def _update_role(self, role: str, data: Dict[str, List[Any]]) -> Dict[str, float]:
        obs = np.array(data["obs"], dtype=np.float32)
        act = np.array(data["act"], dtype=np.float32)
        rew = np.array(data["rew"], dtype=np.float32)
        done = np.array(data["done"], dtype=np.bool_)
        logp_old = np.array(data["logp"], dtype=np.float32)
        values = np.array(data["val"], dtype=np.float32)
        next_values = np.array(data["next_val"], dtype=np.float32)
        adv = np.zeros_like(rew)
        gae = 0.0
        for t in reversed(range(len(rew))):
            mask = 0.0 if done[t] else 1.0
            delta = rew[t] + self.gamma * next_values[t] * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            adv[t] = gae
        returns = adv + values
        adv_std = np.std(adv)
        advantages = adv.copy()
        if adv_std > 1e-6:
            advantages = (advantages - np.mean(advantages)) / (adv_std + 1e-8)
        metrics = self.policies[role].update(obs, act, logp_old, returns, advantages)
        metrics.update({
            "adv_mean": float(np.mean(adv)),
            "adv_std": float(np.std(adv)),
            "batch_size": len(rew),
        })
        return metrics


trainer = Trainer()


async def handle(ws: websockets.WebSocketServerProtocol):
    async for msg in ws:
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            continue
        typ = data.get("type")
        if typ == "act":
            obs = data.get("obs")
            if not isinstance(obs, list):
                await ws.send(json.dumps({
                    "type": "action",
                    "action": [0.0, 0.0, 0.0],
                    "info": {"error": "invalid_obs"}
                }))
                continue
            action, _, _ = trainer.act_with_cache("__single__", obs)
            await ws.send(json.dumps({
                "type": "action",
                "action": action,
                "info": {}
            }))
        elif typ == "act_batch":
            obs_dict = data.get("obs", {})
            actions: Dict[str, List[float]] = {}
            for name, arr in obs_dict.items():
                if not isinstance(arr, list):
                    continue
                action, _, _ = trainer.act_with_cache(name, arr)
                actions[name] = action
            await ws.send(json.dumps({
                "type": "action_batch",
                "actions": actions,
                "info": {}
            }))
        elif typ == "transition":
            obs = data.get("obs")
            action = data.get("action")
            reward = float(data.get("reward", 0.0))
            done = bool(data.get("done", False))
            next_obs = data.get("next_obs")
            info = data.get("info")
            agent_name = ""
            if isinstance(info, dict):
                agent_name = str(info.get("agent", ""))
            if isinstance(obs, list) and isinstance(action, list):
                trainer._store_transition(trainer._role_from_obs(obs), obs, action, reward, done, next_obs, agent_name)
                trainer.maybe_update()
        elif typ == "transition_batch":
            transitions = data.get("transitions", [])
            if isinstance(transitions, list):
                trainer.add_transition_batch(transitions)
                trainer.maybe_update()
        else:
            await ws.send(json.dumps({"type": "echo", "recv": data}))


async def main(host: str = "127.0.0.1", port: int = 8765):
    async with websockets.serve(handle, host, port):
        print(f"WebSocket bridge listening on ws://{host}:{port}")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
