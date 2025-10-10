#!/usr/bin/env python3
"""
Production-ready-ish WebSocket bridge with a background Trainer.
Key changes:
- Dynamic obs_dim derived from first action request (vision-based obs).
- Trainer class manages buffers and updates outside the network I/O path.
"""
import asyncio
import json
import math
import os
import platform
import random
import sys
import threading
from datetime import datetime
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
        self.cfg: PPOConfig | None = None
        self.batch_target = batch_target
        self.roles = ("seeker", "hider")
        self.gamma = 0.99
        self.lam = 0.95
        self.policies: Dict[str, PPOAgent] = {}
        self.buffers: Dict[str, Dict[str, List[Any]]] = {role: self._make_buffer() for role in self.roles}
        self.act_cache: Dict[str, Dict[str, Any]] = {}
        self.updates_done: Dict[str, int] = {role: 0 for role in self.roles}
        self.global_updates = 0
        self.episodes_logged = 0
        self.metadata_env_keys = [
            "TRAIN_APPROACH",
            "TRAIN_VARIANT",
            "TRAIN_RUN_ID",
            "AI_IS_IT",
            "AI_CONTROL_ALL_AGENTS",
            "AI_DISTANCE_REWARD_SCALE",
            "AI_SEEKER_TIME_PENALTY",
            "AI_WIN_BONUS",
            "AI_MAX_STEPS_PER_EPISODE",
            "AI_STEP_TICK_INTERVAL",
            "SELF_PLAY_ROUNDS",
            "SELF_PLAY_DURATION",
        ]
        self._init_run_context()
        self.max_checkpoints = int(os.environ.get("PPO_MAX_CHECKPOINTS", "8"))
        self._reset_episode_stats()
        self.metrics_csv = os.path.join(self.log_dir, "metrics.csv")
        self.metrics_columns = [
            "episode",
            "episode_id",
            "reward_mean",
            "reward_sum",
            "steps",
            "updates",
            "advantage_mean",
            "advantage_std",
            "policy_loss",
            "value_loss",
            "seeker_reward_sum",
            "seeker_reward_mean",
            "seeker_steps",
            "seeker_avg_distance",
            "hider_reward_sum",
            "hider_reward_mean",
            "hider_steps",
            "hider_avg_distance",
            "winner",
            "terminal_reason",
            "duration_sec",
        ]
        self._prepare_metrics_csv()
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            self.writer = SummaryWriter(self.tb_dir)
        except Exception:
            self.writer = None
        self.lock = threading.Lock()
        self.last_metrics: Dict[str, Dict[str, float]] = {
            role: {"adv_mean": 0.0, "adv_std": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
            for role in self.roles
        }
        self.last_adv_mean = 0.0
        self.last_adv_std = 0.0
        self.last_policy_loss = 0.0
        self.last_value_loss = 0.0

    def _make_buffer(self) -> Dict[str, List[Any]]:
        return {
            "obs": [],
            "act": [],
            "rew": [],
            "done": [],
            "logp": [],
            "val": [],
            "next_val": [],
        }

    def _init_run_context(self) -> None:
        base_dir = os.path.dirname(__file__)
        self.logs_root = os.path.join(base_dir, "logs")
        os.makedirs(self.logs_root, exist_ok=True)
        self.approach = os.environ.get("TRAIN_APPROACH") or os.environ.get("TRAIN_VARIANT") or "default"
        self.run_id = os.environ.get("TRAIN_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
        runs_root = os.path.join(self.logs_root, "runs", self.approach)
        os.makedirs(runs_root, exist_ok=True)
        self.log_dir = os.path.join(runs_root, self.run_id)
        os.makedirs(self.log_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.log_dir, "tensorboard")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.metadata_path = os.path.join(self.log_dir, "metadata.json")
        self.latest_marker_path = os.path.join(self.logs_root, "latest_run.txt")
        self.run_started_at = datetime.utcnow().isoformat() + "Z"
        self._write_metadata()
        self._write_latest_marker()

    def _collect_run_metadata(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "run_id": self.run_id,
            "approach": self.approach,
            "started_at": self.run_started_at,
            "batch_target": self.batch_target,
            "gamma": self.gamma,
            "lambda": self.lam,
        }
        env_snapshot = {k: os.environ.get(k, "") for k in self.metadata_env_keys if os.environ.get(k)}
        if env_snapshot:
            meta["env"] = env_snapshot
        meta["platform"] = {
            "python": sys.version,
            "platform": platform.platform(),
        }
        if torch is not None:
            meta["pytorch_version"] = getattr(torch, "__version__", "")
        return meta

    def _write_metadata(self) -> None:
        data = self._collect_run_metadata()
        try:
            with open(self.metadata_path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    def _write_latest_marker(self) -> None:
        try:
            with open(self.latest_marker_path, "w") as f:
                f.write(self.log_dir + "\n")
        except OSError:
            pass

    def _reset_episode_stats(self) -> None:
        self.current_ep_rewards: Dict[str, List[float]] = {role: [] for role in self.roles}
        self.current_ep_steps: Dict[str, int] = {role: 0 for role in self.roles}
        self.current_ep_distances: Dict[str, List[float]] = {role: [] for role in self.roles}
        self.current_ep_duration: float = 0.0
        self.current_ep_winner: str = ""
        self.current_ep_terminal_reason: str = ""
        self.current_ep_episode_id: int | None = None

    def ensure_policy(self, obs_dim: int, act_dim: int = 3) -> None:
        if torch is None or PPOConfig is None or PPOAgent is None:
            return
        if self.cfg is not None and (self.cfg.obs_dim != obs_dim or self.cfg.act_dim != act_dim):
            print(
                "Observation/action space changed from"
                f" ({self.cfg.obs_dim}, {self.cfg.act_dim}) to ({obs_dim}, {act_dim});"
                " reinitialising policies"
            )
            self.cfg = None
            self.policies = {}
            self.buffers = {role: self._make_buffer() for role in self.roles}
            self.act_cache.clear()
            self.updates_done = {role: 0 for role in self.roles}
            self.last_metrics = {
                role: {"adv_mean": 0.0, "adv_std": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
                for role in self.roles
            }
        if self.cfg is not None and self.policies:
            return
        self.cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
        self.policies = {role: PPOAgent(self.cfg) for role in self.roles}
        if os.environ.get("DISABLE_POLICY_LOAD", "0").lower() in ("1", "true", "yes"):
            return
        base_dir = os.path.dirname(__file__)
        legacy_path = os.path.join(base_dir, "policy.pt")
        for role in self.roles:
            path = os.path.join(base_dir, f"policy_{role}.pt")
            try:
                if os.path.exists(path):
                    self.policies[role].load_policy(path)
                    print(f"Loaded {path} for {role}")
                elif os.path.exists(legacy_path) and role == "seeker":
                    self.policies[role].load_policy(legacy_path)
                    print("Loaded legacy policy.pt for seeker")
            except Exception as exc:
                print(f"Policy load failed for {role}:", exc)

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
        self.act_cache[cache_key] = {"logp": logp, "value": value, "role": role}
        return action, logp, value

    def add_transition(
        self,
        obs: List[float],
        act: List[float],
        rew: float,
        done: bool,
        next_obs: Any = None,
        info: Dict[str, Any] | None = None,
        agent_name: str | None = None,
    ) -> None:
        role = self._role_from_obs(obs)
        if self._store_transition(role, obs, act, rew, done, next_obs, agent_name, info):
            self.log_episode()

    def add_transition_batch(self, transitions: List[Dict[str, Any]]) -> None:
        episode_finished = False
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
                    if self._store_transition(role, obs, action, reward, done, next_obs, agent_name, info):
                        episode_finished = True
            except Exception:
                continue
        if episode_finished:
            self.log_episode()

    def maybe_update(self) -> None:
        if not self.policies:
            return
        updates: List[Tuple[str, Dict[str, float]]] = []
        total_samples = 0
        with self.lock:
            snapshot = {role: {k: v[:] for k, v in buf.items()} for role, buf in self.buffers.items()}
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
            if info.get("rolled_back"):
                with self.lock:
                    for key, values in data.items():
                        self.buffers[role][key].extend(values)
                continue
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
        for role, info in updates:
            self.last_metrics[role] = info
            self.updates_done[role] += 1
            self._save_checkpoint(role)
            if role == "seeker":
                self._save_legacy_policy()
            if self.writer is not None:
                step = self.updates_done[role]
                self.writer.add_scalar(f"{role}/updates", step, self.global_updates)
                self.writer.add_scalar(f"{role}/policy_loss", info["policy_loss"], step)
                self.writer.add_scalar(f"{role}/value_loss", info["value_loss"], step)
                self.writer.add_scalar(f"{role}/entropy", info["entropy"], step)

    def log_episode(self) -> None:
        total_samples = sum(len(self.current_ep_rewards.get(role, [])) for role in self.roles)
        if total_samples == 0:
            return
        ep_sum = float(
            sum(float(np.sum(self.current_ep_rewards.get(role, []))) for role in self.roles if self.current_ep_rewards.get(role))
        )
        ep_mean = float(ep_sum / total_samples) if total_samples > 0 else 0.0
        ep_steps = int(max(self.current_ep_steps.values()) if self.current_ep_steps else total_samples)
        self.episodes_logged += 1
        updates_total = sum(self.updates_done.values())
        seeker_rewards = self.current_ep_rewards.get("seeker", [])
        hider_rewards = self.current_ep_rewards.get("hider", [])
        seeker_sum = float(np.sum(seeker_rewards)) if seeker_rewards else 0.0
        hider_sum = float(np.sum(hider_rewards)) if hider_rewards else 0.0
        seeker_mean = float(np.mean(seeker_rewards)) if seeker_rewards else 0.0
        hider_mean = float(np.mean(hider_rewards)) if hider_rewards else 0.0
        seeker_steps = int(self.current_ep_steps.get("seeker", len(seeker_rewards)))
        hider_steps = int(self.current_ep_steps.get("hider", len(hider_rewards)))
        seeker_distances = self.current_ep_distances.get("seeker", [])
        hider_distances = self.current_ep_distances.get("hider", [])
        seeker_avg_distance = float(np.mean(seeker_distances)) if seeker_distances else 0.0
        hider_avg_distance = float(np.mean(hider_distances)) if hider_distances else 0.0
        duration_sec = float(self.current_ep_duration)
        episode_id = self.current_ep_episode_id if self.current_ep_episode_id is not None else self.episodes_logged
        winner = self.current_ep_winner
        terminal_reason = self.current_ep_terminal_reason
        with open(self.metrics_csv, "a") as f:
            f.write(
                ",".join(
                    [
                        str(self.episodes_logged),
                        str(episode_id),
                        f"{ep_mean}",
                        f"{ep_sum}",
                        f"{ep_steps}",
                        f"{updates_total}",
                        f"{self.last_adv_mean}",
                        f"{self.last_adv_std}",
                        f"{self.last_policy_loss}",
                        f"{self.last_value_loss}",
                        f"{seeker_sum}",
                        f"{seeker_mean}",
                        f"{seeker_steps}",
                        f"{seeker_avg_distance}",
                        f"{hider_sum}",
                        f"{hider_mean}",
                        f"{hider_steps}",
                        f"{hider_avg_distance}",
                        winner or "",
                        terminal_reason or "",
                        f"{duration_sec}",
                    ]
                )
                + "\n"
            )
        if self.writer is not None:
            self.writer.add_scalar("episode/reward_mean", ep_mean, self.episodes_logged)
            self.writer.add_scalar("episode/reward_sum", ep_sum, self.episodes_logged)
            self.writer.add_scalar("episode/steps", ep_steps, self.episodes_logged)
            self.writer.add_scalar("episode/seeker_reward_sum", seeker_sum, self.episodes_logged)
            self.writer.add_scalar("episode/hider_reward_sum", hider_sum, self.episodes_logged)
        self._reset_episode_stats()
        self.act_cache.clear()

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

    def _store_transition(
        self,
        role: str,
        obs: List[float],
        act: List[float],
        rew: float,
        done: bool,
        next_obs: Any,
        agent_name: Any,
        info: Any,
    ) -> bool:
        done_flag = bool(done)
        cache = self._pull_cached_act(str(agent_name) if agent_name else "")
        if cache is not None:
            logp = cache.get("logp", 0.0)
            value = cache.get("value", 0.0)
        else:
            logp, value = self._evaluate_offline(role, obs, act)
        next_value = 0.0
        if not done and isinstance(next_obs, list) and role in self.policies:
            next_value = self.policies[role].value(np.array(next_obs, dtype=np.float32))
        info_dict: Dict[str, Any] = info if isinstance(info, dict) else {}
        with self.lock:
            buf = self.buffers.setdefault(role, self._make_buffer())
            buf["obs"].append(list(obs))
            buf["act"].append([float(act[0]), float(act[1]), float(act[2] if len(act) > 2 else 0.0)])
            buf["rew"].append(float(rew))
            buf["done"].append(bool(done))
            buf["logp"].append(float(logp))
            buf["val"].append(float(value))
            buf["next_val"].append(float(0.0 if done else next_value))
            rewards_list = self.current_ep_rewards.setdefault(role, [])
            rewards_list.append(float(rew))
            self.current_ep_steps[role] = self.current_ep_steps.get(role, 0) + 1
            if info_dict:
                dist = info_dict.get("distance_to_other")
                if dist is not None:
                    try:
                        self.current_ep_distances.setdefault(role, []).append(float(dist))
                    except (TypeError, ValueError):
                        pass
                time_elapsed = info_dict.get("time_elapsed")
                if time_elapsed is not None:
                    try:
                        self.current_ep_duration = max(self.current_ep_duration, float(time_elapsed))
                    except (TypeError, ValueError):
                        pass
                if not self.current_ep_winner:
                    winner = info_dict.get("winner")
                    if isinstance(winner, str):
                        self.current_ep_winner = winner
                if not self.current_ep_terminal_reason:
                    terminal_reason = info_dict.get("terminal_reason")
                    if isinstance(terminal_reason, str):
                        self.current_ep_terminal_reason = terminal_reason
                if self.current_ep_episode_id is None:
                    episode_id = info_dict.get("episode")
                    if isinstance(episode_id, (int, float)):
                        self.current_ep_episode_id = int(episode_id)
        if done_flag:
            if agent_name:
                self.act_cache.pop(str(agent_name), None)
        return done_flag

    def _pull_cached_act(self, agent_name: str) -> Dict[str, Any] | None:
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

    def _snapshot_policy(self, role: str) -> Dict[str, Dict[str, torch.Tensor]]:
        policy = self.policies[role]
        return {
            "pi": {k: v.clone() for k, v in policy.pi.state_dict().items()},
            "vf": {k: v.clone() for k, v in policy.vf.state_dict().items()},
        }

    def _restore_policy(self, role: str, snapshot: Dict[str, Dict[str, torch.Tensor]]) -> None:
        policy = self.policies[role]
        policy.pi.load_state_dict(snapshot["pi"])
        policy.vf.load_state_dict(snapshot["vf"])

    def _checkpoint_path(self, role: str, step: int | None = None) -> str:
        suffix = f"{step:05d}" if step is not None else datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.checkpoint_dir, f"{role}_{suffix}.pt")

    def _save_checkpoint(self, role: str) -> None:
        step = self.updates_done[role]
        path = self._checkpoint_path(role, step)
        try:
            self.policies[role].save_policy(path)
            self._trim_checkpoints(role)
        except Exception as exc:
            print(f"Checkpoint save failed for {role}:", exc)

    def _trim_checkpoints(self, role: str) -> None:
        if self.max_checkpoints <= 0:
            return
        files = sorted(
            [p for p in os.listdir(self.checkpoint_dir) if p.startswith(f"{role}_")],
            key=lambda name: os.path.getmtime(os.path.join(self.checkpoint_dir, name)),
        )
        while len(files) > self.max_checkpoints:
            old = files.pop(0)
            try:
                os.remove(os.path.join(self.checkpoint_dir, old))
            except OSError:
                pass

    def _save_legacy_policy(self) -> None:
        base_dir = os.path.dirname(__file__)
        legacy_path = os.path.join(base_dir, "policy.pt")
        try:
            self.policies["seeker"].save_policy(legacy_path)
        except Exception as exc:
            print("Failed to update legacy policy.pt:", exc)

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
        snapshot = self._snapshot_policy(role)
        try:
            metrics = self.policies[role].update(obs, act, logp_old, returns, advantages)
        except Exception as exc:
            print(f"Update failed for {role}:", exc)
            self._restore_policy(role, snapshot)
            metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "rolled_back": True, "batch_size": 0}
            metrics.update({"adv_mean": float(np.mean(adv)), "adv_std": float(np.std(adv))})
            return metrics
        metrics.update({
            "adv_mean": float(np.mean(adv)),
            "adv_std": float(np.std(adv)),
            "batch_size": len(rew),
        })
        if (
            math.isnan(metrics["policy_loss"])
            or math.isnan(metrics["value_loss"])
            or math.isinf(metrics["policy_loss"])
            or math.isinf(metrics["value_loss"])
            or abs(metrics.get("approx_kl", 0.0)) > self.cfg.target_kl * 5.0
        ):
            self._restore_policy(role, snapshot)
            metrics["rolled_back"] = True
            metrics["batch_size"] = 0
            return metrics
        metrics["rolled_back"] = False
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
                finished = trainer._store_transition(
                    trainer._role_from_obs(obs), obs, action, reward, done, next_obs, agent_name, info
                )
                if finished:
                    trainer.log_episode()
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
