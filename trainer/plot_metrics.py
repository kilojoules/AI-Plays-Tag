#!/usr/bin/env python3
import csv
import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
CSV_PATH = os.path.join(LOG_DIR, "metrics.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(OUT_DIR, exist_ok=True)

episodes: List[int] = []
reward_mean: List[float] = []
reward_sum: List[float] = []
steps: List[int] = []
updates: List[int] = []
episodes_adv: List[int] = []
adv_mean: List[float] = []
adv_std: List[float] = []
episodes_policy: List[int] = []
policy_loss: List[float] = []
episodes_value: List[int] = []
value_loss: List[float] = []

if not os.path.exists(CSV_PATH):
    print(f"No metrics found at {CSV_PATH}")
    raise SystemExit(0)

with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        episodes.append(int(row["episode"]))
        reward_mean.append(float(row["reward_mean"]))
        reward_sum.append(float(row["reward_sum"]))
        steps.append(int(row["steps"]))
        updates.append(int(row["updates"]))
        val = row.get("advantage_mean")
        if val:
            episodes_adv.append(int(row["episode"]))
            adv_mean.append(float(val))
        val = row.get("advantage_std")
        if val:
            adv_std.append(float(val))
        val = row.get("policy_loss")
        if val:
            episodes_policy.append(int(row["episode"]))
            policy_loss.append(float(val))
        val = row.get("value_loss")
        if val:
            episodes_value.append(int(row["episode"]))
            value_loss.append(float(val))

def save_plot(x, y, title, ylabel, filename):
    plt.figure(figsize=(8,4))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path)
    print("Saved:", path)

if episodes:
    save_plot(episodes, reward_mean, "Reward Mean per Episode", "Reward Mean", "reward_mean.png")
    save_plot(episodes, reward_sum, "Reward Sum per Episode", "Reward Sum", "reward_sum.png")
    save_plot(episodes, steps, "Episode Length", "Steps", "steps.png")
    save_plot(episodes, updates, "Policy Updates", "Updates", "updates.png")
if episodes_adv and adv_mean:
    save_plot(episodes_adv, adv_mean, "Advantage Mean", "Advantage Mean", "advantage_mean.png")
if episodes_adv and adv_std:
    save_plot(episodes_adv, adv_std, "Advantage Std Dev", "Advantage Std", "advantage_std.png")
if episodes_policy and policy_loss:
    save_plot(episodes_policy, policy_loss, "Policy Loss", "Loss", "policy_loss.png")
if episodes_value and value_loss:
    save_plot(episodes_value, value_loss, "Value Loss", "Loss", "value_loss.png")
