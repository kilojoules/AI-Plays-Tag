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

