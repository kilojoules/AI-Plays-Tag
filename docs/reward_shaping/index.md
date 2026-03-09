---
layout: default
title: Reward Shaping in Multi-Agent Tag
---

# Reward Shaping in Multi-Agent Tag

*How reward function design determines whether RL agents learn to play tag — or learn to hide in corners.*

---

## The Problem

Training two RL agents to play tag seems simple: the seeker gets a reward for catching, the hider gets a reward for surviving. But the details of reward shaping dramatically affect what behaviors emerge.

With minimal or sparse rewards, agents develop degenerate strategies:

<div align="center">
<table>
<tr>
<td align="center"><b>R0 Baseline (PPO)</b><br>Passive pursuit, no urgency</td>
<td align="center"><b>R4 Sparse (PPO)</b><br>Wall camping, no real evasion</td>
</tr>
<tr>
<td><img src="bad_R0_baseline.gif" width="350"/></td>
<td><img src="bad_R4_sparse.gif" width="350"/></td>
</tr>
</table>
</div>

The baseline seeker wanders without urgency. The sparse-reward hider camps in a corner — technically optimal (maximizes distance), but not what we want.

---

## Environment

### Arena

The game takes place in a $30 \times 30$ bounded arena ($L = 15$, coordinates in $[-L, L]^2$) with 4 rectangular obstacles and a central safe zone. The hider has a 15% speed advantage over the seeker. Episodes last $T_{\max} = 200$ action steps (10 seconds of simulated time at $\Delta t = 1/60$ s, 3 physics substeps per action).

### State Space

At each timestep $t$, each agent $i \in \lbrace S, H \rbrace$ (seeker, hider) observes a vector $o_i^t \in \mathbb{R}^{87}$:

$$
o_i^t = \bigl[\, \underbrace{p_i / L}_{\text{position}},\; \underbrace{v_i / 10}_{\text{velocity}},\; \underbrace{(p_j - p_i)/L}_{\text{relative pos.}},\; \underbrace{(v_j - v_i)/10}_{\text{relative vel.}},\; \underbrace{\rho_i}_{\text{role flags}},\; \underbrace{(\cos\theta_i, \sin\theta_i)}_{\text{facing}},\; \underbrace{r_i \in \mathbb{R}^{72}}_{\text{vision rays}},\; \underbrace{z \in \mathbb{R}^3}_{\text{safe zone}} \,\bigr]
$$

where:
- $p_i \in \mathbb{R}^2$ is the agent's position, $v_i \in \mathbb{R}^2$ its velocity
- $j$ denotes the opponent
- $\rho_S = (1, 0)$, $\rho_H = (0, 1)$ are one-hot role flags
- $\theta_i$ is the facing angle
- $r_i$ consists of 36 rays spanning a 120° FOV, each returning (normalized distance, hit type) where hit type encodes wall (0), obstacle (0.5), or agent (1)
- $z = (\text{in\_zone}, \text{exhausted}, \text{cooldown\_frac})$ is the safe zone state

### Action Space

Each agent outputs a continuous action $a_i^t \in [-1, 1]^3$:

$$
a_i^t = (a_x, a_y, a_{\text{sprint}})
$$

where $(a_x, a_y)$ specify a target velocity direction scaled by max speed, and $a_{\text{sprint}}$ controls sprint intensity (unused in this study). The environment applies acceleration-based physics: $v_i \leftarrow v_i + \alpha(a_i \cdot v_{\max} - v_i)\Delta t$ with $\alpha = 20$ and $v_{\max} = 8.0$ for the seeker, $v_{\max} = 9.2$ for the hider.

### Tagging Condition

The seeker tags the hider when $\lVert p_S - p_H \rVert < d_{\text{tag}} = 1.5$, unless the hider is in the safe zone (radius 2.5, centered at origin) with remaining protection time.

---

## Reward Functions

All presets share the same **terminal rewards**:

$$
r_S^{\text{terminal}} = \begin{cases}
+W & \text{if tagged} \\
-W & \text{if timeout}
\end{cases}
\qquad
r_H^{\text{terminal}} = \begin{cases}
-W & \text{if tagged} \\
+B & \text{if timeout}
\end{cases}
$$

with win bonus $W = 10$ and timeout hider bonus $B = 6$. The presets differ in their per-step **shaping rewards**, which are summed to form the total per-step reward for each agent.

### Shaping Components

We define the following notation:

| Symbol | Definition |
|--------|-----------|
| $d_t = \lVert p_S^t - p_H^t \rVert$ | Inter-agent distance at step $t$ |
| $\Delta d_t = d_{t-1} - d_t$ | Distance progress (positive when seeker closes gap) |
| $w_H^t = L - \max(\lvert p_{H,x}^t \rvert, \lvert p_{H,y}^t \rvert)$ | Hider's minimum distance to any wall |
| $s_H^t = \lVert v_H^t \rVert$ | Hider speed |
| $\mathcal{G}_i^t \subseteq \lbrace 1, \ldots, 6 \rbrace^2$ | Set of $6 \times 6$ grid cells visited by agent $i$ up to step $t$ |
| $t / T_{\max}$ | Normalized episode progress |

The per-step shaping reward for each agent is:

$$
r_S^{\text{step}} = \underbrace{c_{\text{time}} \cdot f(t)}_{\text{time penalty}} + \underbrace{c_{\text{dist}} \cdot \Delta d_t}_{\text{pursuit shaping}} + \underbrace{c_{\text{cov}} \cdot \lvert \mathcal{G}_S^t \setminus \mathcal{G}_S^{t-1} \rvert}_{\text{coverage bonus}}
$$

$$
r_H^{\text{step}} = \underbrace{c_{\text{surv}}}_{\text{survival bonus}} - \underbrace{c_{\Delta d} \cdot \Delta d_t}_{\text{evasion (distance change)}} + \underbrace{c_{|d|} \cdot \frac{d_t}{L}}_{\text{absolute distance}} + \underbrace{c_{\text{wall}} \cdot \max\!\Big(0,\; 1 - \frac{w_H^t}{d_w}\Big)}_{\text{wall proximity penalty}} + \underbrace{c_{\text{speed}} \cdot \mathbb{1}[s_H^t > 1]}_{\text{speed bonus}} + \underbrace{c_{\text{cov}} \cdot \lvert \mathcal{G}_H^t \setminus \mathcal{G}_H^{t-1} \rvert}_{\text{coverage bonus}}
$$

where the time penalty scaling function is:

$$
f(t) = \begin{cases}
1 + t / T_{\max} & \text{if escalating urgency is enabled} \\
1 & \text{otherwise}
\end{cases}
$$

and $d_w = 2.0$ is the wall proximity threshold.

### Preset Coefficients

Each preset selects a subset of these terms by setting specific coefficients:

| Preset | $c_{\text{time}}$ | $f(t)$ | $c_{\text{dist}}$ | $c_{\text{surv}}$ | $c_{\Delta d}$ | $c_{\lvert d \rvert}$ | $c_{\text{wall}}$ | $c_{\text{speed}}$ | $c_{\text{cov}}$ |
|--------|-----------------:|:------:|------------------:|-----------------:|---------------:|---------------------:|-----------------:|-------------------:|-----------------:|
| **R0** Baseline | $-0.005$ | $1$ | $0$ | $0.01$ | $0$ | $0$ | $0$ | $0$ | $0$ |
| **R1** Pursuit | $-0.02$ | $1$ | $0.2$ | $0.01$ | $0$ | $0$ | $0$ | $0$ | $0$ |
| **R2** Active | $-0.005$ | $1$ | $0.14$ | $0.01$ | $0.14$ | $0.1$ | $-0.02$ | $0.005$ | $0$ |
| **R3** Both | $-0.02$ | $1$ | $0.2$ | $0.01$ | $0.14$ | $0.1$ | $-0.02$ | $0.005$ | $0$ |
| **R4** Sparse | $0$ | $1$ | $0$ | $0$ | $0$ | $0$ | $0$ | $0$ | $0$ |
| **R5** Escalating | $-0.01$ | $1 + t/T$ | $0.14$ | $0.01$ | $0$ | $0.1$ | $0$ | $0$ | $0$ |
| **R6** Coverage | $-0.005$ | $1$ | $0.14$ | $0.01$ | $0$ | $0.05$ | $0$ | $0$ | $0.1$ |
| **R7** Kitchen Sink | $-0.015$ | $1 + t/T$ | $0.2$ | $0.01$ | $0.14$ | $0.1$ | $-0.02$ | $0.005$ | $0.05$ |

Design rationale:
- **R0** and **R4** are controls: R0 has minimal shaping, R4 has none at all.
- **R1** tests whether strong seeker signals alone suffice.
- **R2** tests whether anti-degenerate hider shaping (wall penalty + speed bonus) prevents corner camping.
- **R3** combines R1 and R2 to see if both-agent shaping compounds.
- **R5** tests whether dynamic pressure (escalating time penalty) creates more interesting pursuit.
- **R6** tests whether exploration incentives (grid coverage bonus) help agents discover the full arena.
- **R7** combines everything, testing whether more terms always helps.

---

## Algorithms

Each preset is trained with two algorithms:

**PPO** (Proximal Policy Optimization) — on-policy actor-critic with clipped surrogate objective. Both agents collect rollouts simultaneously from 64 parallel environments, with batch size 4096 and 10 optimization epochs per update.

**SAC** (Soft Actor-Critic) — off-policy actor-critic that maximizes reward plus an entropy bonus $\mathcal{H}[\pi]$, with automatic temperature tuning. Each agent maintains a separate replay buffer (500K transitions), twin Q-networks, and squashed Gaussian policy. The entropy term is:

$$
J_\pi = \mathbb{E}\Big[\sum_t r_t + \alpha \mathcal{H}\big[\pi(\cdot \mid s_t)\big]\Big]
$$

where $\alpha$ is automatically tuned to maintain target entropy $-\dim(\mathcal{A}) = -3$. This implicit exploration bonus proves critical for hider performance.

Both algorithms use the same network architecture: 2-layer MLP (256 hidden units, ReLU activations). All runs use learning rate $3 \times 10^{-4}$, $\gamma = 0.99$, and train for 5M timesteps across 3 random seeds.

---

## What Good Behavior Looks Like

With the right reward shaping, agents learn genuine pursuit and evasion:

<div align="center">
<table>
<tr>
<td align="center"><b>R2 Active Hider (PPO)</b><br>Wall penalty + speed bonus</td>
<td align="center"><b>R5 Escalating (SAC)</b><br>Increasing seeker urgency</td>
</tr>
<tr>
<td><img src="hero_R2_active.gif" width="350"/></td>
<td><img src="hero_R5_escalating.gif" width="350"/></td>
</tr>
<tr>
<td align="center"><b>R3 Both Shaped (SAC)</b><br>Seeker pursuit + hider evasion</td>
<td align="center"><b>R7 Kitchen Sink (PPO)</b><br>All shaping terms combined</td>
</tr>
<tr>
<td><img src="hero_R3_both.gif" width="350"/></td>
<td><img src="hero_R7_kitchen_sink.gif" width="350"/></td>
</tr>
</table>
</div>

## Cross-Config Gauntlet

To measure which reward functions produce *genuinely capable* agents (not just agents that look good against their training partner), I ran a full cross-evaluation: every seeker vs every hider across all 16 configurations, 20 episodes per matchup (256 matchups total).

![Gauntlet Heatmap](gauntlet_heatmap.png)

*Each cell shows seeker win rate (%). Rows = seeker config, columns = hider config. Green = seeker wins, red = hider survives.*

### Key Patterns

The heatmap reveals a striking **algorithm asymmetry**: SAC hiders (odd columns) are dramatically harder to catch than PPO hiders (even columns), forming a clear vertical stripe pattern. Meanwhile, SAC and PPO seekers are more comparable.

## Seeker and Hider Strength

Aggregating across all opponents gives each config's overall strength:

![Strength Bars](strength_bars.png)

**Findings:**

- **SAC dominates hider performance** — every SAC hider achieves 65–87% survival rate vs 7–45% for PPO hiders. SAC's entropy maximization naturally encourages diverse, hard-to-predict evasion.
- **Shaped PPO seekers match SAC** — R7 Kitchen Sink PPO (60% win rate) rivals the best SAC seekers. Dense reward shaping compensates for PPO's lack of entropy bonus.
- **Sparse rewards fail for PPO** — R4 Sparse PPO produces the worst agents in both roles (7% each). But R4 Sparse *SAC* produces the best hider (87% survival) — SAC's entropy bonus compensates for missing reward signal.
- **More shaping isn't always better** — R7 Kitchen Sink doesn't dominate. R5 Escalating SAC is the strongest seeker (64%), and it uses fewer reward terms than R7.

## PPO vs SAC

![Algorithm Comparison](algo_comparison.png)

*Points above the diagonal = SAC is stronger. Nearly all hider points sit above the line — SAC produces fundamentally better evaders.*

The algorithm comparison reveals that the PPO vs SAC choice matters more for hiders than seekers. Pursuit is a relatively simple objective that PPO can solve with enough reward shaping. Evasion requires the kind of stochastic, unpredictable behavior that SAC's entropy maximization provides naturally.

## Learning Curves

![Learning Curves](learning_curves.png)

*Seeker win rate over training (mean $\pm$ std across 3 seeds). 50% = balanced game.*

Notable dynamics:
- **R4 Sparse PPO** never learns — win rate stays near 20% throughout training
- **R1 Pursuit PPO** spikes to 80%+ early then stabilizes — strong seeker shaping converges fast
- **SAC curves are noisier** but generally converge to more balanced equilibria (closer to 50%)
- **R5 Escalating** shows the most interesting dynamics — win rate oscillates as agents co-adapt

## Reward Design Lessons

1. **Anti-degenerate shaping matters more than reward magnitude.** The wall proximity penalty $c_{\text{wall}}$ and speed bonus $c_{\text{speed}}$ (R2) prevent corner camping more effectively than increasing other reward terms.

2. **Algorithm choice interacts with reward design.** SAC with sparse rewards ($c = 0$ everywhere) outperforms PPO with dense rewards for evasion — the entropy bonus $\alpha \mathcal{H}[\pi]$ is a form of implicit reward shaping.

3. **Escalating pressure creates drama.** R5's time-scaling $f(t) = 1 + t/T_{\max}$ produces the most dynamic episodes because the seeker's behavior visibly shifts from cautious to aggressive as time runs out.

4. **Kitchen sink is not optimal.** R7 uses every shaping term but doesn't produce the best agents. The reward terms can conflict — the coverage bonus $c_{\text{cov}}$ pulls agents away from each other, undermining pursuit/evasion signals from $c_{\text{dist}}$ and $c_{\Delta d}$.

---

*48 training runs (8 reward presets $\times$ 2 algorithms $\times$ 3 seeds), 5M timesteps each. Cross-evaluated with $16 \times 16 = 256$ matchups $\times$ 20 episodes. Built with custom vectorized NumPy environment and PyTorch PPO/SAC.*

*[View source code](https://github.com/kilojoules/AI-Plays-Tag)*
