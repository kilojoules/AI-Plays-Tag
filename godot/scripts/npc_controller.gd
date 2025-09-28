extends Node

# Simple NPC brain for the other agent during training.
# Modes:
# - "wander": random walk within arena
# - "chase": move toward target agent

@export_enum("wander", "chase") var mode: String = "wander"
@export var target_path: NodePath
@export var change_dir_interval_ticks: int = 45
@export var arena_half_size: float = 12.0
@export var jump_interval_ticks: int = 180
@export var enable_jump: bool = true
@export var min_ground_ticks_before_jump: int = 45
@export var jump_cooldown_ticks: int = 120

var parent_agent: Node = null
var target: Node3D = null
var tick: int = 0
var dir: Vector2 = Vector2.ZERO
var jump_timer: int = 0
var ground_ticks: int = 0
var jump_cooldown: int = 0
var rng := RandomNumberGenerator.new()

func _ready() -> void:
    parent_agent = get_parent()
    rng.randomize()
    if target_path != NodePath(""):
        target = get_node_or_null(target_path) as Node3D
    if parent_agent:
        parent_agent.control_mode = "ai"
    _pick_random_dir()

func _physics_process(_delta: float) -> void:
    tick += 1
    var desired := Vector2.ZERO
    if mode == "wander":
        if (tick % change_dir_interval_ticks) == 0:
            _pick_random_dir()
        var pos: Vector3 = Vector3.ZERO
        if parent_agent:
            pos = parent_agent.global_transform.origin
        var to_center := Vector2(-pos.x, -pos.z)
        if to_center.length() > 0.001:
            to_center = to_center.normalized()
        desired = (to_center * 0.6 + dir * 0.4).normalized()
    elif mode == "chase" and target:
        var p: Vector3 = parent_agent.global_transform.origin
        var t: Vector3 = target.global_transform.origin
        var v := Vector2(t.x - p.x, t.z - p.z)
        if v.length() > 0.001:
            var base := v.normalized()
            var strafe := Vector2(-base.y, base.x) * rng.randf_range(-0.2, 0.2)
            desired = (base * 0.85 + strafe * 0.15).normalized()
            if v.length() < 3.0:
                desired = (strafe * 0.6 + base * 0.4).normalized()
        var to_center_chase := Vector2(-p.x, -p.z)
        if to_center_chase.length() > 0.001:
            desired = (desired * 0.8 + to_center_chase.normalized() * 0.2).normalized()
    elif mode == "chase" and not target and target_path != NodePath(""):
        target = get_node_or_null(target_path) as Node3D
    if parent_agent:
        parent_agent.ai_move_input = desired.clamp(Vector2(-1,-1), Vector2(1,1))
        _maybe_jump(parent_agent)

func _pick_random_dir() -> void:
    dir = Vector2(rng.randf_range(-1,1), rng.randf_range(-1,1)).normalized()

func _maybe_jump(agent: Node) -> void:
    if not enable_jump:
        return
    if not (agent is CharacterBody3D):
        return
    if jump_interval_ticks <= 0:
        return
    var body := agent as CharacterBody3D
    if body.is_on_floor():
        ground_ticks = min(ground_ticks + 1, 100000)
    else:
        ground_ticks = 0
    if jump_cooldown > 0:
        jump_cooldown -= 1
    if ground_ticks < min_ground_ticks_before_jump:
        jump_timer = max(jump_timer - 1, 0)
        body.ai_jump = 0.0
        return
    if jump_timer > 0:
        jump_timer -= 1
        body.ai_jump = 0.0
        return
    if jump_cooldown > 0:
        body.ai_jump = 0.0
        return
    if body.is_on_floor():
        body.ai_jump = 1.0
        jump_timer = jump_interval_ticks + int(rng.randf_range(-0.3, 0.5) * float(jump_interval_ticks))
        ground_ticks = 0
        jump_cooldown = jump_cooldown_ticks + int(rng.randf_range(0.0, 0.5) * float(jump_cooldown_ticks))

func refresh_behavior() -> void:
    jump_timer = 0
    _pick_random_dir()
    change_dir_interval_ticks = max(20, change_dir_interval_ticks + int(rng.randf_range(-0.5, 0.5) * float(change_dir_interval_ticks)))
    jump_cooldown = jump_cooldown_ticks
    ground_ticks = 0
