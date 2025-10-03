extends Node3D

@export var follow_distance: float = 6.0
@export var follow_height: float = 3.0
@export var follow_smooth: float = 8.0
@export var yaw_degrees: float = 0.0
@export var fixed_camera: bool = true
@export var view_target: Vector3 = Vector3.ZERO
@export var base_fov: float = 60.0
@export var action_zoom_factor: float = 0.8
@export var action_threshold: float = 3.0
@export var zoom_smooth: float = 6.0
@export var focus_group: StringName = ""
@export var baseline_pair_distance: float = 8.0
@export var follow_distance_compensation: float = 0.35
@export var min_follow_distance: float = 5.0
@export var max_follow_distance: float = 24.0
@export var fov_compensation: float = 1.0
@export var min_fov: float = 45.0
@export var max_fov: float = 75.0

var target: Node3D = null
@onready var spring: SpringArm3D = $SpringArm3D
@onready var cam: Camera3D = $SpringArm3D/Camera3D

func set_target(node: Node) -> void:
    if node is Node3D:
        target = node

func _process(delta: float) -> void:
    var pair_distance: float = _agents_distance()
    if fixed_camera:
        # Hold position; keep looking at the configured target.
        look_at(view_target, Vector3.UP)
        # Smooth zoom based on agent proximity
        var target_fov: float = _target_fov_for_distance(pair_distance)
        if pair_distance > 0.0 and pair_distance <= action_threshold:
            target_fov = min(target_fov, base_fov * action_zoom_factor)
        _apply_fov(delta, target_fov)
        return
    if not target:
        var focus := _compute_focus_point()
        if focus == null:
            return
        var basis := Basis(Vector3.UP, deg_to_rad(yaw_degrees))
        var desired_offset := basis * _compute_follow_offset(pair_distance)
        var desired_pos := focus + desired_offset
        global_transform.origin = global_transform.origin.lerp(desired_pos, clamp(follow_smooth * delta, 0.0, 1.0))
        look_at(focus, Vector3.UP)
        _apply_fov(delta, _target_fov_for_distance(pair_distance))
        return
    var focus_point := _compute_focus_point(target.global_transform.origin)
    var basis := Basis(Vector3.UP, deg_to_rad(yaw_degrees))
    var desired_offset := basis * _compute_follow_offset(pair_distance)
    var desired_pos := focus_point + desired_offset
    global_transform.origin = global_transform.origin.lerp(desired_pos, clamp(follow_smooth * delta, 0.0, 1.0))
    look_at(focus_point + Vector3(0, 1.5, 0), Vector3.UP)
    _apply_fov(delta, _target_fov_for_distance(pair_distance))

func _agents_distance() -> float:
    var nodes := get_tree().get_nodes_in_group("agents")
    if nodes.size() >= 2:
        var a: Node3D = nodes[0]
        var b: Node3D = nodes[1]
        return a.global_transform.origin.distance_to(b.global_transform.origin)
    return -1.0

func _compute_follow_offset(pair_distance: float) -> Vector3:
    var distance := follow_distance
    if pair_distance > 0.0:
        distance += (pair_distance - baseline_pair_distance) * follow_distance_compensation
        distance = clamp(distance, min_follow_distance, max_follow_distance)
    return Vector3(0, follow_height, distance)

func _target_fov_for_distance(pair_distance: float) -> float:
    var fov := base_fov
    if pair_distance > 0.0:
        fov += (pair_distance - baseline_pair_distance) * fov_compensation
    return clamp(fov, min_fov, max_fov)

func _apply_fov(delta: float, target_fov: float) -> void:
    cam.fov = lerp(cam.fov, target_fov, clamp(zoom_smooth * delta, 0.0, 1.0))

func _compute_focus_point(default_pos: Vector3 = Vector3.ZERO) -> Vector3:
    if focus_group != "":
        var nodes := get_tree().get_nodes_in_group(focus_group)
        if nodes.size() > 0:
            var accum := Vector3.ZERO
            var count := 0
            for n in nodes:
                if n is Node3D:
                    accum += (n as Node3D).global_transform.origin
                    count += 1
            if count > 0:
                return accum / count
    if target:
        return target.global_transform.origin
    return default_pos
