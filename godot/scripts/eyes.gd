extends Node3D

@export var pupil_offset: float = 0.18
@export var eye_separation: float = 0.28
@export var look_smooth: float = 10.0

var target_dir: Vector3 = Vector3.FORWARD
var cur_dir: Vector3 = Vector3.FORWARD

func set_look_direction(dir: Vector3) -> void:
    if dir.length() > 0.001:
        target_dir = dir.normalized()

func _process(delta: float) -> void:
    cur_dir = cur_dir.lerp(target_dir.normalized(), clamp(look_smooth * delta, 0.0, 1.0))
    _apply_to_pupils()

func _apply_to_pupils() -> void:
    var left := $LeftPupil as Node3D
    var right := $RightPupil as Node3D
    if left and right:
        var lateral := Vector3(eye_separation * 0.5, 0, 0)
        left.transform.origin = Vector3(-eye_separation * 0.5, 0.0, 0.0) + cur_dir * pupil_offset
        right.transform.origin = Vector3(eye_separation * 0.5, 0.0, 0.0) + cur_dir * pupil_offset

