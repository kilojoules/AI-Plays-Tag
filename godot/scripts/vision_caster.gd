extends Node3D

@export var rays: int = 36
@export var fov_degrees: float = 120.0
@export var max_distance: float = 15.0
@export var debug_draw_in_world: bool = false

var distances: PackedFloat32Array = PackedFloat32Array()
var types: Array = []  # "none", "wall", "agent"

func _ready() -> void:
    distances.resize(rays)
    types.resize(rays)

func sample(space: PhysicsDirectSpaceState3D) -> void:
    var origin: Vector3 = global_transform.origin + Vector3(0, 0.5, 0)
    var forward := -global_transform.basis.z
    var left_axis := global_transform.basis.x
    var up := Vector3.UP
    var half := fov_degrees * 0.5
    for i in range(rays):
        var t := float(i) / float(max(1, rays - 1))
        var angle := deg_to_rad(-half + t * fov_degrees)
        var dir := (Basis(up, angle) * forward).normalized()
        var to: Vector3 = origin + dir * max_distance
        var query := PhysicsRayQueryParameters3D.create(origin, to)
        query.exclude = [get_parent()]  # exclude self
        var hit := space.intersect_ray(query)
        if hit.size() > 0:
            var pos: Vector3 = hit["position"]
            var obj: Object = hit["collider"]
            distances[i] = origin.distance_to(pos)
            if obj and obj is Node and (obj as Node).is_in_group("agents"):
                types[i] = "agent"
            else:
                types[i] = "wall"
        else:
            distances[i] = max_distance
            types[i] = "none"
        if debug_draw_in_world:
            _debug_draw_ray(origin, dir, distances[i])

func _physics_process(_delta: float) -> void:
    var space := get_world_3d().direct_space_state
    if space:
        sample(space)

func get_distances() -> PackedFloat32Array:
    return distances

func get_types() -> Array:
    return types

func _debug_draw_ray(origin: Vector3, dir: Vector3, dist: float) -> void:
    # Use debug shapes if available (optional)
    pass

