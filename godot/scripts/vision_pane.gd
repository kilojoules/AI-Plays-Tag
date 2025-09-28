extends Control

@export var agent_name: StringName = ""

func _process(_delta: float) -> void:
    queue_redraw()

func _draw() -> void:
    draw_rect(Rect2(Vector2.ZERO, size), Color(0,0,0,0.5))
    var a := _find_agent()
    if a == null:
        return
    var vc := a.get_node_or_null("VisionCaster")
    if vc == null:
        return
    var distances: PackedFloat32Array = vc.call("get_distances")
    var types: Array = vc.call("get_types")
    if distances == null:
        return
    var n: int = distances.size()
    var cx: float = size.x * 0.5
    var cy: float = size.y * 0.9
    var rmax: float = min(size.x, size.y) * 0.45
    var md: float = 15.0
    if vc and vc.has_method("get"):
        var v = vc.get("max_distance")
        if typeof(v) == TYPE_FLOAT or typeof(v) == TYPE_INT:
            md = float(v)
    for i in range(n):
        var tval: float = float(i) / float(max(1, n - 1))
        var angle: float = deg_to_rad(-60.0 + tval * 120.0)
        var norm: float = float(distances[i]) / md
        norm = clamp(norm, 0.0, 1.0)
        var length: float = rmax * (1.0 - norm)
        var col: Color = Color(0.2, 0.8, 1.0)
        if types and i < types.size():
            var ty: String = str(types[i])
            if ty == "agent": col = Color(1.0, 0.6, 0.2)
            elif ty == "wall": col = Color(0.6, 0.6, 0.6)
        var p1: Vector2 = Vector2(cx, cy)
        var p2: Vector2 = p1 + Vector2(sin(angle), -cos(angle)) * length
        draw_line(p1, p2, col, 2.0)

func _find_agent() -> Node:
    if String(agent_name) == "":
        return null
    var nodes := get_tree().get_nodes_in_group("agents")
    for n in nodes:
        if n.name == agent_name:
            return n
    return null

