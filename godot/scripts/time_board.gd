extends Label3D

func _process(_delta: float) -> void:
    var gm := _get_game_manager()
    if gm and ("time_left" in gm):
        var tl: float = float(gm.time_left)
        text = "Time: %.1fs" % tl

func _get_game_manager() -> Node:
    var nodes := get_tree().get_nodes_in_group("game_manager")
    if nodes.size() > 0:
        return nodes[0]
    return null
