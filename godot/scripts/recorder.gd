extends Node
const DataPaths = preload("res://scripts/data_paths.gd")

@export var enabled: bool = false
@export var fps: int = 60
var i := 0
var acc := 0.0

func _ready() -> void:
    _apply_env_overrides()
    if enabled:
        DataPaths.frames_dir()

func _process(delta: float) -> void:
    if not enabled:
        return
    # Skip recording if in headless mode (no rendering)
    if DisplayServer.get_name().to_lower() == "headless":
        return
    acc += delta
    var frame_time := 1.0 / float(max(1, fps))
    while acc >= frame_time:
        acc -= frame_time
        var tex := get_viewport().get_texture()
        if tex == null:
            return
        var img: Image = tex.get_image()
        if img == null:
            return
        var path := DataPaths.frames_dir().path_join("frame_%05d.png" % i)
        img.save_png(path)
        i += 1

func _apply_env_overrides() -> void:
    var v
    v = OS.get_environment("AI_RECORD")
    if v != "":
        var low = v.to_lower()
        enabled = (low == "1" or low == "true" or low == "yes")
    v = OS.get_environment("AI_RECORD_FPS")
    if v != "":
        fps = int(v)
