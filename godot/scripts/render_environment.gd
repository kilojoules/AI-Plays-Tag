extends Node

class_name RenderEnvironment

static func is_headless() -> bool:
    if OS.has_feature("headless"):
        return true
    var name := DisplayServer.get_name().to_lower()
    return name == "headless" or name == "dummy"
