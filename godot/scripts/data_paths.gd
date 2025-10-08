class_name DataPaths

static var _data_root := ""
static var _trajectories_dir := ""
static var _frames_dir := ""

static func data_root() -> String:
    if _data_root != "":
        return _data_root
    var env := OS.get_environment("AI_DATA_ROOT").strip_edges()
    if env != "":
        if env.begins_with("res://") or env.begins_with("user://"):
            _data_root = ProjectSettings.globalize_path(env)
        else:
            _data_root = env
        return _data_root
    var repo_root := ProjectSettings.globalize_path("res://").get_base_dir()
    _data_root = repo_root.path_join("data")
    return _data_root

static func _ensure_dir(path: String) -> void:
    if DirAccess.dir_exists_absolute(path):
        return
    var err := DirAccess.make_dir_recursive_absolute(path)
    if err != OK:
        push_error("Failed to ensure directory %s (err=%s)" % [path, err])

static func trajectories_dir() -> String:
    if _trajectories_dir == "":
        _trajectories_dir = data_root().path_join("trajectories")
    _ensure_dir(_trajectories_dir)
    return _trajectories_dir

static func frames_dir() -> String:
    if _frames_dir == "":
        _frames_dir = data_root().path_join("frames")
    _ensure_dir(_frames_dir)
    return _frames_dir

static func trajectory_file(basename: String) -> String:
    return trajectories_dir().path_join(basename)

static func frames_pattern() -> String:
    return frames_dir().path_join("frame_%05d.png")

static func path_separator() -> String:
    var sep := OS.get_environment("AI_PATHSEP")
    if sep == "":
        return ":"
    return sep

static func _legacy_dirs(env_key: String) -> PackedStringArray:
    var result := PackedStringArray()
    var raw := OS.get_environment(env_key)
    if raw == "":
        return result
    var sep := path_separator()
    for entry in raw.split(sep):
        var trimmed := entry.strip_edges()
        if trimmed != "":
            result.append(trimmed)
    return result

static func legacy_trajectory_dirs() -> PackedStringArray:
    return _legacy_dirs("AI_LEGACY_TRAJECTORY_DIRS")

static func legacy_frames_dirs() -> PackedStringArray:
    return _legacy_dirs("AI_LEGACY_FRAMES_DIRS")
