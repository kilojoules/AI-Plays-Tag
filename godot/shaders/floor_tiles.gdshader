shader_type spatial;
render_mode cull_back, specular_schlick_ggx;

uniform vec3 color_a : source_color = vec3(0.12, 0.14, 0.18);
uniform vec3 color_b : source_color = vec3(0.16, 0.19, 0.24);
uniform vec3 grout_color : source_color = vec3(0.05, 0.06, 0.08);
uniform float tile_scale : hint_range(0.5, 10.0) = 3.5;
uniform float grout_thickness : hint_range(0.001, 0.08) = 0.02;
uniform float micro_detail_scale : hint_range(1.0, 120.0) = 40.0;
uniform float micro_detail_strength : hint_range(0.0, 0.1) = 0.025;
uniform float gloss_shift : hint_range(0.0, 1.0) = 0.25;
uniform float metallic_amount : hint_range(0.0, 1.0) = 0.05;

float hash(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p.x + p.y) * 43758.5453123);
}

float tile_height(vec2 uv) {
    vec2 f = fract(uv) - 0.5;
    float grout = smoothstep(0.5 - grout_thickness, 0.5, max(abs(f.x), abs(f.y)));
    float micro = (hash(floor(uv * micro_detail_scale)) - 0.5) * micro_detail_strength;
    return -grout * 0.02 + micro;
}

void fragment() {
    vec2 uv = UV * tile_scale;
    vec2 cell = floor(uv);
    float random = hash(cell);
    float blend = smoothstep(0.0, 1.0, random);
    vec3 base = mix(color_a, color_b, blend);

    vec2 f = abs(fract(uv) - 0.5);
    float grout = smoothstep(0.5 - grout_thickness * 0.5, 0.5, max(f.x, f.y));
    grout = pow(grout, 6.0);
    float grout_mask = 1.0 - grout;

    vec3 tile_color = mix(color_a, color_b, random * 0.7 + 0.15);
    vec3 albedo = mix(tile_color, grout_color, grout);

    ALBEDO = albedo;
    METALLIC = metallic_amount * grout_mask;
    ROUGHNESS = clamp(0.3 + gloss_shift * random * grout_mask, 0.1, 0.9);

    vec2 eps = vec2(0.002, 0.0);
    float h = tile_height(uv);
    float hx = tile_height(uv + vec2(eps.x, 0.0));
    float hy = tile_height(uv + vec2(0.0, eps.x));
    vec3 n = normalize(vec3(h - hx, 0.25, h - hy));
    NORMAL = normalize(mix(vec3(0.0, 1.0, 0.0), n, 0.9));

    float seam_glow = smoothstep(0.5 - grout_thickness, 0.5, max(f.x, f.y));
    seam_glow = pow(seam_glow, 4.0);
    SPECULAR = 0.35;
    EMISSION = mix(vec3(0.0), vec3(0.08, 0.18, 0.3), seam_glow * 0.5);
}
