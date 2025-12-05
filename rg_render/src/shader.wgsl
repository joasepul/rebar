// Vertex shader

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec2<f32>, // Per-vertex (quad corner)
};

struct InstanceInput {
    @location(1) instance_pos: vec2<f32>,
    @location(2) radius: f32,
    @location(3) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Scale the quad by radius
    let world_pos = instance.instance_pos + model.position * instance.radius;
    
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 0.0, 1.0);
    out.color = instance.color;
    out.uv = model.position; // -1 to 1
    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Circle SDF
    let dist = length(in.uv);
    if (dist > 1.0) {
        discard;
    }
    
    // Simple anti-aliasing
    let alpha = 1.0 - smoothstep(0.9, 1.0, dist);
    
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}

// Edge Shader

struct EdgeInput {
    @location(0) start_pos: vec2<f32>,
    @location(1) end_pos: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) width: f32,
    @builtin(vertex_index) vertex_index: u32,
};

struct EdgeOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_edge(input: EdgeInput) -> EdgeOutput {
    var out: EdgeOutput;
    
    // Calculate line direction and normal
    let dir = input.end_pos - input.start_pos;
    let len = length(dir);
    
    if (len == 0.0) {
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        return out;
    }
    
    let normal = normalize(vec2<f32>(-dir.y, dir.x));
    let width = input.width;
    
    // 6 vertices for a quad (2 triangles)
    // 0: Start - Normal
    // 1: Start + Normal
    // 2: End - Normal
    // 3: End - Normal
    // 4: Start + Normal
    // 5: End + Normal
    
    var pos: vec2<f32>;
    let idx = input.vertex_index % 6u;
    
    if (idx == 0u) {
        pos = input.start_pos - normal * width * 0.5;
    } else if (idx == 1u) {
        pos = input.start_pos + normal * width * 0.5;
    } else if (idx == 2u) {
        pos = input.end_pos - normal * width * 0.5;
    } else if (idx == 3u) {
        pos = input.end_pos - normal * width * 0.5;
    } else if (idx == 4u) {
        pos = input.start_pos + normal * width * 0.5;
    } else {
        pos = input.end_pos + normal * width * 0.5;
    }
    
    out.clip_position = camera.view_proj * vec4<f32>(pos, 0.0, 1.0);
    out.color = input.color;
    return out;
}

@fragment
fn fs_edge(in: EdgeOutput) -> @location(0) vec4<f32> {
    return in.color;
}
