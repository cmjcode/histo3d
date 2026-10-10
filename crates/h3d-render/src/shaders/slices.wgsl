// Irisan di 3D: quad bertekstur thumbnail. `model` memetakan kotak satuan
// (u, v ∈ 0..1) → ruang R (mm); dibangun di CPU dari T_i, ukuran slide, z_i
// dan konversi µm → mm (lihat `slices3d::slice_model_matrix`).

struct Globals {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    light0: vec4<f32>,
    light1: vec4<f32>,
    ambient: vec4<f32>,
    clip_plane: vec4<f32>,
};

struct SliceParams {
    model: mat4x4<f32>,
    // x = alfa
    alpha: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> slice: SliceParams;
@group(2) @binding(0) var slice_tex: texture_2d<f32>;
@group(2) @binding(1) var slice_samp: sampler;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world: vec3<f32>,
};

fn corner(i: u32) -> vec2<f32> {
    switch i {
        case 0u: { return vec2<f32>(0.0, 0.0); }
        case 1u: { return vec2<f32>(1.0, 0.0); }
        case 2u: { return vec2<f32>(1.0, 1.0); }
        case 3u: { return vec2<f32>(0.0, 0.0); }
        case 4u: { return vec2<f32>(1.0, 1.0); }
        default: { return vec2<f32>(0.0, 1.0); }
    }
}

@vertex
fn vs_slice(@builtin(vertex_index) vi: u32) -> VOut {
    let c = corner(vi);
    let world = slice.model * vec4<f32>(c, 0.0, 1.0);
    var out: VOut;
    out.clip = globals.view_proj * world;
    out.uv = c;
    out.world = world.xyz;
    return out;
}

@fragment
fn fs_slice(in: VOut) -> @location(0) vec4<f32> {
    if dot(globals.clip_plane.xyz, in.world) - globals.clip_plane.w > 0.0 {
        discard;
    }
    let c = textureSample(slice_tex, slice_samp, in.uv);
    return vec4<f32>(c.rgb, c.a * slice.alpha.x);
}
