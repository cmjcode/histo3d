// Irisan di 3D: quad bertekstur thumbnail. `model` memetakan kotak satuan
// (u, v ∈ 0..1) → ruang R (mm); dibangun di CPU dari T_i, ukuran slide, z_i
// dan konversi µm → mm (lihat `slices3d::slice_model_matrix`).
// Kaca opsional (`alpha.y` > 0): bingkai terang di tepi quad dan kaca tipis
// kebiruan di latar transparan, dikomposit di bawah jaringan.

struct Globals {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    light0: vec4<f32>,
    light1: vec4<f32>,
    ambient: vec4<f32>,
    clip_plane: vec4<f32>,
    // x = faktor renggang Z (tampilan saja; 1 = skala sebenarnya).
    params: vec4<f32>,
    ground: vec4<f32>,
};

struct SliceParams {
    model: mat4x4<f32>,
    // x = alfa, y = kekuatan kaca 0..1
    alpha: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> slice: SliceParams;
@group(2) @binding(0) var slice_tex: texture_2d<f32>;
@group(2) @binding(1) var slice_samp: sampler;

// Warna kaca/bingkai (biru-sian pucat).
const GLASS_COLOR: vec3<f32> = vec3<f32>(0.72, 0.90, 1.0);

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
    var world = slice.model * vec4<f32>(c, 0.0, 1.0);
    world.z = world.z * globals.params.x;
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
    let a_t = c.a * slice.alpha.x;
    let glass = clamp(slice.alpha.y, 0.0, 1.0);
    if glass <= 0.0 {
        return vec4<f32>(c.rgb, a_t);
    }
    // Lebar bingkai dalam uv, tetap tipis walau quad diperbesar.
    let e = min(min(in.uv.x, 1.0 - in.uv.x), min(in.uv.y, 1.0 - in.uv.y));
    let fw = max(fwidth(e), 1.0e-5);
    let frame = 1.0 - smoothstep(1.5 * fw, 3.5 * fw, e);
    let a_g = glass * (0.035 + 0.65 * frame);

    // Jaringan di atas kaca (komposit "over"), keluaran alfa non-premultiplied.
    let a = a_t + a_g * (1.0 - a_t);
    let rgb = (c.rgb * a_t + GLASS_COLOR * a_g * (1.0 - a_t)) / max(a, 1.0e-4);
    return vec4<f32>(mix(rgb, GLASS_COLOR, frame * glass * 0.6), a);
}
