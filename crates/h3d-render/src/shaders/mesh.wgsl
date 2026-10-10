// Mesh 3D (ruang R, mm, Z ke atas): 2 lampu arah + ambient, alfa per
// objek, bidang potong, pewarnaan skalar dari colormap 256×1.

struct Globals {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    // xyz = arah ke lampu (dinormalkan), w = intensitas
    light0: vec4<f32>,
    light1: vec4<f32>,
    // rgb = ambient
    ambient: vec4<f32>,
    // Bidang potong: fragmen dengan dot(xyz, world) − w > 0 dibuang.
    // Nonaktif = xyz nol dan w sangat besar.
    clip_plane: vec4<f32>,
};

struct MeshParams {
    color: vec4<f32>,
    // (opasitas, mode skalar > 0.5, skalar min, skalar maks)
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> mesh: MeshParams;
@group(2) @binding(0) var colormap: texture_2d<f32>;

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) scalar: f32,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) scalar: f32,
};

// Skalar NODATA dikodekan di CPU sebagai nilai hingga yang sangat besar
// (lihat `mesh::NODATA_SCALAR`), karena uji NaN di WGSL tidak andal.
const NODATA_THRESHOLD: f32 = 1.0e38;

@vertex
fn vs_mesh(in: VIn) -> VOut {
    var out: VOut;
    out.clip = globals.view_proj * vec4<f32>(in.pos, 1.0);
    out.world = in.pos;
    out.normal = in.normal;
    out.scalar = in.scalar;
    return out;
}

@fragment
fn fs_mesh(in: VOut) -> @location(0) vec4<f32> {
    if dot(globals.clip_plane.xyz, in.world) - globals.clip_plane.w > 0.0 {
        discard;
    }
    var base = mesh.color.rgb;
    if mesh.params.y > 0.5 {
        if in.scalar > NODATA_THRESHOLD {
            base = vec3<f32>(0.5, 0.5, 0.5);
        } else {
            let range = max(mesh.params.w - mesh.params.z, 1.0e-30);
            let t = clamp((in.scalar - mesh.params.z) / range, 0.0, 1.0);
            base = textureLoad(colormap, vec2<i32>(i32(t * 255.0 + 0.5), 0), 0).rgb;
        }
    }
    var n = normalize(in.normal);
    let v = normalize(globals.eye.xyz - in.world);
    // Pencahayaan dua sisi: winding tidak diandalkan (pembalikan y mencerminkan mesh).
    if dot(n, v) < 0.0 {
        n = -n;
    }
    let d0 = max(dot(n, normalize(globals.light0.xyz)), 0.0) * globals.light0.w;
    let d1 = max(dot(n, normalize(globals.light1.xyz)), 0.0) * globals.light1.w;
    let shade = globals.ambient.rgb + vec3<f32>(d0 + d1);
    return vec4<f32>(clamp(base * shade, vec3<f32>(0.0), vec3<f32>(1.0)), mesh.params.x);
}
