// Grid tak terbatas di bidang horizontal z = konstan (ruang R, mm, Z ke atas).
//
// Satu segitiga layar penuh; tiap fragmen menembakkan sinar dari bidang dekat
// ke bidang jauh (`inv_view_proj`, kedalaman 0..1) lalu memotongnya dengan
// bidang grid. Jarak garis dipilih per fragmen dari ukuran piksel di bidang
// (`fwidth`), berjenjang kelipatan 10 dengan peralihan halus antar jenjang,
// sehingga grid tampak tak terbatas tanpa moiré. Kedalaman fragmen ditulis
// ulang agar mesh di depan/belakang bidang tertutup dengan benar.

struct GridParams {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    // xyz = mata (mm), w = jarak kamera ke target (mm)
    eye: vec4<f32>,
    // x = tinggi bidang z (mm, ruang berenggang), y = jarak garis terkecil
    // dalam piksel, z = faktor jarak pudar (× jarak kamera), w = tak dipakai
    params: vec4<f32>,
    minor: vec4<f32>,
    major: vec4<f32>,
    axis_x: vec4<f32>,
    axis_y: vec4<f32>,
};

@group(0) @binding(0) var<uniform> grid: GridParams;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_grid(@builtin(vertex_index) vi: u32) -> VOut {
    // Segitiga (-1,-1), (3,-1), (-1,3) menutupi seluruh viewport.
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vi & 2u) * 2.0 - 1.0;
    var out: VOut;
    out.clip = vec4<f32>(x, y, 0.0, 1.0);
    out.ndc = vec2<f32>(x, y);
    return out;
}

fn unproject(ndc: vec2<f32>, depth: f32) -> vec3<f32> {
    let h = grid.inv_view_proj * vec4<f32>(ndc, depth, 1.0);
    return h.xyz / h.w;
}

// Kekuatan garis (0..1) berjarak `s` mm di titik `p`; `fw` = mm per piksel.
// Lebar garis ±1 piksel, tepi dihaluskan.
fn line_strength(p: vec2<f32>, s: f32, fw: vec2<f32>) -> f32 {
    let d = abs(fract(p / s + 0.5) - 0.5) * s / max(fw, vec2<f32>(1.0e-12));
    return 1.0 - min(min(d.x, d.y), 1.0);
}

// Bobot jenjang garis: x = posisi relatif jenjang (0 = baru muncul, ≥ 2 =
// mayor). Kontinu saat jenjang bergeser satu kelipatan 10.
fn level_alpha(x: f32) -> f32 {
    let minor = grid.minor.a;
    let major = grid.major.a;
    if x < 1.0 {
        return minor * clamp(x, 0.0, 1.0);
    }
    return mix(minor, major, clamp(x - 1.0, 0.0, 1.0));
}

struct FOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_grid(in: VOut) -> FOut {
    let near = unproject(in.ndc, 0.0);
    let far = unproject(in.ndc, 1.0);
    let dir = far - near;
    let z0 = grid.params.x;
    // Sinar sejajar bidang tidak memotong; dijaga agar tidak membagi nol.
    let dz = select(dir.z, 1.0e-12, abs(dir.z) < 1.0e-12);
    let t = (z0 - near.z) / dz;
    let hit = near + t * dir;
    let p = hit.xy;

    // Turunan dihitung di aliran seragam (sebelum keputusan buang).
    let fw = fwidth(p);
    let min_px = max(grid.params.y, 1.0);
    let lod = log2(max(max(fw.x, fw.y) * min_px, 1.0e-12)) / log2(10.0);
    let f = fract(lod);
    let s0 = pow(10.0, floor(lod));
    let s1 = s0 * 10.0;
    let s2 = s1 * 10.0;

    let a0 = line_strength(p, s0, fw) * level_alpha(1.0 - f);
    let a1 = line_strength(p, s1, fw) * level_alpha(2.0 - f);
    let a2 = line_strength(p, s2, fw) * level_alpha(3.0 - f);
    let a = max(a0, max(a1, a2));
    // Warna bergeser dari minor ke mayor sesuai kekuatan jenjang dominan.
    let k = clamp((a - grid.minor.a) / max(grid.major.a - grid.minor.a, 1.0e-6), 0.0, 1.0);
    var rgb = mix(grid.minor.rgb, grid.major.rgb, k);
    var alpha = a;

    // Sumbu: garis y = 0 (sumbu X) dan x = 0 (sumbu Y), ±1,5 piksel.
    let ax = (1.0 - min(abs(p.y) / max(fw.y, 1.0e-12) / 1.5, 1.0)) * grid.axis_x.a;
    let ay = (1.0 - min(abs(p.x) / max(fw.x, 1.0e-12) / 1.5, 1.0)) * grid.axis_y.a;
    if ax > alpha {
        rgb = grid.axis_x.rgb;
        alpha = ax;
    }
    if ay > alpha {
        rgb = grid.axis_y.rgb;
        alpha = ay;
    }

    // Pudar menjauhi kamera dan pada sudut pandang landai (cakrawala).
    let view_dir = normalize(dir);
    let grazing = smoothstep(0.0, 0.12, abs(view_dir.z));
    let dist = length(hit - grid.eye.xyz);
    let fade_end = max(grid.eye.w * grid.params.z, 1.0e-6);
    let distance_fade = 1.0 - smoothstep(fade_end * 0.25, fade_end, dist);
    alpha = alpha * grazing * distance_fade;

    let clip = grid.view_proj * vec4<f32>(hit, 1.0);
    let depth = clip.z / clip.w;
    // Di belakang mata, di luar bidang jauh, atau tak terlihat → buang.
    if t < 0.0 || t > 1.0 || alpha <= 0.002 || depth < 0.0 || depth > 1.0 {
        discard;
    }
    var out: FOut;
    out.color = vec4<f32>(rgb, alpha);
    out.depth = depth;
    return out;
}
