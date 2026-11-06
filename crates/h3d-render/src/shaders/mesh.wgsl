// Mesh 3D (ruang R, mm, Z ke atas): 2 lampu arah + ambient belahan langit/
// tanah, alfa per objek, bidang potong, pewarnaan skalar dari colormap
// 256×1 atau warna pulasan verteks. Material: kilau Blinn-Phong, difus
// terbungkus (kesan jaringan), cahaya tepi, oklusi ambien terbakar, dan
// tonjolan "sel" prosedural ILUSTRATIF (Voronoi 3D, hash tetap).
// Material bawaan (semua 0, tanah = ambient) = tampilan teknis lama.

struct Globals {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    // xyz = arah ke lampu (dinormalkan), w = intensitas
    light0: vec4<f32>,
    light1: vec4<f32>,
    // rgb = ambient langit (normal ke +Z)
    ambient: vec4<f32>,
    // Bidang potong: fragmen dengan dot(xyz, world) − w > 0 dibuang.
    // Nonaktif = xyz nol dan w sangat besar.
    clip_plane: vec4<f32>,
    // x = faktor renggang Z (tampilan saja; 1 = skala sebenarnya).
    params: vec4<f32>,
    // rgb = ambient tanah (normal ke −Z)
    ground: vec4<f32>,
};

struct MeshParams {
    color: vec4<f32>,
    // (opasitas, mode skalar > 0.5, skalar min, skalar maks)
    params: vec4<f32>,
    // xyz = geseran tampilan (mm, ruang R, setelah renggang Z), w = sorot 0..1
    offset: vec4<f32>,
    // (kilau, eksponen kilau, difus terbungkus, kekuatan oklusi)
    material: vec4<f32>,
    // rgb = warna rim, w = kekuatan rim
    rim: vec4<f32>,
    // (ukuran sel mm; 0 = mati, kekuatan tonjolan, campuran inti,
    //  w > 0.5 = warna pulasan dengan penguat saturasi w)
    cells: vec4<f32>,
};

// Warna sorot objek terpilih (putih-kuning lembut).
const HIGHLIGHT_COLOR: vec3<f32> = vec3<f32>(1.0, 0.9, 0.4);
// Warna "inti" ilustratif (ungu hematoksilin).
const NUCLEUS_COLOR: vec3<f32> = vec3<f32>(0.36, 0.16, 0.50);

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> mesh: MeshParams;
@group(2) @binding(0) var colormap: texture_2d<f32>;

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) scalar: f32,
    @location(3) color: vec4<f32>,
    @location(4) density: f32,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) scalar: f32,
    @location(3) color: vec4<f32>,
    @location(4) density: f32,
    // Posisi model (mm, sebelum renggang Z dan geseran): sel menempel ke objek.
    @location(5) local: vec3<f32>,
};

// Skalar NODATA dikodekan di CPU sebagai nilai hingga yang sangat besar
// (lihat `mesh::NODATA_SCALAR`), karena uji NaN di WGSL tidak andal.
const NODATA_THRESHOLD: f32 = 1.0e38;

@vertex
fn vs_mesh(in: VIn) -> VOut {
    var out: VOut;
    // Renggang Z mengelilingi pusat volume (ruang R berpusat di sana).
    // Geseran tampilan ditambahkan setelah renggang Z (mengangkat objek).
    let pos = vec3<f32>(in.pos.xy, in.pos.z * globals.params.x) + mesh.offset.xyz;
    out.clip = globals.view_proj * vec4<f32>(pos, 1.0);
    out.world = pos;
    // Normal mengikuti renggang Z (transpose-invers skala diagonal).
    let zs = max(globals.params.x, 1.0e-6);
    out.normal = vec3<f32>(in.normal.xy, in.normal.z / zs);
    out.scalar = in.scalar;
    out.color = in.color;
    out.density = in.density;
    out.local = in.pos;
    return out;
}

// Hash 3D → [0,1)³ (deterministik, tanpa status).
fn hash3(p: vec3<f32>) -> vec3<f32> {
    let q = vec3<f32>(
        dot(p, vec3<f32>(127.1, 311.7, 74.7)),
        dot(p, vec3<f32>(269.5, 183.3, 246.1)),
        dot(p, vec3<f32>(113.5, 271.9, 124.6)),
    );
    return fract(sin(q) * 43758.5453);
}

struct CellHit {
    // Pusat sel terdekat (satuan sel).
    center: vec3<f32>,
    // Jarak ke pusat terdekat dan kedua.
    d1: f32,
    d2: f32,
};

// Voronoi 3D (F1, F2) dengan pusat terjitter di tiap sel kisi.
fn voronoi(p: vec3<f32>) -> CellHit {
    let i = floor(p);
    let f = p - i;
    var best = vec3<f32>(0.0);
    var f1 = 8.0;
    var f2 = 8.0;
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let g = vec3<f32>(f32(x), f32(y), f32(z));
                let o = hash3(i + g) * 0.7 + 0.15;
                let r = g + o - f;
                let d = dot(r, r);
                if d < f1 {
                    f2 = f1;
                    f1 = d;
                    best = i + g + o;
                } else if d < f2 {
                    f2 = d;
                }
            }
        }
    }
    var h: CellHit;
    h.center = best;
    h.d1 = sqrt(f1);
    h.d2 = sqrt(f2);
    return h;
}

@fragment
fn fs_mesh(in: VOut) -> @location(0) vec4<f32> {
    if dot(globals.clip_plane.xyz, in.world) - globals.clip_plane.w > 0.0 {
        discard;
    }
    var base = mesh.color.rgb;
    if mesh.cells.w > 0.5 {
        // Penguat saturasi di sekitar luminans, lalu sedikit digelapkan
        // agar pulasan pucat terbaca sebagai jaringan.
        let c = in.color.rgb;
        let y = dot(c, vec3<f32>(0.299, 0.587, 0.114));
        base = clamp(mix(vec3<f32>(y), c, mesh.cells.w), vec3<f32>(0.0), vec3<f32>(1.0));
        base = pow(base, vec3<f32>(1.25));
    }

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

    // Tonjolan sel ilustratif: normal dimiringkan menjauhi pusat sel (kubah),
    // celah antar sel digelapkan, inti ungu di tengah. Bobot densitas inti
    // (bila ada) memperkuat efek di daerah padat sel.
    var crease = 0.0;
    let cell_mm = mesh.cells.x;
    if cell_mm > 0.0 {
        var w = 1.0;
        if in.density >= 0.0 {
            w = mix(0.3, 1.0, clamp(in.density, 0.0, 1.0));
        }
        let pc = in.local / cell_mm;
        let hit = voronoi(pc);
        let r = pc - hit.center;
        let rt = r - dot(r, n) * n;
        let k = mesh.cells.y * w;
        n = normalize(n + rt * (2.2 * k));
        crease = (1.0 - smoothstep(0.0, 0.18, hit.d2 - hit.d1)) * k;
        let nuc = 1.0 - smoothstep(0.14, 0.26, hit.d1);
        base = mix(base, NUCLEUS_COLOR, clamp(mesh.cells.z * nuc * w, 0.0, 1.0));
    }

    let wrap = clamp(mesh.material.z, 0.0, 1.0);
    let l0 = normalize(globals.light0.xyz);
    let l1 = normalize(globals.light1.xyz);
    let d0 = max((dot(n, l0) + wrap) / (1.0 + wrap), 0.0) * globals.light0.w;
    let d1 = max((dot(n, l1) + wrap) / (1.0 + wrap), 0.0) * globals.light1.w;
    let hemi = mix(globals.ground.rgb, globals.ambient.rgb, 0.5 + 0.5 * n.z);
    let ao = mix(1.0, in.color.a, clamp(mesh.material.w, 0.0, 1.0)) * (1.0 - 0.55 * crease);
    let shade = hemi * ao + vec3<f32>(d0 + d1) * mix(1.0, ao, 0.5);
    var color = base * shade;

    // Kilau Blinn-Phong (lampu utama penuh, pengisi separuh).
    let ks = mesh.material.x;
    if ks > 0.0 {
        let e = max(mesh.material.y, 1.0);
        let s0 = pow(max(dot(n, normalize(l0 + v)), 0.0), e) * globals.light0.w;
        let s1 = pow(max(dot(n, normalize(l1 + v)), 0.0), e) * globals.light1.w * 0.5;
        color += vec3<f32>(ks * (s0 + s1) * ao);
    }
    // Cahaya tepi (kesan tembus/basah pada siluet).
    let fres = pow(1.0 - max(dot(n, v), 0.0), 3.0);
    color += mesh.rim.rgb * (mesh.rim.w * fres) * mix(1.0, ao, 0.5);

    // Sorot: campur ke warna sorot, tambah ambient dan rim di tepi siluet
    // supaya objek terpilih jelas dari segala arah.
    let h = clamp(mesh.offset.w, 0.0, 1.0);
    if h > 0.0 {
        let lit = base * (shade + vec3<f32>(0.25)) + HIGHLIGHT_COLOR * fres * 0.6;
        color = mix(color, mix(lit, HIGHLIGHT_COLOR, 0.35), h);
    }
    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), mesh.params.x);
}
