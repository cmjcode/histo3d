// Tile 2D: satu quad per tile (instanced), tekstur Rgba8UnormSrgb.
//
// Alur koordinat: rect tile (µm ruang U irisan) → affine T_i (U → W, µm)
// → kurangi pusat pandangan, bagi µm/px → NDC (y dibalik: layar y ke
// bawah, NDC y ke atas).

struct Layer {
    // Baris 0 dari T_i: x' = a·x + b·y + tx  →  (a, b, tx, 0)
    affine0: vec4<f32>,
    // Baris 1 dari T_i: y' = c·x + d·y + ty  →  (c, d, ty, 0)
    affine1: vec4<f32>,
    // (center_x µm, center_y µm, µm per point, 0)
    view: vec4<f32>,
    // (lebar viewport point, tinggi viewport point, opasitas, mode)
    viewport: vec4<f32>,
    // (ukuran kotak papan catur dalam piksel fisik, viewport_x px, viewport_y px, 0)
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> layer: Layer;
@group(1) @binding(0) var tile_tex: texture_2d<f32>;
@group(1) @binding(1) var tile_samp: sampler;

struct Inst {
    // (x0, y0, x1, y1) µm ruang U
    @location(0) rect: vec4<f32>,
    // (u0, v0, u1, v1)
    @location(1) uv: vec4<f32>,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
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

fn layer_to_ndc(p: vec2<f32>) -> vec4<f32> {
    let w = vec2<f32>(
        layer.affine0.x * p.x + layer.affine0.y * p.y + layer.affine0.z,
        layer.affine1.x * p.x + layer.affine1.y * p.y + layer.affine1.z,
    );
    let px = (w - layer.view.xy) / layer.view.z;
    let ndc = vec2<f32>(px.x / (layer.viewport.x * 0.5), -px.y / (layer.viewport.y * 0.5));
    return vec4<f32>(ndc, 0.0, 1.0);
}

@vertex
fn vs_tile(@builtin(vertex_index) vi: u32, inst: Inst) -> VOut {
    let c = corner(vi);
    var out: VOut;
    let p = mix(inst.rect.xy, inst.rect.zw, c);
    out.clip = layer_to_ndc(p);
    out.uv = mix(inst.uv.xy, inst.uv.zw, c);
    return out;
}

// Quad penuh viewport (mode kotak hitam), tanpa tekstur.
@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VOut {
    let c = corner(vi);
    var out: VOut;
    out.clip = vec4<f32>(c * 2.0 - 1.0, 0.0, 1.0);
    out.uv = c;
    return out;
}

fn checker_parity(frag_px: vec2<f32>) -> u32 {
    let size = max(layer.params.x, 1.0);
    let cell = floor((frag_px - layer.params.yz) / size);
    return u32(abs(cell.x + cell.y)) % 2u;
}

@fragment
fn fs_tile(in: VOut) -> @location(0) vec4<f32> {
    let mode = u32(layer.viewport.w + 0.5);
    let opacity = layer.viewport.z;
    if mode == 1u || mode == 2u {
        if checker_parity(in.clip.xy) != (mode - 1u) {
            discard;
        }
    }
    let c = textureSample(tile_tex, tile_samp, in.uv);
    if mode == 3u || mode == 4u {
        // Kegelapan (1 − luminansi): jaringan terang, latar kaca gelap.
        let lum = dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
        let d = clamp(1.0 - lum, 0.0, 1.0) * c.a * opacity;
        if mode == 3u {
            return vec4<f32>(d, 0.0, d, 1.0);
        }
        return vec4<f32>(0.0, d, 0.0, 1.0);
    }
    return vec4<f32>(c.rgb, c.a * opacity);
}

@fragment
fn fs_solid(in: VOut) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
