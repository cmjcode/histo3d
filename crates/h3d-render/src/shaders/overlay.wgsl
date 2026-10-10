// Overlay label: tile R8Uint + palet 256×1 RGBA. Mode (Layer.viewport.w):
// 0 = isi penuh, 1 = garis tepi (bandingkan dengan 4 tetangga texel).
// Catatan: tetangga di luar tile di-clamp ke tepi tile, jadi tepi label yang
// tepat berimpit dengan batas tile tidak terdeteksi (dapat diterima).

struct Layer {
    affine0: vec4<f32>,
    affine1: vec4<f32>,
    view: vec4<f32>,
    viewport: vec4<f32>,
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> layer: Layer;
@group(1) @binding(0) var label_tex: texture_2d<u32>;
@group(2) @binding(0) var palette: texture_2d<f32>;

struct Inst {
    @location(0) rect: vec4<f32>,
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
fn vs_overlay(@builtin(vertex_index) vi: u32, inst: Inst) -> VOut {
    let c = corner(vi);
    var out: VOut;
    out.clip = layer_to_ndc(mix(inst.rect.xy, inst.rect.zw, c));
    out.uv = mix(inst.uv.xy, inst.uv.zw, c);
    return out;
}

fn label_at(tc: vec2<i32>, dim: vec2<i32>) -> u32 {
    let t = clamp(tc, vec2<i32>(0, 0), dim - vec2<i32>(1, 1));
    return textureLoad(label_tex, t, 0).r;
}

@fragment
fn fs_overlay(in: VOut) -> @location(0) vec4<f32> {
    let dim = vec2<i32>(textureDimensions(label_tex));
    let tc = vec2<i32>(floor(in.uv * vec2<f32>(dim)));
    let id = label_at(tc, dim);
    let mode = u32(layer.viewport.w + 0.5);
    if mode == 1u {
        let same = label_at(tc + vec2<i32>(1, 0), dim) == id
            && label_at(tc - vec2<i32>(1, 0), dim) == id
            && label_at(tc + vec2<i32>(0, 1), dim) == id
            && label_at(tc - vec2<i32>(0, 1), dim) == id;
        if same {
            discard;
        }
    }
    let color = textureLoad(palette, vec2<i32>(i32(id & 255u), 0), 0);
    let a = color.a * layer.viewport.z;
    if a <= 0.0 {
        discard;
    }
    return vec4<f32>(color.rgb, a);
}
