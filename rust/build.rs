fn main() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    // Prefer the workspace proto; fall back to the CI-copied local copy.
    let root = if manifest
        .parent()
        .unwrap()
        .join("proto/inferential.proto")
        .exists()
    {
        manifest.parent().unwrap().join("proto")
    } else {
        manifest.join(".proto")
    };
    prost_build::compile_protos(&[root.join("inferential.proto")], &[&root]).unwrap();
}
