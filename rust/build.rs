fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("proto");
    prost_build::compile_protos(&[root.join("inferential.proto")], &[&root]).unwrap();
}
