fn main() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let proto_root = manifest.parent().unwrap().join("proto");
    let proto_file = proto_root.join("inferential.proto");

    if proto_file.exists() {
        prost_build::compile_protos(&[&proto_file], &[&proto_root]).unwrap();
    } else {
        // Fallback for `cargo publish`: proto dir is outside the package tarball.
        // Copy the committed pre-generated bindings into OUT_DIR.
        let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
        std::fs::copy(
            manifest.join("src/inferential_generated.rs"),
            out.join("inferential.rs"),
        )
        .unwrap();
    }
}
