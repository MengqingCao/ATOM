---
repo: MengqingCao/ATOM
remote: fork
integration_branch: agent/prefill-all-paths
never_touch: ROCm/ATOM
design_entry: atom/mesh/README.md
gate_targeted: cd atom/mesh && cargo fmt --check && cargo test --lib --features atom-engine -- --skip test_load_counter_performance && cargo check --features ext-proc --tests
gate_task: cd atom/mesh && cargo fmt --check && cargo test --lib --features atom-engine -- --skip test_load_counter_performance && cargo check --features ext-proc --tests && cargo test --tests -- --skip test_load_counter_performance && cargo test --manifest-path mocker/Cargo.toml --target-dir mocker/target/mocker --release test_atomesh_harness && CARGO_TARGET_DIR=target_194 cargo +1.94.0 test --lib -- --skip test_load_counter_performance
gate_post_land: for t in test_atomesh_v4_routing test_atomesh_lmcache_disk test_atomesh_multinode_layout test_atomesh_mooncake_store test_atomesh_moriio test_atomesh_v4_workflow; do TMPDIR=/tmp PYTHONPATH=tests timeout 900 python3 -m unittest tests.$t || exit 1; done
publish_language: English
max_tasks: 2
split_loc: 1000
---
Mesh (Rust router) work in `atom/mesh`. PRs at most 1000 lines and 15 files.
The deliverable is upstreamed by the owner as a gh-stack PR on ROCm/ATOM above #3568;
agents never write to ROCm/ATOM. Do not commit `target*/` build directories.
