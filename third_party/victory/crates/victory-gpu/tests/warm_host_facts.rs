//! Offline cache warming (`victory translate --warm`) sets `VICTORY_HOST_WAVE`
//! and `VICTORY_VIEW_LIMIT` so a cache can be warmed for another host GPU
//! without touching the local adapter. `shader_cache::warm` must not clobber
//! those explicit host facts by probing the local D3D12 device.

#![cfg(windows)]

#[test]
fn warm_preserves_explicit_host_wave_and_view_limit() {
    let dir = tempfile::tempdir().expect("tempdir");
    victory_gpu::shader_cache::use_shader_cache(dir.path().to_path_buf());

    // 1. Both host facts explicitly supplied (e.g. warming for a 64-lane / 2M-view
    //    AMD host on an NVIDIA 32-lane / 1M-view PC): neither is overwritten.
    victory_shader::emit::set_host_max_wave(64);
    victory_shader::emit::set_view_heap_limit(2_097_151);
    let expected_fp = victory_shader::emit::translation_fingerprint();

    let _ = victory_gpu::shader_cache::warm(b"synthetic");

    assert_eq!(
        victory_shader::emit::host_max_wave(),
        64,
        "shader_cache::warm overwrote explicit VICTORY_HOST_WAVE"
    );
    assert_eq!(
        victory_shader::emit::view_heap_limit(),
        2_097_151,
        "shader_cache::warm overwrote explicit VICTORY_VIEW_LIMIT"
    );
    assert_eq!(
        victory_shader::emit::translation_fingerprint(),
        expected_fp,
        "shader_cache::warm changed translation_fingerprint despite explicit host facts"
    );

    // 2. Only VICTORY_HOST_WAVE supplied (`view_heap_limit == 0`): probing the
    //    local D3D12 adapter populates view_heap_limit without clobbering wave.
    victory_shader::emit::set_view_heap_limit(0);
    let _ = victory_gpu::shader_cache::warm(b"synthetic");

    assert_eq!(
        victory_shader::emit::host_max_wave(),
        64,
        "host D3D12 probe overwrote explicit VICTORY_HOST_WAVE"
    );
    assert_ne!(
        victory_shader::emit::view_heap_limit(),
        0,
        "missing view_heap_limit was not probed from host D3D12 adapter"
    );
}
