//! Placeholder for the MuJoCo-FFI backend.
//!
//! The `mujoco-ffi` cargo feature is reserved for a future release.
//! Enabling it today is a configuration error — the build fails at
//! compile time.
//!
//! To build locomotion envs against the pure-Rust Rapier3D backend, enable
//! only `--features locomotion` (and omit `mujoco-ffi`).

compile_error!(
    "The `mujoco-ffi` backend is not yet implemented. \
     Remove the `mujoco-ffi` feature to build against the \
     pure-Rust Rapier3D backend."
);
