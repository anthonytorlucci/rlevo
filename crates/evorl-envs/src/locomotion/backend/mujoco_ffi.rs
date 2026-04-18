//! Placeholder for the MuJoCo-FFI backend.
//!
//! The `mujoco-ffi` cargo feature is reserved for a future spec (see
//! `projects/burn-evorl/specs/<date>-mujoco-ffi-backend.md`). Enabling it in
//! v1 is a configuration error — the build fails at compile time with a
//! pointer to the follow-up work.
//!
//! To build locomotion envs against the pure-Rust Rapier3D backend, enable
//! only `--features locomotion` (and omit `mujoco-ffi`).

compile_error!(
    "The `mujoco-ffi` backend is unimplemented in v1 of evorl-envs::locomotion. \
     See the deferred spec (`specs/<date>-mujoco-ffi-backend.md`) for the \
     follow-up work. Remove the `mujoco-ffi` feature to build against the \
     pure-Rust Rapier3D backend."
);
