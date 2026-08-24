//! Kryo - Sub-second cold starts for GPU inference
//!
//! This library provides process checkpoint/restore functionality
//! using CRIU and cuda-checkpoint for GPU state management.

pub mod criu;
pub mod cuda;
pub mod error;
pub mod snapshot;

pub use criu::Criu;
pub use cuda::CudaCheckpoint;
pub use error::{Error, Result};
pub use snapshot::{Snapshot, SnapshotMetadata};
