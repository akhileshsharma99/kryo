//! cuda-checkpoint wrapper for CUDA state management

use crate::{Error, Result};
use std::process::Command;

/// CUDA checkpoint operations using nvidia's cuda-checkpoint utility
pub struct CudaCheckpoint;

impl CudaCheckpoint {
    /// Toggle CUDA state (suspend or resume) for a process
    ///
    /// When called on a running CUDA process, it suspends CUDA state.
    /// When called on a suspended CUDA process, it resumes CUDA state.
    pub fn toggle(pid: u32) -> Result<()> {
        let status = Command::new("cuda-checkpoint")
            .args(["--toggle", "--pid", &pid.to_string()])
            .status()?;

        if status.success() {
            Ok(())
        } else {
            Err(Error::Cuda(format!(
                "cuda-checkpoint failed with exit code: {:?}",
                status.code()
            )))
        }
    }
}
