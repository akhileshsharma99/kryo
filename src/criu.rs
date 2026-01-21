//! CRIU (Checkpoint/Restore in Userspace) wrapper

use crate::{Error, Result};
use std::path::Path;
use std::process::Command;

/// CRIU checkpoint/restore operations
pub struct Criu {
    images_dir: std::path::PathBuf,
}

impl Criu {
    /// Create a new CRIU instance with the specified images directory
    pub fn new(images_dir: impl AsRef<Path>) -> Self {
        Self {
            images_dir: images_dir.as_ref().to_path_buf(),
        }
    }

    /// Checkpoint a process by PID
    pub fn checkpoint(&self, pid: u32) -> Result<()> {
        let status = Command::new("criu")
            .args(["dump", "-t", &pid.to_string()])
            .arg("-D")
            .arg(&self.images_dir)
            .arg("--shell-job")
            .status()?;

        if status.success() {
            Ok(())
        } else {
            Err(Error::Criu(format!(
                "checkpoint failed with exit code: {:?}",
                status.code()
            )))
        }
    }

    /// Restore a process from checkpoint
    pub fn restore(&self) -> Result<()> {
        let status = Command::new("criu")
            .args(["restore"])
            .arg("-D")
            .arg(&self.images_dir)
            .arg("--shell-job")
            .status()?;

        if status.success() {
            Ok(())
        } else {
            Err(Error::Criu(format!(
                "restore failed with exit code: {:?}",
                status.code()
            )))
        }
    }
}
