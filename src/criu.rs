//! CRIU (Checkpoint/Restore in Userspace) wrapper

use crate::{Error, Result};
use std::fs;
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
            .arg("--tcp-close")
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

    /// Restore a process from checkpoint (blocking)
    pub fn restore(&self) -> Result<()> {
        let status = Command::new("criu")
            .args(["restore"])
            .arg("-D")
            .arg(&self.images_dir)
            .arg("--shell-job")
            .arg("--tcp-close")
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

    /// Restore a process from checkpoint (detached, returns PID)
    pub fn restore_detached(&self) -> Result<u32> {
        let pidfile = self.images_dir.join("restore.pid");

        let status = Command::new("criu")
            .args(["restore"])
            .arg("-D")
            .arg(&self.images_dir)
            .arg("--shell-job")
            .arg("--tcp-close")
            .arg("-d") // Detach
            .arg("--pidfile")
            .arg(&pidfile)
            .status()?;

        if !status.success() {
            return Err(Error::Criu(format!(
                "restore failed with exit code: {:?}",
                status.code()
            )));
        }

        // Read PID from pidfile
        let pid_str = fs::read_to_string(&pidfile)
            .map_err(|e| Error::Criu(format!("failed to read pidfile: {}", e)))?;

        let pid: u32 = pid_str
            .trim()
            .parse()
            .map_err(|e| Error::Criu(format!("failed to parse PID '{}': {}", pid_str.trim(), e)))?;

        // Clean up pidfile
        let _ = fs::remove_file(&pidfile);

        Ok(pid)
    }
}
