//! CRIU (Checkpoint/Restore in Userspace) wrapper

use crate::{Error, Result};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

/// CRIU checkpoint/restore operations
pub struct Criu {
    images_dir: PathBuf,
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
            .arg("--ext-unix-sk")
            .arg("--file-locks")
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

    /// Whether restore should use CRIU lazy-pages (userfaultfd).
    pub fn lazy_pages_requested() -> bool {
        match std::env::var("KRYO_LAZY_PAGES") {
            Ok(value) => matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            ),
            Err(_) => false,
        }
    }

    /// Start the CRIU lazy-pages daemon for this image directory.
    pub fn start_lazy_pages(&self) -> Result<Child> {
        let mut child = Command::new("criu")
            .arg("lazy-pages")
            .arg("-D")
            .arg(&self.images_dir)
            .current_dir(&self.images_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| Error::Criu(format!("failed to spawn criu lazy-pages: {e}")))?;

        let socket = self.images_dir.join("lazy-pages.socket");
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            match child.try_wait() {
                Ok(Some(status)) => {
                    return Err(Error::Criu(format!(
                        "criu lazy-pages exited early: {:?}",
                        status.code()
                    )));
                }
                Ok(None) => {}
                Err(error) => {
                    return Err(Error::Criu(format!(
                        "failed to poll criu lazy-pages: {error}"
                    )));
                }
            }
            if socket.exists() {
                return Ok(child);
            }
            thread::sleep(Duration::from_millis(50));
        }
        let _ = child.kill();
        let _ = child.wait();
        Err(Error::Criu(
            "timed out waiting for criu lazy-pages socket".into(),
        ))
    }

    /// Restore a process from checkpoint (blocking)
    pub fn restore(&self) -> Result<()> {
        let mut command = Command::new("criu");
        command
            .arg("restore")
            .arg("-D")
            .arg(&self.images_dir)
            .arg("--shell-job")
            .arg("--tcp-close")
            .arg("--ext-unix-sk")
            .arg("--file-locks");
        if Self::lazy_pages_requested() {
            command.arg("--lazy-pages");
        }
        let status = command.status()?;

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
    pub fn restore_detached(&self, lazy_pages: bool) -> Result<u32> {
        let pidfile = self.images_dir.join("restore.pid");

        let mut command = Command::new("criu");
        command
            .arg("restore")
            .arg("-D")
            .arg(&self.images_dir)
            .arg("--shell-job")
            .arg("--tcp-close")
            .arg("--ext-unix-sk")
            .arg("--file-locks")
            .arg("-d") // Detach
            .arg("--pidfile")
            .arg(&pidfile);
        if lazy_pages {
            command.arg("--lazy-pages");
        }
        let status = command.status()?;

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
