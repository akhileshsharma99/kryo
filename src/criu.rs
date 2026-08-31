//! CRIU (Checkpoint/Restore in Userspace) wrapper

use crate::{Error, Result};
use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::MetadataExt;
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

    fn nvidia_externals(command: &mut Command, restore: bool) {
        // NVIDIA char devices cannot be dumped. Mark every node under
        // /dev/nvidia* external so Triton fds (nvidiactl, caps, uvm) reconnect.
        for (majmin, name, path) in nvidia_char_devices() {
            command.arg("--external");
            if restore {
                command.arg(format!("{name}:{path}"));
            } else {
                command.arg(format!("dev[{majmin}]:{name}"));
            }
        }
    }

    /// Checkpoint a process by PID
    pub fn checkpoint(&self, pid: u32) -> Result<()> {
        let mut command = Command::new("criu");
        command
            .args(["dump", "-t", &pid.to_string()])
            .arg("-D")
            .arg(&self.images_dir)
            .arg("--shell-job")
            .arg("--tcp-close")
            .arg("--ext-unix-sk")
            .arg("--file-locks")
            .arg("--link-remap");
        Self::nvidia_externals(&mut command, false);
        let status = command.status()?;

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
            .arg("--file-locks")
            .arg("--link-remap");
        Self::nvidia_externals(&mut command, true);
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
            .arg("--link-remap")
            .arg("-d") // Detach
            .arg("--pidfile")
            .arg(&pidfile);
        Self::nvidia_externals(&mut command, true);
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

fn linux_majmin(rdev: u64) -> (u32, u32) {
    let major = ((rdev >> 8) & 0xfff) as u32;
    let minor = ((rdev & 0xff) | ((rdev >> 12) & 0xfff00)) as u32;
    (major, minor)
}

fn nvidia_char_devices() -> Vec<(String, String, String)> {
    let mut by_majmin: BTreeMap<String, (String, String)> = BTreeMap::new();
    for (majmin, name, path) in [
        ("195/255", "nvidiactl", "/dev/nvidiactl"),
        ("195/254", "nvidiauvm", "/dev/nvidia-uvm"),
        ("195/253", "nvidiamodeset", "/dev/nvidia-modeset"),
        ("195/0", "nvidia0", "/dev/nvidia0"),
    ] {
        by_majmin.insert(
            majmin.to_string(),
            (name.to_string(), path.to_string()),
        );
    }
    for dir in ["/dev", "/dev/nvidia-caps"] {
        let Ok(entries) = fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(fname) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if dir == "/dev" && !fname.starts_with("nvidia") {
                continue;
            }
            let Ok(meta) = fs::metadata(&path) else {
                continue;
            };
            if meta.rdev() == 0 {
                continue;
            }
            let (maj, min) = linux_majmin(meta.rdev());
            if maj != 195 && !fname.starts_with("nvidia") {
                continue;
            }
            let majmin = format!("{maj}/{min}");
            let id: String = fname
                .chars()
                .filter(|c| c.is_ascii_alphanumeric())
                .collect();
            if id.is_empty() {
                continue;
            }
            by_majmin.insert(majmin, (id, path.display().to_string()));
        }
    }
    by_majmin
        .into_iter()
        .map(|(majmin, (name, path))| (majmin, name, path))
        .collect()
}
