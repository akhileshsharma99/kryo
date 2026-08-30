//! cuda-checkpoint wrapper for CUDA state management

use crate::{Error, Result};
use std::process::Command;

/// CUDA checkpoint operations using nvidia's cuda-checkpoint utility
pub struct CudaCheckpoint;

impl CudaCheckpoint {
    /// Suspend CUDA for dump: lock, then checkpoint GPU memory to host.
    pub fn suspend(pid: u32) -> Result<()> {
        Self::action("lock", pid, Some(60_000))?;
        Self::action("checkpoint", pid, None)
    }

    /// Resume CUDA after CRIU restore.
    ///
    /// CRIU brings back the host-side copy, so the process is often `locked`
    /// rather than `checkpointed`. `--toggle` left it frozen; `--action restore`
    /// fails in the locked state. Unlock (and restore first if needed).
    pub fn resume(pid: u32) -> Result<()> {
        let state = Self::state(pid)?;
        match state.as_str() {
            "checkpointed" => {
                Self::action("restore", pid, None)?;
                Self::action("unlock", pid, None)
            }
            "locked" => Self::action("unlock", pid, None),
            "running" => Ok(()),
            other => Err(Error::Cuda(format!(
                "cuda-checkpoint pid {pid} in unexpected state {other:?}"
            ))),
        }
    }

    fn state(pid: u32) -> Result<String> {
        let output = Command::new("cuda-checkpoint")
            .args(["--get-state", "--pid", &pid.to_string()])
            .output()?;
        if !output.status.success() {
            return Err(Self::fail("get-state", pid, &output));
        }
        Ok(String::from_utf8_lossy(&output.stdout)
            .trim()
            .to_ascii_lowercase())
    }

    fn action(action: &str, pid: u32, timeout_ms: Option<u32>) -> Result<()> {
        let mut command = Command::new("cuda-checkpoint");
        command.args(["--action", action, "--pid", &pid.to_string()]);
        if let Some(ms) = timeout_ms {
            command.args(["--timeout", &ms.to_string()]);
        }
        let output = command.output()?;
        if output.status.success() {
            return Ok(());
        }
        Err(Self::fail(action, pid, &output))
    }

    fn fail(op: &str, pid: u32, output: &std::process::Output) -> Error {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let detail = [stderr.trim(), stdout.trim()]
            .into_iter()
            .find(|s| !s.is_empty())
            .unwrap_or("no output");
        Error::Cuda(format!(
            "cuda-checkpoint {op} pid {pid} failed ({:?}): {detail}",
            output.status.code()
        ))
    }
}
