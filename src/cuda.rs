//! cuda-checkpoint wrapper for CUDA state management

use crate::{Error, Result};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
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
        eprintln!("kryo: cuda-checkpoint resume pid {pid} state={state}");
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

    /// PID in `root`'s process tree that actually holds a CUDA context.
    ///
    /// vLLM and Triton fork GPU work into a child (EngineCore). `--wait`
    /// snapshots the frontend; cuda-checkpoint on that PID fails with
    /// "Could not find restore thread". Prefer a descendant listed by
    /// nvidia-smi, then any descendant `cuda-checkpoint --get-state` accepts.
    pub fn gpu_pid_in_tree(root: u32) -> u32 {
        let tree = proc_tree(root);
        let hits: Vec<u32> = nvidia_compute_pids()
            .into_iter()
            .filter(|pid| tree.contains(pid))
            .collect();
        let chosen = hits
            .iter()
            .copied()
            .find(|pid| *pid != root)
            .or_else(|| hits.first().copied());
        if let Some(pid) = chosen {
            if pid != root {
                eprintln!("kryo: cuda-checkpoint pid {pid} (spawned {root})");
            }
            return pid;
        }
        for pid in &tree {
            if *pid != root && Self::state(*pid).is_ok() {
                eprintln!("kryo: cuda-checkpoint pid {pid} via get-state (spawned {root})");
                return *pid;
            }
        }
        root
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

fn nvidia_compute_pids() -> Vec<u32> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let Ok(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    parse_nvidia_pids(&String::from_utf8_lossy(&output.stdout))
}

fn parse_nvidia_pids(stdout: &str) -> Vec<u32> {
    stdout
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("n/a") {
                None
            } else {
                trimmed.parse().ok()
            }
        })
        .collect()
}

fn ppid_from_stat(stat: &str) -> Option<u32> {
    let rparen = stat.rfind(')')?;
    let mut rest = stat[rparen + 1..].split_whitespace();
    rest.next()?; // state
    rest.next()?.parse().ok()
}

fn ppid_of(pid: u32) -> Option<u32> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    ppid_from_stat(&stat)
}

fn proc_tree(root: u32) -> HashSet<u32> {
    let mut children: HashMap<u32, Vec<u32>> = HashMap::new();
    let Ok(dir) = fs::read_dir("/proc") else {
        return HashSet::from([root]);
    };
    for entry in dir.flatten() {
        let Ok(pid) = entry.file_name().to_string_lossy().parse::<u32>() else {
            continue;
        };
        if let Some(ppid) = ppid_of(pid) {
            children.entry(ppid).or_default().push(pid);
        }
    }
    let mut out = HashSet::from([root]);
    let mut queue = VecDeque::from([root]);
    while let Some(pid) = queue.pop_front() {
        for child in children.get(&pid).into_iter().flatten() {
            if out.insert(*child) {
                queue.push_back(*child);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_nvidia_pids_skips_empty_and_na() {
        assert_eq!(parse_nvidia_pids("14687\nN/A\n\n  99 \n"), vec![14687, 99]);
    }

    #[test]
    fn ppid_from_stat_handles_spaces_in_comm() {
        let stat = "14687 (VLLM::EngineCore) S 14001 14001 14001 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0";
        assert_eq!(ppid_from_stat(stat), Some(14001));
    }
}
