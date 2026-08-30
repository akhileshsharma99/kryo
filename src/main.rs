use clap::{Parser, Subcommand};
use kryo::{Criu, CudaCheckpoint, Snapshot};
use signal_hook::consts::{SIGINT, SIGTERM, SIGUSR1};
use signal_hook::iterator::SignalsInfo;
use signal_hook::iterator::exfiltrator::WithOrigin;
use std::fs;
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const KRYO_CLI_PID_ENV: &str = "KRYO_CLI_PID";

#[derive(Parser)]
#[command(name = "kryo")]
#[command(version)]
#[command(about = "Sub-second cold starts for GPU inference")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Manage snapshots
    Snapshot {
        #[command(subcommand)]
        action: SnapshotCommands,
    },
    /// Restore and run a snapshot
    Run {
        /// Name of the snapshot to restore
        #[arg(long)]
        snapshot: String,
    },
}

#[derive(Subcommand)]
enum SnapshotCommands {
    /// Create a new snapshot
    Create {
        /// Name for the snapshot
        #[arg(long)]
        name: String,

        /// Wait before checkpointing the direct command (default: wait for SIGUSR1)
        #[arg(long)]
        wait: Option<u64>,

        /// Command to run and snapshot
        #[arg(last = true)]
        command: Vec<String>,
    },
    /// List all snapshots
    List,
    /// Show snapshot details
    Inspect {
        /// Name of the snapshot
        name: String,
    },
    /// Delete a snapshot
    Delete {
        /// Name of the snapshot
        name: String,
    },
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Snapshot { action } => match action {
            SnapshotCommands::Create {
                name,
                wait,
                command,
            } => cmd_snapshot_create(&name, wait, &command),
            SnapshotCommands::List => cmd_snapshot_list(),
            SnapshotCommands::Inspect { name } => cmd_snapshot_inspect(&name),
            SnapshotCommands::Delete { name } => cmd_snapshot_delete(&name),
        },
        Commands::Run { snapshot } => cmd_run(&snapshot),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn cmd_snapshot_create(
    name: &str,
    wait: Option<u64>,
    command: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    if command.is_empty() {
        return Err("No command provided".into());
    }

    let mut snapshot = Snapshot::create(name, command.to_vec())?;
    let result = create_snapshot(&mut snapshot, wait, command);

    if result.is_err() {
        if let Err(cleanup_error) = Snapshot::delete(name) {
            eprintln!(
                "Warning: failed to remove incomplete snapshot '{}': {}",
                name, cleanup_error
            );
        }
    }

    result?;
    println!("Snapshot '{}' created", name);
    Ok(())
}

fn create_snapshot(
    snapshot: &mut Snapshot,
    wait: Option<u64>,
    command: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let images_dir = snapshot.images_dir();
    fs::create_dir_all(&images_dir)?;

    // Register before spawning so a fast workload cannot signal before the
    // handler is ready.
    let mut signals = SignalsInfo::<WithOrigin>::new([SIGUSR1, SIGINT, SIGTERM])?;

    let child = Command::new(&command[0])
        .args(&command[1..])
        .stdin(Stdio::null())
        .env(KRYO_CLI_PID_ENV, std::process::id().to_string())
        .process_group(0)
        .spawn()?;
    let mut child = ChildGuard::new(child);
    let root_pid = child.id();

    let workload_pid = match wait {
        Some(seconds) => wait_for_duration(
            child.child_mut(),
            &mut signals,
            root_pid,
            Duration::from_secs(seconds),
        )?
        .unwrap_or(root_pid),
        None => wait_for_signal(child.child_mut(), &mut signals, root_pid)?,
    };

    snapshot.set_workload_pid(workload_pid)?;

    check_for_interruption(&mut signals)?;
    CudaCheckpoint::suspend(workload_pid)?;
    child.mark_cuda_suspended(workload_pid);
    check_for_interruption(&mut signals)?;

    let criu = Criu::new(&images_dir);
    criu.checkpoint(root_pid)?;

    // CRIU killed the process tree and captured the suspended CUDA state.
    child.mark_checkpointed();
    check_for_interruption(&mut signals)?;
    let _ = child.child_mut().wait();
    check_for_interruption(&mut signals)?;
    child.disarm();

    Ok(())
}

/// Wait for SIGUSR1 and return the PID of the process that sent it.
fn wait_for_signal(
    child: &mut Child,
    signals: &mut SignalsInfo<WithOrigin>,
    process_group: u32,
) -> Result<u32, Box<dyn std::error::Error>> {
    loop {
        for origin in signals.pending() {
            match origin.signal {
                SIGUSR1 => {
                    if let Some(pid) = valid_workload_sender(&origin, process_group)? {
                        return Ok(pid);
                    }
                }
                SIGINT | SIGTERM => return Err("Snapshot creation interrupted".into()),
                _ => {}
            }
        }

        if let Some(status) = child.try_wait()? {
            return Err(format!("Command exited before checkpointing ({status})").into());
        }

        std::thread::sleep(Duration::from_millis(10));
    }
}

fn wait_for_duration(
    child: &mut Child,
    signals: &mut SignalsInfo<WithOrigin>,
    process_group: u32,
    duration: Duration,
) -> Result<Option<u32>, Box<dyn std::error::Error>> {
    let deadline = Instant::now() + duration;
    let mut workload_pid = None;
    loop {
        for origin in signals.pending() {
            match origin.signal {
                SIGUSR1 => {
                    if let Some(pid) = valid_workload_sender(&origin, process_group)? {
                        workload_pid = Some(pid);
                    }
                }
                SIGINT | SIGTERM => return Err("Snapshot creation interrupted".into()),
                _ => {}
            }
        }

        if let Some(status) = child.try_wait()? {
            return Err(format!("Command exited before checkpointing ({status})").into());
        }

        let now = Instant::now();
        if now >= deadline {
            return Ok(workload_pid);
        }

        std::thread::sleep((deadline - now).min(Duration::from_millis(10)));
    }
}

fn check_for_interruption(
    signals: &mut SignalsInfo<WithOrigin>,
) -> Result<(), Box<dyn std::error::Error>> {
    for origin in signals.pending() {
        if matches!(origin.signal, SIGINT | SIGTERM) {
            return Err("Snapshot creation interrupted".into());
        }
    }
    Ok(())
}

fn valid_workload_sender(
    origin: &signal_hook::low_level::siginfo::Origin,
    process_group: u32,
) -> Result<Option<u32>, Box<dyn std::error::Error>> {
    let Some(process) = origin.process else {
        return Err("Checkpoint signal did not include a sender PID".into());
    };
    let pid = u32::try_from(process.pid)
        .map_err(|_| format!("Invalid checkpoint signal PID: {}", process.pid))?;
    let pid_i32 =
        i32::try_from(pid).map_err(|_| format!("Invalid checkpoint signal PID: {pid}"))?;
    let expected_group = i32::try_from(process_group)
        .map_err(|_| format!("Invalid workload process group: {process_group}"))?;
    let actual_group = unsafe { libc::getpgid(pid_i32) };

    if actual_group == expected_group {
        Ok(Some(pid))
    } else {
        Ok(None)
    }
}

fn cmd_snapshot_list() -> Result<(), Box<dyn std::error::Error>> {
    let snapshots = Snapshot::list()?;

    if snapshots.is_empty() {
        println!("No snapshots");
        return Ok(());
    }

    println!("{:<20} {:<25} COMMAND", "NAME", "CREATED");
    println!("{}", "-".repeat(70));

    for snapshot in snapshots {
        let created = snapshot.created_at.format("%Y-%m-%d %H:%M:%S UTC");
        let command = snapshot.command.join(" ");
        let command_display = if command.len() > 30 {
            format!("{}...", &command[..27])
        } else {
            command
        };
        println!("{:<20} {:<25} {}", snapshot.name, created, command_display);
    }

    Ok(())
}

fn cmd_snapshot_inspect(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let snapshot = Snapshot::load(name)?;

    println!("Name:     {}", snapshot.metadata.name);
    println!(
        "Created:  {}",
        snapshot.metadata.created_at.format("%Y-%m-%d %H:%M:%S UTC")
    );
    println!("Command:  {}", snapshot.metadata.command.join(" "));
    println!("Path:     {}", snapshot.path.display());

    // Show size if images exist
    let images_dir = snapshot.images_dir();
    if images_dir.exists() {
        let size = dir_size(&images_dir).unwrap_or(0);
        println!("Size:     {}", format_size(size));
    }

    Ok(())
}

fn cmd_snapshot_delete(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    Snapshot::delete(name)?;
    println!("Snapshot '{}' deleted", name);
    Ok(())
}

fn cmd_run(snapshot_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let snapshot = Snapshot::load(snapshot_name)?;
    let images_dir = snapshot.images_dir();

    if !images_dir.exists() {
        return Err(format!("Snapshot '{}' has no checkpoint images", snapshot_name).into());
    }

    let criu = Criu::new(&images_dir);
    let mut lazy_daemon = None;
    if Criu::lazy_pages_requested() {
        lazy_daemon = Some(criu.start_lazy_pages()?);
    }

    let run_result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let root_pid = criu.restore_detached(lazy_daemon.is_some())?;
        let workload_pid = snapshot.metadata.workload_pid.unwrap_or(root_pid);

        CudaCheckpoint::resume(workload_pid)?;
        // SIGUSR2 is often ignored after CUDA restore; SIGRTMIN+1 is not.
        #[cfg(target_os = "linux")]
        send_signal(workload_pid, libc::SIGRTMIN() + 1)?;
        let _ = send_signal(workload_pid, libc::SIGUSR2);
        wait_for_process(root_pid)?;
        Ok(())
    })();

    if let Some(mut daemon) = lazy_daemon {
        let _ = daemon.kill();
        let _ = daemon.wait();
    }

    run_result
}

struct ChildGuard {
    child: Child,
    suspended_cuda_pid: Option<u32>,
    armed: bool,
}

impl ChildGuard {
    fn new(child: Child) -> Self {
        Self {
            child,
            suspended_cuda_pid: None,
            armed: true,
        }
    }

    fn id(&self) -> u32 {
        self.child.id()
    }

    fn child_mut(&mut self) -> &mut Child {
        &mut self.child
    }

    fn mark_cuda_suspended(&mut self, pid: u32) {
        self.suspended_cuda_pid = Some(pid);
    }

    fn mark_checkpointed(&mut self) {
        self.suspended_cuda_pid = None;
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }

        if let Some(pid) = self.suspended_cuda_pid {
            let _ = CudaCheckpoint::resume(pid);
        }

        if let Ok(process_group) = i32::try_from(self.child.id()) {
            // The workload is started in its own process group so wrappers and
            // their descendants are cleaned up together.
            let _ = unsafe { libc::kill(-process_group, libc::SIGKILL) };
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Send a signal to a process
fn send_signal(pid: u32, signal: i32) -> Result<(), Box<dyn std::error::Error>> {
    let result = unsafe { libc::kill(pid as i32, signal) };
    if result == 0 {
        Ok(())
    } else {
        Err(format!("Failed to send signal {} to PID {}", signal, pid).into())
    }
}

/// Wait for a process to exit
fn wait_for_process(pid: u32) -> Result<(), Box<dyn std::error::Error>> {
    let mut status: i32 = 0;
    let result = unsafe { libc::waitpid(pid as i32, &mut status, 0) };
    if result == -1 {
        // Process might not be a child, try polling
        loop {
            let kill_result = unsafe { libc::kill(pid as i32, 0) };
            if kill_result == -1 {
                // Process no longer exists
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
    Ok(())
}

/// Calculate directory size recursively
fn dir_size(path: &std::path::Path) -> std::io::Result<u64> {
    let mut size = 0;
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if metadata.is_dir() {
                size += dir_size(&entry.path())?;
            } else {
                size += metadata.len();
            }
        }
    }
    Ok(size)
}

/// Format bytes as human-readable size
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static SIGNAL_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn wait_for_signal_returns_sender_pid() {
        let _lock = SIGNAL_TEST_LOCK.lock().unwrap();
        let mut signals = SignalsInfo::<WithOrigin>::new([SIGUSR1, SIGINT, SIGTERM]).unwrap();
        let mut child = Command::new("sh")
            .args([
                "-c",
                "sh -c 'kill -USR1 \"$KRYO_CLI_PID\"; sleep 10' & wait",
            ])
            .env(KRYO_CLI_PID_ENV, std::process::id().to_string())
            .process_group(0)
            .spawn()
            .unwrap();
        let process_group = child.id();

        let sender_pid = wait_for_signal(&mut child, &mut signals, process_group).unwrap();

        assert_ne!(sender_pid, process_group);
        let _ = unsafe { libc::kill(-(process_group as i32), libc::SIGKILL) };
        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    fn wait_for_signal_reports_early_child_exit() {
        let _lock = SIGNAL_TEST_LOCK.lock().unwrap();
        let mut signals = SignalsInfo::<WithOrigin>::new([SIGUSR1, SIGINT, SIGTERM]).unwrap();
        let mut child = Command::new("sh")
            .args(["-c", "exit 7"])
            .process_group(0)
            .spawn()
            .unwrap();
        let process_group = child.id();

        let error = wait_for_signal(&mut child, &mut signals, process_group).unwrap_err();

        assert!(error.to_string().contains("exited before checkpointing"));
    }
}
