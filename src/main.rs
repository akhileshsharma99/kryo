use clap::{Parser, Subcommand};
use kryo::{Criu, CudaCheckpoint, Snapshot};
use signal_hook::consts::SIGUSR1;
use signal_hook::iterator::Signals;
use std::fs;
use std::process::{Command, Stdio};
use std::time::Duration;

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

        /// Wait time in seconds before checkpointing (default: wait for SIGUSR1 signal)
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

    // Create snapshot directory and metadata
    let snapshot = Snapshot::create(name, command.to_vec())?;

    // Create images directory for CRIU
    let images_dir = snapshot.images_dir();
    fs::create_dir_all(&images_dir)?;

    // Spawn the user's command
    let mut child = Command::new(&command[0])
        .args(&command[1..])
        .stdin(Stdio::null())
        .spawn()?;

    let pid = child.id();

    // Wait for process to be ready
    match wait {
        Some(seconds) => {
            std::thread::sleep(Duration::from_secs(seconds));
        }
        None => {
            wait_for_signal()?;
        }
    }

    // Suspend CUDA state
    CudaCheckpoint::toggle(pid)?;

    // Checkpoint the process with CRIU
    let criu = Criu::new(&images_dir);
    criu.checkpoint(pid)?;

    // Process is killed by CRIU after checkpoint
    let _ = child.wait();

    println!("Snapshot '{}' created", name);

    Ok(())
}

/// Wait for SIGUSR1 signal from child process
fn wait_for_signal() -> Result<(), Box<dyn std::error::Error>> {
    let mut signals = Signals::new([SIGUSR1])?;

    // Block until we receive SIGUSR1
    for signal in signals.forever() {
        if signal == SIGUSR1 {
            return Ok(());
        }
    }

    Err("Signal handler terminated unexpectedly".into())
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

    // Restore the process with CRIU (detached mode)
    let criu = Criu::new(&images_dir);
    let pid = criu.restore_detached()?;

    // Resume CUDA state
    CudaCheckpoint::toggle(pid)?;

    // Wake up the restored process
    send_signal(pid, libc::SIGUSR2)?;

    // Wait for the restored process to complete
    wait_for_process(pid)?;

    Ok(())
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
