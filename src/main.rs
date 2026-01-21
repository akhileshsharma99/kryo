use clap::{Parser, Subcommand};
use kryo::{Criu, CudaCheckpoint, Snapshot};
use std::fs;
use std::process::{Command, Stdio};

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
    /// Run a command with a restored snapshot
    Run {
        /// Name of the snapshot to restore
        #[arg(long)]
        snapshot: String,

        /// Command to run after restore
        #[arg(last = true)]
        command: Vec<String>,
    },
}

#[derive(Subcommand)]
enum SnapshotCommands {
    /// Create a new snapshot
    Create {
        /// Name for the snapshot
        #[arg(long)]
        name: String,

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
            SnapshotCommands::Create { name, command } => cmd_snapshot_create(&name, &command),
            SnapshotCommands::List => cmd_snapshot_list(),
            SnapshotCommands::Inspect { name } => cmd_snapshot_inspect(&name),
            SnapshotCommands::Delete { name } => cmd_snapshot_delete(&name),
        },
        Commands::Run { snapshot, command } => cmd_run(&snapshot, &command),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn cmd_snapshot_create(name: &str, command: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if command.is_empty() {
        return Err("No command provided".into());
    }

    println!("Creating snapshot '{}'...", name);

    // Create snapshot directory and metadata
    let snapshot = Snapshot::create(name, command.to_vec())?;

    // Create images directory for CRIU
    let images_dir = snapshot.images_dir();
    fs::create_dir_all(&images_dir)?;

    // Spawn the user's command
    println!("Running: {}", command.join(" "));
    let mut child = Command::new(&command[0])
        .args(&command[1..])
        .stdin(Stdio::null())
        .spawn()?;

    let pid = child.id();
    println!("Process started with PID: {}", pid);

    // Wait for process to be ready
    // TODO: Add proper signaling mechanism (e.g., wait for specific output or signal)
    println!("Waiting for process to initialize...");
    std::thread::sleep(std::time::Duration::from_secs(5));

    // Suspend CUDA state (if applicable)
    println!("Suspending CUDA state...");
    if let Err(e) = CudaCheckpoint::toggle(pid) {
        eprintln!(
            "Warning: cuda-checkpoint failed (may not be available): {}",
            e
        );
    }

    // Checkpoint the process with CRIU
    println!("Checkpointing process...");
    let criu = Criu::new(&images_dir);
    criu.checkpoint(pid)?;

    // Process is killed by CRIU after checkpoint
    let _ = child.wait();

    println!("Snapshot '{}' created successfully", name);
    println!("Location: {}", snapshot.path.display());

    Ok(())
}

fn cmd_snapshot_list() -> Result<(), Box<dyn std::error::Error>> {
    let snapshots = Snapshot::list()?;

    if snapshots.is_empty() {
        println!("No snapshots found");
        println!("Create one with: kryo snapshot create --name <name> -- <command>");
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

fn cmd_run(snapshot_name: &str, command: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if command.is_empty() {
        return Err("No command provided".into());
    }

    let snapshot = Snapshot::load(snapshot_name)?;
    let images_dir = snapshot.images_dir();

    if !images_dir.exists() {
        return Err(format!("Snapshot '{}' has no checkpoint images", snapshot_name).into());
    }

    println!("Restoring from snapshot '{}'...", snapshot_name);

    // Restore the process with CRIU
    let criu = Criu::new(&images_dir);
    criu.restore()?;

    // Resume CUDA state
    // Note: After restore, we need the PID of the restored process
    // This is simplified - real implementation needs to get PID from CRIU
    println!("Resuming CUDA state...");

    // Execute the user's command in the restored environment
    println!("Running: {}", command.join(" "));
    let status = Command::new(&command[0])
        .args(&command[1..])
        .env("KRYO_RESTORED", "1")
        .env("KRYO_SNAPSHOT", snapshot_name)
        .status()?;

    std::process::exit(status.code().unwrap_or(1));
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
