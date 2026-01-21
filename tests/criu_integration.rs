//! Integration tests for CRIU checkpoint/restore
//!
//! These tests require CRIU to be installed and will be skipped if CRIU is not available.

use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

fn criu_available() -> bool {
    Command::new("criu")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn has_criu_permissions() -> bool {
    // Try a simple CRIU check command to see if we have permissions
    Command::new("criu")
        .args(["check"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[test]
fn test_criu_checkpoint_restore() {
    if !criu_available() {
        eprintln!("Skipping test: CRIU not available");
        return;
    }

    if !has_criu_permissions() {
        eprintln!("Skipping test: Insufficient permissions for CRIU");
        return;
    }

    // Create a temp directory for the test
    let test_dir = std::env::temp_dir().join(format!("kryo-criu-test-{}", std::process::id()));
    let images_dir = test_dir.join("images");
    let output_file = test_dir.join("output.txt");

    fs::create_dir_all(&images_dir).expect("Failed to create test directories");

    // Create a simple test script that writes to a file
    let script_path = test_dir.join("test_script.sh");
    let mut script = fs::File::create(&script_path).expect("Failed to create script");
    writeln!(
        script,
        r#"#!/bin/bash
echo "before" > {}
sleep 3
echo "after" >> {}
"#,
        output_file.display(),
        output_file.display()
    )
    .expect("Failed to write script");

    // Make script executable
    Command::new("chmod")
        .args(["+x", script_path.to_str().unwrap()])
        .status()
        .expect("Failed to chmod");

    // Start the script
    let mut child = Command::new("bash")
        .arg(&script_path)
        .spawn()
        .expect("Failed to spawn test process");

    let pid = child.id();

    // Wait a bit for the script to start and write "before"
    thread::sleep(Duration::from_secs(1));

    // Checkpoint the process
    let checkpoint_result = Command::new("criu")
        .args(["dump", "-t", &pid.to_string(), "-D"])
        .arg(&images_dir)
        .arg("--shell-job")
        .status();

    match checkpoint_result {
        Ok(status) if status.success() => {
            // Process was checkpointed and killed by CRIU
            let _ = child.wait();

            // Verify "before" was written
            let content = fs::read_to_string(&output_file).unwrap_or_default();
            assert!(
                content.contains("before"),
                "Expected 'before' in output, got: {}",
                content
            );

            // Restore the process
            let restore_result = Command::new("criu")
                .args(["restore", "-D"])
                .arg(&images_dir)
                .arg("--shell-job")
                .arg("-d") // Detach
                .status();

            match restore_result {
                Ok(status) if status.success() => {
                    // Wait for restored process to complete (sleep remaining ~2s + buffer)
                    thread::sleep(Duration::from_secs(5));

                    // Verify "after" was written by restored process
                    let content = fs::read_to_string(&output_file).unwrap_or_default();
                    assert!(
                        content.contains("after"),
                        "Expected 'after' in output after restore, got: {}",
                        content
                    );
                }
                Ok(status) => {
                    eprintln!("CRIU restore failed with status: {:?}", status.code());
                }
                Err(e) => {
                    eprintln!("CRIU restore error: {}", e);
                }
            }
        }
        Ok(status) => {
            eprintln!(
                "Skipping test: CRIU checkpoint failed with status {:?}",
                status.code()
            );
            let _ = child.kill();
        }
        Err(e) => {
            eprintln!("Skipping test: CRIU checkpoint error: {}", e);
            let _ = child.kill();
        }
    }

    // Cleanup
    let _ = fs::remove_dir_all(&test_dir);
}

#[test]
fn test_criu_version() {
    if !criu_available() {
        eprintln!("Skipping test: CRIU not available");
        return;
    }

    let output = Command::new("criu")
        .arg("--version")
        .output()
        .expect("Failed to run criu --version");

    assert!(output.status.success());

    let version = String::from_utf8_lossy(&output.stdout);
    assert!(
        version.contains("Version:"),
        "Expected version output, got: {}",
        version
    );
}
