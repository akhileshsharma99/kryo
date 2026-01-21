//! Snapshot management

use crate::{Error, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// Metadata stored with each snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub command: Vec<String>,
}

/// A Kryo snapshot
#[derive(Debug)]
pub struct Snapshot {
    pub path: PathBuf,
    pub metadata: SnapshotMetadata,
}

impl Snapshot {
    /// Get the default snapshots directory
    pub fn default_base_dir() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".kryo")
            .join("snapshots")
    }

    /// Create a new snapshot directory
    pub fn create(name: &str, command: Vec<String>) -> Result<Self> {
        let base_dir = Self::default_base_dir();
        let path = base_dir.join(name);

        if path.exists() {
            return Err(Error::SnapshotExists(name.to_string()));
        }

        fs::create_dir_all(&path)?;

        let metadata = SnapshotMetadata {
            name: name.to_string(),
            created_at: Utc::now(),
            command,
        };

        let metadata_path = path.join("metadata.json");
        let file = fs::File::create(&metadata_path)?;
        serde_json::to_writer_pretty(file, &metadata)?;

        Ok(Self { path, metadata })
    }

    /// Load an existing snapshot by name
    pub fn load(name: &str) -> Result<Self> {
        let base_dir = Self::default_base_dir();
        let path = base_dir.join(name);

        if !path.exists() {
            return Err(Error::SnapshotNotFound(name.to_string()));
        }

        let metadata_path = path.join("metadata.json");
        let file = fs::File::open(&metadata_path)?;
        let metadata: SnapshotMetadata = serde_json::from_reader(file)?;

        Ok(Self { path, metadata })
    }

    /// List all available snapshots
    pub fn list() -> Result<Vec<SnapshotMetadata>> {
        let base_dir = Self::default_base_dir();
        let mut snapshots = Vec::new();

        if !base_dir.exists() {
            return Ok(snapshots);
        }

        for entry in fs::read_dir(&base_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let metadata_path = entry.path().join("metadata.json");
                if metadata_path.exists() {
                    let file = fs::File::open(&metadata_path)?;
                    if let Ok(metadata) = serde_json::from_reader::<_, SnapshotMetadata>(file) {
                        snapshots.push(metadata);
                    }
                }
            }
        }

        Ok(snapshots)
    }

    /// Delete a snapshot by name
    pub fn delete(name: &str) -> Result<()> {
        let base_dir = Self::default_base_dir();
        let path = base_dir.join(name);

        if !path.exists() {
            return Err(Error::SnapshotNotFound(name.to_string()));
        }

        fs::remove_dir_all(&path)?;
        Ok(())
    }

    /// Get the CRIU images directory for this snapshot
    pub fn images_dir(&self) -> PathBuf {
        self.path.join("images")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn test_base_dir() -> PathBuf {
        env::temp_dir().join("kryo-test").join(format!(
            "test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn cleanup(path: &PathBuf) {
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn test_snapshot_create_and_load() {
        let base_dir = test_base_dir();
        fs::create_dir_all(&base_dir).unwrap();

        let snapshot_path = base_dir.join("test-snapshot");
        fs::create_dir_all(&snapshot_path).unwrap();

        // Create metadata manually for testing
        let metadata = SnapshotMetadata {
            name: "test-snapshot".to_string(),
            created_at: Utc::now(),
            command: vec!["python".to_string(), "test.py".to_string()],
        };

        let metadata_path = snapshot_path.join("metadata.json");
        let file = fs::File::create(&metadata_path).unwrap();
        serde_json::to_writer_pretty(file, &metadata).unwrap();

        // Load and verify
        let loaded_file = fs::File::open(&metadata_path).unwrap();
        let loaded: SnapshotMetadata = serde_json::from_reader(loaded_file).unwrap();

        assert_eq!(loaded.name, "test-snapshot");
        assert_eq!(loaded.command, vec!["python", "test.py"]);

        cleanup(&base_dir);
    }

    #[test]
    fn test_snapshot_metadata_serialization() {
        let metadata = SnapshotMetadata {
            name: "my-snapshot".to_string(),
            created_at: Utc::now(),
            command: vec![
                "uv".to_string(),
                "run".to_string(),
                "python".to_string(),
                "app.py".to_string(),
            ],
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let deserialized: SnapshotMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.name, metadata.name);
        assert_eq!(deserialized.command, metadata.command);
    }

    #[test]
    fn test_images_dir() {
        let snapshot = Snapshot {
            path: PathBuf::from("/tmp/kryo/snapshots/test"),
            metadata: SnapshotMetadata {
                name: "test".to_string(),
                created_at: Utc::now(),
                command: vec![],
            },
        };

        assert_eq!(
            snapshot.images_dir(),
            PathBuf::from("/tmp/kryo/snapshots/test/images")
        );
    }
}
