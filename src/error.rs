use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("CRIU error: {0}")]
    Criu(String),

    #[error("CUDA checkpoint error: {0}")]
    Cuda(String),

    #[error("Snapshot not found: {0}")]
    SnapshotNotFound(String),

    #[error("Snapshot already exists: {0}")]
    SnapshotExists(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Command failed: {0}")]
    CommandFailed(String),
}

pub type Result<T> = std::result::Result<T, Error>;
