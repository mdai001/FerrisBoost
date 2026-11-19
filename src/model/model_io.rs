//! Failure-safe model replacement. Byte serialization and parallel writing
//! remain the caller's responsibility.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

struct PendingFile(PathBuf);

impl Drop for PendingFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Commit only after the writer and file synchronization succeed. A failed
/// write leaves an existing destination untouched. Readers see either complete
/// version; hard links to the old inode continue to refer to that old version.
/// This does not promise power-loss durability of the directory entry.
pub(crate) fn atomic_write(
    path: &Path,
    write: impl FnOnce(&File) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    // Preserve the previous follow-symlink behavior for existing model links.
    let destination = match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => path.canonicalize()?,
        _ => path.to_path_buf(),
    };
    let permissions = match std::fs::metadata(&destination) {
        Ok(metadata) => {
            if !metadata.is_file() {
                anyhow::bail!("model destination is not a regular file");
            }
            if metadata.permissions().readonly() {
                anyhow::bail!("model destination is read-only");
            }
            Some(metadata.permissions())
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let (pending, file) = loop {
        let serial = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(
            ".ferrisboost-save-{}-{serial}.tmp",
            std::process::id()
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&candidate) {
            Ok(file) => break (PendingFile(candidate), file),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    };
    write(&file)?;
    if let Some(permissions) = permissions {
        file.set_permissions(permissions)?;
    }
    file.sync_all()?;
    // Close before replacement, including on Windows. std::fs::rename replaces
    // an existing regular destination on supported Unix and Windows targets.
    drop(file);
    std::fs::rename(&pending.0, &destination)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    struct TestDirectory(PathBuf);
    impl TestDirectory {
        fn new() -> Self {
            let serial = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("fb-model-io-{}-{serial}", std::process::id()));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn assert_only_model(&self) {
            assert_eq!(std::fs::read_dir(&self.0).unwrap().count(), 1);
        }
    }
    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn replacement_and_failed_write_preserve_complete_versions() {
        let directory = TestDirectory::new();
        let path = directory.0.join("model.json");
        atomic_write(&path, |mut file| {
            file.write_all(b"old model")?;
            Ok(())
        })
        .unwrap();
        let result = atomic_write(&path, |mut file| {
            file.write_all(b"partial")?;
            anyhow::bail!("injected write failure")
        });
        assert!(result.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"old model");
        directory.assert_only_model();
        atomic_write(&path, |mut file| {
            file.write_all(b"new")?;
            Ok(())
        })
        .unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
        directory.assert_only_model();
    }

    #[test]
    fn panic_cleans_up_without_replacing_target() {
        let directory = TestDirectory::new();
        let path = directory.0.join("model.json");
        std::fs::write(&path, b"original").unwrap();
        let result = std::panic::catch_unwind(|| {
            let _ = atomic_write(&path, |mut file| {
                file.write_all(b"partial")?;
                panic!("injected writer panic");
            });
        });
        assert!(result.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"original");
        directory.assert_only_model();
    }

    #[cfg(unix)]
    #[test]
    fn symlink_target_and_permissions_are_preserved() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let directory = TestDirectory::new();
        let path = directory.0.join("model.json");
        std::fs::write(&path, b"old").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        let link = directory.0.join("link.json");
        symlink(&path, &link).unwrap();
        atomic_write(&link, |mut file| {
            file.write_all(b"new")?;
            Ok(())
        })
        .unwrap();
        assert!(std::fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
}
