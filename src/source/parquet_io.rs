//! Bounded coalesced reads for dense, wide Parquet projections. No mmap or
//! whole-file materialization. Offset readers never share a mutable file cursor
//! outside the lock; reads across cache boundaries preserve ordinary EOF rules.
use bytes::Bytes;
use parquet::errors::{ParquetError, Result};
use parquet::file::reader::{ChunkReader, Length};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::sync::{Arc, Mutex};

pub(crate) const READ_WINDOW_BYTES: usize = 8 * 1024 * 1024;

struct State {
    file: File,
    start: u64,
    valid: usize,
    buffer: Vec<u8>,
}

struct Inner {
    length: u64,
    capacity: usize,
    state: Mutex<State>,
}

pub(crate) struct CoalescedFile(Arc<Inner>);

impl CoalescedFile {
    pub(crate) fn new(file: File) -> io::Result<Self> {
        Self::with_capacity(file, READ_WINDOW_BYTES)
    }

    fn with_capacity(file: File, capacity: usize) -> io::Result<Self> {
        let length = file.metadata()?.len();
        Ok(Self(Arc::new(Inner {
            length,
            capacity: capacity.max(1),
            state: Mutex::new(State {
                file,
                start: 0,
                valid: 0,
                buffer: Vec::new(),
            }),
        })))
    }
}

impl Inner {
    fn read_at(&self, position: u64, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() || position >= self.length {
            return Ok(0);
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| io::Error::other("Parquet read cache lock poisoned"))?;
        let offset = position
            .checked_sub(state.start)
            .and_then(|n| usize::try_from(n).ok());
        if !offset.is_some_and(|n| n < state.valid) {
            state.valid = 0;
            state.start = position;
            let length =
                usize::try_from((self.length - position).min(self.capacity as u64)).unwrap();
            if state.buffer.len() < length {
                let additional = length - state.buffer.len();
                state
                    .buffer
                    .try_reserve_exact(additional)
                    .map_err(|_| io::Error::other("Parquet read cache allocation failed"))?;
                state.buffer.resize(length, 0);
            }
            state.file.seek(SeekFrom::Start(position))?;
            while state.valid < length {
                let State {
                    file,
                    buffer,
                    valid,
                    ..
                } = &mut *state;
                match file.read(&mut buffer[*valid..length]) {
                    Ok(0) => break,
                    Ok(n) => *valid += n,
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(error) => return Err(error),
                }
            }
        }
        let offset = usize::try_from(position - state.start).unwrap();
        let count = output.len().min(state.valid.saturating_sub(offset));
        output[..count].copy_from_slice(&state.buffer[offset..offset + count]);
        Ok(count)
    }
}

pub(crate) struct OffsetReader {
    inner: Arc<Inner>,
    position: u64,
}
impl Read for OffsetReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let count = self.inner.read_at(self.position, buffer)?;
        self.position += count as u64;
        Ok(count)
    }
}

impl Length for CoalescedFile {
    fn len(&self) -> u64 {
        self.0.length
    }
}
impl ChunkReader for CoalescedFile {
    type T = OffsetReader;
    fn get_read(&self, start: u64) -> Result<Self::T> {
        Ok(OffsetReader {
            inner: self.0.clone(),
            position: start,
        })
    }
    fn get_bytes(&self, start: u64, length: usize) -> Result<Bytes> {
        if start > self.len() || (length as u64) > self.len() - start {
            return Err(ParquetError::EOF(format!(
                "Expected to read {length} bytes, read only {}",
                self.len().saturating_sub(start)
            )));
        }
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(length)
            .map_err(|_| ParquetError::General("Parquet byte range allocation failed".into()))?;
        let read = self
            .get_read(start)?
            .take(length as u64)
            .read_to_end(&mut buffer)?;
        if read != length {
            return Err(ParquetError::EOF(format!(
                "Expected to read {length} bytes, read only {read}"
            )));
        }
        Ok(buffer.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn independent_offsets_cross_windows_and_reject_out_of_range_bytes() {
        let path = std::env::temp_dir().join(format!(
            "fb-coalesced-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let values: Vec<_> = (0..257).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &values).unwrap();
        let source = CoalescedFile::with_capacity(File::open(&path).unwrap(), 17).unwrap();
        let mut first = source.get_read(3).unwrap();
        let mut second = source.get_read(141).unwrap();
        // Give each worker its own mutable cursor and advance them in
        // interleaved chunks while they contend for the shared cache window.
        let barrier = std::sync::Barrier::new(4);
        std::thread::scope(|scope| {
            for start in [0, 47, 99, 157] {
                let source = &source;
                let barrier = &barrier;
                let values = &values;
                scope.spawn(move || {
                    let expected = &values[start..start + 73];
                    let mut reader = source.get_read(start as u64).unwrap();
                    let mut actual = Vec::with_capacity(expected.len());
                    barrier.wait();
                    for chunk in [13, 29, 31] {
                        let old_len = actual.len();
                        actual.resize(old_len + chunk, 0);
                        reader.read_exact(&mut actual[old_len..]).unwrap();
                    }
                    assert_eq!(actual, expected);
                });
            }
        });
        let mut bytes = [0; 87];
        first.read_exact(&mut bytes).unwrap();
        assert_eq!(bytes.as_slice(), &values[3..90]);
        assert_eq!(&source.get_bytes(2, 200).unwrap()[..], &values[2..202]);
        second.read_exact(&mut bytes).unwrap();
        assert_eq!(bytes.as_slice(), &values[141..228]);
        first.read_exact(&mut bytes).unwrap();
        assert_eq!(bytes.as_slice(), &values[90..177]);
        assert!(source.get_bytes(256, 2).is_err());
        assert!(source.get_bytes(u64::MAX, 1).is_err());
        assert!(source.get_bytes(257, 0).unwrap().is_empty());
        let mut eof = source.get_read(257).unwrap();
        assert_eq!(eof.read(&mut bytes).unwrap(), 0);
        let mut beyond_eof = source.get_read(300).unwrap();
        assert_eq!(beyond_eof.read(&mut bytes).unwrap(), 0);
        let mut empty_read = source.get_read(3).unwrap();
        assert_eq!(empty_read.read(&mut []).unwrap(), 0);
        let mut next_byte = [0];
        empty_read.read_exact(&mut next_byte).unwrap();
        assert_eq!(next_byte[0], values[3]);
        let state = source.0.state.lock().unwrap();
        assert!(state.buffer.len() <= 17);
        drop(state);
        drop(first);
        drop(second);
        drop(eof);
        drop(beyond_eof);
        drop(empty_read);
        drop(source);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn exact_window_boundary_lengths_work_and_empty_file_is_eof() {
        let path = std::env::temp_dir().join(format!(
            "fb-coalesced-boundary-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let values: Vec<_> = (0..96).map(|i| (i % 239) as u8).collect();
        std::fs::write(&path, &values).unwrap();
        let source = CoalescedFile::with_capacity(File::open(&path).unwrap(), 17).unwrap();

        for length in [16, 17, 18] {
            for start in [0, 15, 16, 17, 31, 78] {
                if start + length <= values.len() {
                    assert_eq!(
                        &source.get_bytes(start as u64, length).unwrap()[..],
                        &values[start..start + length]
                    );
                }
            }
        }
        assert!(source.get_bytes(95, 2).is_err());
        drop(source);
        std::fs::remove_file(path).unwrap();

        let empty_path = std::env::temp_dir().join(format!(
            "fb-coalesced-empty-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&empty_path, []).unwrap();
        let empty = CoalescedFile::with_capacity(File::open(&empty_path).unwrap(), 17).unwrap();
        assert_eq!(empty.len(), 0);
        assert!(empty.get_bytes(0, 0).unwrap().is_empty());
        assert!(empty.get_bytes(0, 1).is_err());
        let mut reader = empty.get_read(0).unwrap();
        let mut byte = [0];
        assert_eq!(reader.read(&mut byte).unwrap(), 0);
        drop(empty);
        std::fs::remove_file(empty_path).unwrap();
    }

    #[test]
    fn uncached_read_after_file_truncation_observes_unexpected_eof() {
        let path = std::env::temp_dir().join(format!(
            "fb-coalesced-truncated-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, vec![0x5a; 96]).unwrap();
        // The reader snapshots the original metadata length before truncation.
        let source = CoalescedFile::with_capacity(File::open(&path).unwrap(), 17).unwrap();
        std::fs::write(&path, vec![0x5a; 40]).unwrap();

        // Position 51 has not been cached. The stale snapshot says bytes
        // remain, but the underlying file returns EOF and read_exact reports it.
        let mut reader = source.get_read(51).unwrap();
        let error = reader.read_exact(&mut [0; 4]).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);

        // Match the pinned File ChunkReader's EOF variant and diagnostic,
        // both for stale metadata truncation and ordinary out-of-range bytes.
        let ordinary = File::open(&path).unwrap();
        let expected = ordinary.get_bytes(51, 4).unwrap_err();
        let actual = source.get_bytes(51, 4).unwrap_err();
        assert!(matches!(actual, ParquetError::EOF(_)));
        assert_eq!(actual.to_string(), expected.to_string());

        drop(reader);
        drop(source);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn byte_range_eof_error_matches_ordinary_file() {
        let path = std::env::temp_dir().join(format!(
            "fb-coalesced-error-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, [1, 2, 3, 4]).unwrap();
        let ordinary = File::open(&path).unwrap();
        let source = CoalescedFile::with_capacity(File::open(&path).unwrap(), 2).unwrap();
        for (start, length) in [(0, 5), (3, 2), (4, 1), (5, 1)] {
            let expected = ordinary.get_bytes(start, length).unwrap_err();
            let actual = source.get_bytes(start, length).unwrap_err();
            assert!(matches!(actual, ParquetError::EOF(_)));
            assert_eq!(actual.to_string(), expected.to_string());
        }
        drop(source);
        drop(ordinary);
        std::fs::remove_file(path).unwrap();
    }
}
