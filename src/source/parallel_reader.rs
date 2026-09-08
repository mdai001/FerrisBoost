//! Bounded, deterministic parallel file decoding.
//!
//! Workers may decode different files concurrently, but every file owns a
//! one-slot channel and the consumer drains those channels in input order.
//! This keeps row order stable and caps decoded data at one queued batch plus
//! one current batch per active reader instead of collecting whole files.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};

use arrow::record_batch::RecordBatch;

pub type BatchIter = Box<dyn Iterator<Item = anyhow::Result<RecordBatch>>>;

enum FileMessage {
    Batch(anyhow::Result<RecordBatch>),
    Done,
}

pub struct OrderedParallelBatches {
    receivers: Vec<mpsc::Receiver<FileMessage>>,
    current: usize,
}

impl Iterator for OrderedParallelBatches {
    type Item = anyhow::Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(rx) = self.receivers.get(self.current) {
            match rx.recv() {
                Ok(FileMessage::Batch(batch)) => return Some(batch),
                Ok(FileMessage::Done) | Err(_) => self.current += 1,
            }
        }
        None
    }
}

/// Decode up to `threads` independent files concurrently while yielding their
/// batches in the original file and row order. Each active file may hold one
/// current and one queued Arrow batch.
pub fn ordered_parallel_batches(
    n_files: usize,
    threads: usize,
    open: impl Fn(usize) -> anyhow::Result<BatchIter> + Send + Sync + 'static,
) -> OrderedParallelBatches {
    let mut senders = Vec::with_capacity(n_files);
    let mut receivers = Vec::with_capacity(n_files);
    for _ in 0..n_files {
        let (tx, rx) = mpsc::sync_channel(1);
        senders.push(tx);
        receivers.push(rx);
    }
    let senders = Arc::new(senders);
    let next_file = Arc::new(AtomicUsize::new(0));
    let open = Arc::new(open);
    for _ in 0..threads.max(1).min(n_files.max(1)) {
        let senders = Arc::clone(&senders);
        let next_file = Arc::clone(&next_file);
        let open = Arc::clone(&open);
        std::thread::spawn(move || loop {
            let index = next_file.fetch_add(1, Ordering::Relaxed);
            if index >= senders.len() {
                break;
            }
            let tx = &senders[index];
            match open(index) {
                Ok(iter) => {
                    for batch in iter {
                        let failed = batch.is_err();
                        if tx.send(FileMessage::Batch(batch)).is_err() || failed {
                            break;
                        }
                    }
                }
                Err(error) => {
                    let _ = tx.send(FileMessage::Batch(Err(error)));
                }
            }
            let _ = tx.send(FileMessage::Done);
        });
    }
    OrderedParallelBatches {
        receivers,
        current: 0,
    }
}
