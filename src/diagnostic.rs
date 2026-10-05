//! Observation-only host ranges for private diagnostic builds.
//!
//! `--cfg fb_diagnostic_nvtx` links a separately built NVTX3 C bridge. Ordinary
//! builds compile these guards away: no environment reads, CUDA calls, clocks,
//! synchronization, allocation, or dynamic library loading are introduced.

use std::marker::PhantomData;
use std::rc::Rc;

#[cfg(fb_diagnostic_nvtx)]
extern "C" {
    fn fb_diagnostic_nvtx_push(label: *const std::ffi::c_char);
    fn fb_diagnostic_nvtx_pop();
}

/// A push/pop range must end on the thread that started it, including unwinding.
#[must_use]
pub(crate) struct Phase(PhantomData<Rc<()>>);

impl Phase {
    #[inline]
    pub(crate) fn new(label: &'static [u8]) -> Self {
        #[cfg(fb_diagnostic_nvtx)]
        {
            let label = std::ffi::CStr::from_bytes_with_nul(label)
                .expect("diagnostic phase label must contain exactly one trailing NUL");
            // The checked C string is static; the bridge observes it synchronously.
            unsafe { fb_diagnostic_nvtx_push(label.as_ptr()) };
        }
        #[cfg(not(fb_diagnostic_nvtx))]
        let _ = label;
        Self(PhantomData)
    }
}

impl Drop for Phase {
    #[inline]
    fn drop(&mut self) {
        #[cfg(fb_diagnostic_nvtx)]
        // Phase is !Send and !Sync, preserving NVTX's per-thread range stack.
        unsafe {
            fb_diagnostic_nvtx_pop()
        };
    }
}

#[cfg(all(test, not(fb_diagnostic_nvtx)))]
mod tests {
    use super::Phase;

    #[test]
    fn ordinary_build_ranges_are_zero_sized_and_need_no_bridge() {
        assert_eq!(std::mem::size_of::<Phase>(), 0);
        let outer = Phase::new(b"fb.test.outer\0");
        let inner = Phase::new(b"fb.test.inner\0");
        drop(inner);
        drop(outer);
    }
}
