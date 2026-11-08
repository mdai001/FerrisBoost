//! Typed marker for rejected user input.
//!
//! Core validation (labels, weights, parameters, schemas, base scores, feature
//! types) reports through `anyhow`. The Python layer has to tell "the caller
//! passed something invalid" (`ValueError`) from "something failed at run time"
//! (I/O, CUDA, internal: `RuntimeError`). Sniffing message text is brittle, so
//! validation errors carry this marker and `python::to_py_err` maps on it.
//!
//! Use [`bail_input!`] / [`ensure_input!`] where `anyhow::bail!` /
//! `anyhow::ensure!` would reject the caller's input; the message and the
//! `{:#}` rendering are unchanged.

/// An input the caller must fix. Displays as its message only.
#[derive(Debug)]
pub struct InputError(String);

impl InputError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for InputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InputError {}

/// `anyhow::Error` carrying an [`InputError`].
pub fn input_error(message: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(InputError::new(message))
}

/// Whether any error in the chain is a rejected input.
pub fn is_input_error(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.downcast_ref::<InputError>().is_some())
}

/// `anyhow::bail!` for rejected input.
#[macro_export]
macro_rules! bail_input {
    ($($arg:tt)+) => {
        return ::core::result::Result::Err($crate::error::input_error(format!($($arg)+)))
    };
}

/// `anyhow::ensure!` for rejected input (a message is required).
#[macro_export]
macro_rules! ensure_input {
    ($cond:expr, $($arg:tt)+) => {
        if !($cond) {
            $crate::bail_input!($($arg)+);
        }
    };
}

/// `anyhow::anyhow!` for rejected input (an error value, e.g. in `ok_or_else`).
#[macro_export]
macro_rules! input_err {
    ($($arg:tt)+) => {
        $crate::error::input_error(format!($($arg)+))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(n: i32) -> anyhow::Result<()> {
        ensure_input!(n >= 0, "n must be non-negative, got {n}");
        if n == 0 {
            bail_input!("zero is not allowed");
        }
        Ok(())
    }

    #[test]
    fn rejected_input_is_marked_and_keeps_its_message() {
        let error = check(-3).unwrap_err();
        assert!(is_input_error(&error));
        assert_eq!(format!("{error:#}"), "n must be non-negative, got -3");
        assert!(is_input_error(&check(0).unwrap_err()));
        assert!(check(1).is_ok());
    }

    #[test]
    fn the_marker_survives_added_context() {
        use anyhow::Context;
        let error = check(-1).context("while validating labels").unwrap_err();
        assert!(is_input_error(&error));
        assert!(format!("{error:#}").contains("while validating labels"));
    }

    #[test]
    fn other_errors_are_not_input_errors() {
        assert!(!is_input_error(&anyhow::anyhow!("device lost")));
        let io: anyhow::Error = std::io::Error::other("disk").into();
        assert!(!is_input_error(&io));
        assert!(is_input_error(&crate::input_err!("bad value {}", 7)));
    }
}
