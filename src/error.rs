//! Functionality related to errors and error handling.
use snafu::{prelude::*, Backtrace, Error, ResultExt};
use std::ffi::{CStr, CString, NulError};
use std::fmt::{self, Display};
use std::num::TryFromIntError;
use std::str::Utf8Error;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
#[non_exhaustive]
pub enum FfiError {
    #[snafu(display("Could not convert c_str to rust string."))]
    CStringConversion {
        source: NulError,
    },
    NumericConversion {
        source: TryFromIntError,
    },
}

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
#[non_exhaustive]
pub enum DMatrixError {
    #[snafu(display("Could not create a new dmatrix instance."))]
    Dimension {
        source: XGBError,
    },
    #[snafu(display("Could not load a dmatrix from file {}. With error: {}", path.to_str().unwrap(), source.desc))]
    CreateFromFile {
        path: CString,
        source: XGBError,
    },
    #[snafu(display(
        "Could not load or create a dmatrix, due to an error from the ffi: {}",
        source
    ))]
    FfiError {
        source: FfiError,
    },
    CreateNewInstance {
        source: XGBError,
    },
}

impl From<FfiError> for DMatrixError {
    fn from(src: FfiError) -> Self {
        DMatrixError::FfiError {
            source: match src {
                FfiError::CStringConversion { source } => FfiError::CStringConversion { source },
                FfiError::NumericConversion { source } => FfiError::NumericConversion { source },
            },
        }

        // match src {
        //     FfiError::CStringConversion { source } => DMatrixError::FfiError {
        //         source: FfiError::CStringConversion { source },
        //     },
        //     FfiError::NumericConversion { source } => todo!(),
        // }

        // DMatrixError::FfiError {
        //     source: FfiError::CStringConversion {
        //         source: match src {
        //             FfiError::CStringConversion { source } => source,
        //         },
        //     },
        // }
    }
}

pub type DMatrixResult<T> = std::result::Result<T, DMatrixError>;

/// Convenience return type for most operations which can return an `XGBError`.
pub type XGBResult<T> = std::result::Result<T, XGBError>;

// TODO: rename to xgbliberror
/// Wrap errors returned by the `XGBoost` library.
#[derive(Debug, Snafu, Eq, PartialEq)]
pub struct XGBError {
    desc: String,
}

impl XGBError {
    pub(crate) fn new<S: Into<String>>(desc: S) -> Self {
        XGBError { desc: desc.into() }
    }

    /// Check the return value from an `XGBoost` FFI call, and return the last error message on error.
    ///
    /// Return values of 0 are treated as success, returns values of -1 are treated as errors.
    ///
    /// Meaning of any other return values are undefined, and will cause a panic.
    pub(crate) fn check_return_value(ret_val: i32) -> XGBResult<()> {
        match ret_val {
            0 => Ok(()),
            -1 => Err(XGBError::from_xgboost()),
            _ => panic!("unexpected return value '{ret_val}', expected 0 or -1"),
        }
    }

    /// Get the last error message from `XGBoost`.
    fn from_xgboost() -> Self {
        let c_str = unsafe { CStr::from_ptr(xgboost_rs_sys::XGBGetLastError()) };
        let str_slice = c_str.to_str().unwrap();
        XGBError {
            desc: str_slice.to_owned(),
        }
    }
}

// impl Error for XGBError {}
//
// impl Display for XGBError {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         write!(f, "XGBoost error: {}", &self.desc)
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn return_value_handling() {
        let result = XGBError::check_return_value(0);
        assert_eq!(result, Ok(()));

        let result = XGBError::check_return_value(-1);
        assert_eq!(
            result,
            Err(XGBError {
                desc: String::new()
            })
        );
    }
}
