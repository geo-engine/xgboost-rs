extern crate derive_builder;
extern crate indexmap;
extern crate libc;
extern crate log;
extern crate tempfile;
extern crate xgboost_rs_sys;

macro_rules! xgb_call {
    ($x:expr) => {
        XGBError::check_return_value(unsafe { $x })
    };
}

mod error;
pub use error::{XGBError, XGBResult};

mod dmatrix;
pub use dmatrix::DMatrix;

mod booster;
pub use booster::{Booster, FeatureMap, FeatureType};
pub mod parameters;
