use indexmap::IndexMap;

use log::debug;
use std::collections::{BTreeMap, HashMap};
use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::{ffi, fmt, fs::File, ptr, slice};

use crate::parameters::{BoosterParameters, TrainingParameters};
use crate::{XGBError, XGBResult};

use super::DMatrix;

pub type CustomObjective = fn(&[f32], &DMatrix) -> (Vec<f32>, Vec<f32>);

/// Used to control the return type of predictions made by C Booster API.
enum PredictOption {
    OutputMargin,
    PredictLeaf,
    PredictContribitions,
    //ApproximateContributions,
    PredictInteractions,
}

impl PredictOption {
    /// Convert list of options into a bit mask.
    fn options_as_mask(options: &[PredictOption]) -> i32 {
        let mut option_mask = 0x00;
        for option in options {
            let value = match *option {
                PredictOption::OutputMargin => 0x01,
                PredictOption::PredictLeaf => 0x02,
                PredictOption::PredictContribitions => 0x04,
                //PredictOption::ApproximateContributions => 0x08,
                PredictOption::PredictInteractions => 0x10,
            };
            option_mask |= value;
        }

        option_mask
    }
}

/// Core model in `XGBoost`, containing functions for training, evaluating and predicting.
///
/// Usually created through the [`train`](struct.Booster.html#method.train) function, which
/// creates and trains a Booster in a single call.
///
/// For more fine grained usage, can be created using [`new`](struct.Booster.html#method.new) or
/// [`new_with_cached_dmats`](struct.Booster.html#method.new_with_cached_dmats), then trained by calling
/// [`update`](struct.Booster.html#method.update) or [`update_custom`](struct.Booster.html#method.update_custom)
/// in a loop.
#[derive(Clone)]
pub struct Booster {
    handle: xgboost_rs_sys::BoosterHandle,
}

unsafe impl Send for Booster {}
unsafe impl Sync for Booster {}

impl Booster {
    /// Create a new Booster model with given parameters.
    ///
    /// This model can then be trained using calls to update/boost as appropriate.
    ///
    /// The [`train`](struct.Booster.html#method.train)  function is often a more convenient way of constructing,
    /// training and evaluating a Booster in a single call.
    pub fn new(params: &BoosterParameters) -> XGBResult<Self> {
        Self::new_with_cached_dmats(params, &[])
    }

    pub fn new_with_json_config(
        dmats: &[&DMatrix],

        config: HashMap<&str, &str>,
    ) -> XGBResult<Self> {
        let mut handle = ptr::null_mut();
        // TODO: check this is safe if any dmats are freed
        let s: Vec<xgboost_rs_sys::DMatrixHandle> = dmats.iter().map(|x| x.handle).collect();
        xgb_call!(xgboost_rs_sys::XGBoosterCreate(
            s.as_ptr(),
            dmats.len() as u64,
            &mut handle
        ))?;

        let mut booster = Booster { handle };
        booster.set_param_from_json(config);
        Ok(booster)
    }

    /// Create a new booster model with given parameters and list of `DMatrix` to cache.
    ///
    /// Cached `DMatrix` can sometimes be used internally by `XGBoost` to speed up certain operations.
    pub fn new_with_cached_dmats(
        params: &BoosterParameters,
        dmats: &[&DMatrix],
    ) -> XGBResult<Self> {
        let mut handle = ptr::null_mut();
        // TODO: check this is safe if any dmats are freed
        let s: Vec<xgboost_rs_sys::DMatrixHandle> = dmats.iter().map(|x| x.handle).collect();
        xgb_call!(xgboost_rs_sys::XGBoosterCreate(
            s.as_ptr(),
            dmats.len() as u64,
            &mut handle
        ))?;

        let mut booster = Booster { handle };
        booster.set_params(params)?;
        Ok(booster)
    }

    /// Save this Booster as a binary file at given path.
    ///
    /// # Panics
    ///
    /// Will panic, if the model saving fails with an error not coming from `XGBoost`.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> XGBResult<()> {
        debug!("Writing Booster to: {}", path.as_ref().display());
        let fname = ffi::CString::new(path.as_ref().as_os_str().as_bytes()).unwrap();
        xgb_call!(xgboost_rs_sys::XGBoosterSaveModel(
            self.handle,
            fname.as_ptr()
        ))
    }

    /// Load a `Booster` from a binary file at given path.
    ///
    /// # Panics
    ///
    /// Will panic, if the model couldn't be loaded, because of an error not coming from `XGBoost`.
    /// Could also panic, if a `Booster` couldn't be created because of an error not coming from `XGBoost`.
    pub fn load<P: AsRef<Path>>(path: P) -> XGBResult<Self> {
        debug!("Loading Booster from: {}", path.as_ref().display());

        // gives more control over error messages, avoids stack trace dump from C++
        if !path.as_ref().exists() {
            return Err(XGBError::new(format!(
                "File not found: {}",
                path.as_ref().display()
            )));
        }

        let fname = ffi::CString::new(path.as_ref().as_os_str().as_bytes()).unwrap();
        let mut handle = ptr::null_mut();
        xgb_call!(xgboost_rs_sys::XGBoosterCreate(ptr::null(), 0, &mut handle))?;
        xgb_call!(xgboost_rs_sys::XGBoosterLoadModel(handle, fname.as_ptr()))?;
        Ok(Booster { handle })
    }

    /// Load a Booster directly from a buffer.
    pub fn load_buffer(bytes: &[u8]) -> XGBResult<Self> {
        debug!("Loading Booster from buffer (length = {})", bytes.len());

        let mut handle = ptr::null_mut();
        xgb_call!(xgboost_rs_sys::XGBoosterCreate(ptr::null(), 0, &mut handle))?;
        xgb_call!(xgboost_rs_sys::XGBoosterLoadModelFromBuffer(
            handle,
            bytes.as_ptr().cast(),
            bytes.len() as u64
        ))?;
        Ok(Booster { handle })
    }

    /// Trains the model incrementally.
    ///
    /// # Panics
    ///
    /// Will panic, if `XGBoost` fails to load the number of processes from `Rabit`.
    pub fn train_increment(params: &TrainingParameters, model_name: &str) -> XGBResult<Self> {
        let mut dmats = vec![params.dtrain];
        if let Some(eval_sets) = params.evaluation_sets {
            for (dmat, _) in eval_sets {
                dmats.push(*dmat);
            }
        }

        let path = Path::new(model_name);
        let bytes = std::fs::read(path).expect("can't read saved booster file");
        let mut bst = Booster::load_buffer(&bytes[..]).expect("can't load booster from buffer");

        // load distributed code checkpoint from rabit
        let version = bst.load_rabit_checkpoint()?;
        debug!("Loaded Rabit checkpoint: version={}", version);
        assert!(unsafe { xgboost_rs_sys::RabitGetWorldSize() != 1 || version == 0 });

        unsafe { xgboost_rs_sys::RabitGetRank() };
        let start_iteration = version / 2;

        for i in start_iteration..params.boost_rounds as i32 {
            // distributed code: need to resume to this point
            // skip first update if a recovery step
            if version % 2 == 0 {
                if let Some(objective_fn) = params.custom_objective_fn {
                    debug!("Boosting in round: {}", i);
                    bst.update_custom(params.dtrain, objective_fn)?;
                } else {
                    debug!("Updating in round: {}", i);
                    bst.update(params.dtrain, i)?;
                }
                bst.save_rabit_checkpoint()?;
            }

            assert!(unsafe {
                xgboost_rs_sys::RabitGetWorldSize() == 1
                    || version == xgboost_rs_sys::RabitVersionNumber()
            });

            //nboost += 1;

            if let Some(eval_sets) = params.evaluation_sets {
                let mut dmat_eval_results = bst.eval_set(eval_sets, i)?;

                if let Some(eval_fn) = params.custom_evaluation_fn {
                    let eval_name = "custom";
                    for (dmat, dmat_name) in eval_sets {
                        let margin = bst.predict_margin(dmat)?;
                        let eval_result = eval_fn(&margin, dmat);
                        let eval_results = dmat_eval_results
                            .entry(eval_name.to_string())
                            .or_insert_with(IndexMap::new);
                        eval_results.insert(String::from(*dmat_name), eval_result);
                    }
                }

                // convert to map of eval_name -> (dmat_name -> score)
                let mut eval_dmat_results = BTreeMap::new();
                for (dmat_name, eval_results) in &dmat_eval_results {
                    for (eval_name, result) in eval_results {
                        let dmat_results = eval_dmat_results
                            .entry(eval_name)
                            .or_insert_with(BTreeMap::new);
                        dmat_results.insert(dmat_name, result);
                    }
                }
            }
        }

        Ok(bst)
    }

    pub fn train(
        evaluation_sets: Option<&[(&DMatrix, &str)]>,
        dtrain: &DMatrix,
        config: HashMap<&str, &str>,
        bst: Option<Booster>,
    ) -> XGBResult<Self> {
        let cached_dmats = {
            let mut dmats = vec![dtrain];
            if let Some(eval_sets) = evaluation_sets {
                for (dmat, _) in eval_sets {
                    dmats.push(*dmat);
                }
            }
            dmats
        };

        let mut bst: Booster = {
            if let Some(booster) = bst {
                let mut length: u64 = 0;
                let mut buffer_string = ptr::null();

                xgb_call!(xgboost_rs_sys::XGBoosterSerializeToBuffer(
                    booster.handle,
                    &mut length,
                    &mut buffer_string
                ))
                .expect("couldn't serialize to buffer!");

                let mut bst_handle = ptr::null_mut();

                let cached_dmat_handles: Vec<xgboost_rs_sys::DMatrixHandle> =
                    cached_dmats.iter().map(|x| x.handle).collect();

                xgb_call!(xgboost_rs_sys::XGBoosterCreate(
                    cached_dmat_handles.as_ptr(),
                    cached_dmats.len() as u64,
                    &mut bst_handle
                ))?;

                let mut bst_unserialize = Booster { handle: bst_handle };

                xgb_call!(xgboost_rs_sys::XGBoosterUnserializeFromBuffer(
                    bst_unserialize.handle,
                    buffer_string as *mut ffi::c_void,
                    length,
                ))
                .expect("couldn't unserialize from buffer!");

                bst_unserialize.set_param_from_json(config);
                bst_unserialize
            } else {
                Booster::new_with_json_config(&cached_dmats, config)?
            }
        };

        for i in 0..16 {
            bst.update(dtrain, i)?;

            if let Some(eval_sets) = evaluation_sets {
                let dmat_eval_results = bst.eval_set(eval_sets, i)?;

                // convert to map of eval_name -> (dmat_name -> score)
                let mut eval_dmat_results = BTreeMap::new();
                for (dmat_name, eval_results) in &dmat_eval_results {
                    for (eval_name, result) in eval_results {
                        let dmat_results = eval_dmat_results
                            .entry(eval_name)
                            .or_insert_with(BTreeMap::new);
                        dmat_results.insert(dmat_name, result);
                    }
                }
            }
        }

        Ok(bst)
    }

    /// Saves the config as a json file.
    ///
    /// # Panics
    ///
    /// Will panic, if the config cant be created, because of an error not coming from `XGBoost`.
    pub fn save_config(&self) -> String {
        let mut length: u64 = 1;
        let mut json_string = ptr::null();

        let _json = unsafe {
            xgboost_rs_sys::XGBoosterSaveJsonConfig(self.handle, &mut length, &mut json_string)
        };

        let out = unsafe {
            ffi::CStr::from_ptr(json_string)
                .to_str()
                .unwrap()
                .to_owned()
        };

        out
    }

    /// Update this Booster's parameters.
    pub fn set_params(&mut self, p: &BoosterParameters) -> XGBResult<()> {
        for (key, value) in p.as_string_pairs() {
            self.set_param(&key, &value)?;
        }
        Ok(())
    }

    /// Update this model by training it for one round with given training matrix.
    ///
    /// Uses `XGBoost`'s objective function that was specificed in this Booster's learning objective parameters.
    ///
    /// * `dtrain` - matrix to train the model with for a single iteration
    /// * `iteration` - current iteration number
    pub fn update(&mut self, dtrain: &DMatrix, iteration: i32) -> XGBResult<()> {
        xgb_call!(xgboost_rs_sys::XGBoosterUpdateOneIter(
            self.handle,
            iteration,
            dtrain.handle
        ))
    }

    /// Update this model by training it for one round with a custom objective function.
    pub fn update_custom(
        &mut self,
        dtrain: &DMatrix,
        objective_fn: CustomObjective,
    ) -> XGBResult<()> {
        let pred = self.predict(dtrain)?;
        let (gradient, hessian) = objective_fn(&pred, dtrain);
        self.boost(dtrain, &gradient, &hessian)
    }

    /// Update this model by directly specifying the first and second order gradients.
    ///
    /// This is typically used instead of `update` when using a customised loss function.
    ///
    /// * `dtrain` - matrix to train the model with for a single iteration
    /// * `gradient` - first order gradient
    /// * `hessian` - second order gradient
    fn boost(&mut self, dtrain: &DMatrix, gradient: &[f32], hessian: &[f32]) -> XGBResult<()> {
        if gradient.len() != hessian.len() {
            let msg = format!(
                "Mismatch between length of gradient and hessian arrays ({} != {})",
                gradient.len(),
                hessian.len()
            );
            return Err(XGBError::new(msg));
        }
        assert_eq!(gradient.len(), hessian.len());

        // TODO: _validate_feature_names
        let mut grad_vec = gradient.to_vec();
        let mut hess_vec = hessian.to_vec();
        xgb_call!(xgboost_rs_sys::XGBoosterBoostOneIter(
            self.handle,
            dtrain.handle,
            grad_vec.as_mut_ptr(),
            hess_vec.as_mut_ptr(),
            grad_vec.len() as u64
        ))
    }

    fn eval_set(
        &self,
        evals: &[(&DMatrix, &str)],
        iteration: i32,
    ) -> XGBResult<IndexMap<String, IndexMap<String, f32>>> {
        let (dmats, names) = {
            let mut dmats = Vec::with_capacity(evals.len());
            let mut names = Vec::with_capacity(evals.len());
            for (dmat, name) in evals {
                dmats.push(dmat);
                names.push(*name);
            }
            (dmats, names)
        };
        assert_eq!(dmats.len(), names.len());

        let mut s: Vec<xgboost_rs_sys::DMatrixHandle> = dmats.iter().map(|x| x.handle).collect();

        // build separate arrays of C strings and pointers to them to ensure they live long enough
        let mut evnames: Vec<ffi::CString> = Vec::with_capacity(names.len());
        let mut evptrs: Vec<*const libc::c_char> = Vec::with_capacity(names.len());

        for name in &names {
            let cstr = ffi::CString::new(*name).unwrap();
            evptrs.push(cstr.as_ptr());
            evnames.push(cstr);
        }

        // shouldn't be necessary, but guards against incorrect array sizing
        evptrs.shrink_to_fit();

        let mut out_result = ptr::null();
        xgb_call!(xgboost_rs_sys::XGBoosterEvalOneIter(
            self.handle,
            iteration,
            s.as_mut_ptr(),
            evptrs.as_mut_ptr(),
            dmats.len() as u64,
            &mut out_result
        ))?;
        let out = unsafe { ffi::CStr::from_ptr(out_result).to_str().unwrap().to_owned() };
        Ok(Booster::parse_eval_string(&out, &names))
    }

    /// Evaluate given matrix against this model using metrics defined in this model's parameters.
    ///
    /// See `parameter::learning::EvaluationMetric` for a full list.
    ///
    /// Returns a map of evaluation metric name to score.
    ///
    /// # Panics
    ///
    /// Will panic, if the given matrix cannot be evaluated with the given metric.
    pub fn evaluate(&self, dmat: &DMatrix, name: &str) -> XGBResult<HashMap<String, f32>> {
        let mut eval = self.eval_set(&[(dmat, name)], 0)?;
        let mut result = HashMap::new();
        eval.remove(name).unwrap().into_iter().for_each(|(k, v)| {
            result.insert(k, v);
        });

        Ok(result)
    }

    /// Get a string attribute that was previously set for this model.
    ///
    /// # Panics
    ///
    /// Will panic, if the attribute can't be retrieved, or the key can't be represented
    /// as a `CString`.
    pub fn get_attribute(&self, key: &str) -> XGBResult<Option<String>> {
        let key = ffi::CString::new(key).unwrap();
        let mut out_buf = ptr::null();
        let mut success = 0;
        xgb_call!(xgboost_rs_sys::XGBoosterGetAttr(
            self.handle,
            key.as_ptr(),
            &mut out_buf,
            &mut success
        ))?;
        if success == 0 {
            return Ok(None);
        }
        assert!(success == 1);

        let c_str: &ffi::CStr = unsafe { ffi::CStr::from_ptr(out_buf) };
        let out = c_str.to_str().unwrap();
        Ok(Some(out.to_owned()))
    }

    /// Store a string attribute in this model with given key.
    ///
    /// # Panics
    ///
    /// Will panic, if the attribute can't be set by `XGBoost`.
    pub fn set_attribute(&mut self, key: &str, value: &str) -> XGBResult<()> {
        let key = ffi::CString::new(key).unwrap();
        let value = ffi::CString::new(value).unwrap();
        xgb_call!(xgboost_rs_sys::XGBoosterSetAttr(
            self.handle,
            key.as_ptr(),
            value.as_ptr()
        ))
    }

    /// Get names of all attributes stored in this model. Values can then be fetched with calls to `get_attribute`.
    ///
    /// # Panics
    ///
    /// Will panic, if the attribtue name cannot be retrieved from `XGBoost`.
    pub fn get_attribute_names(&self) -> XGBResult<Vec<String>> {
        let mut out_len = 0;
        let mut out = ptr::null_mut();
        xgb_call!(xgboost_rs_sys::XGBoosterGetAttrNames(
            self.handle,
            &mut out_len,
            &mut out
        ))?;

        let out_ptr_slice = unsafe { slice::from_raw_parts(out, out_len as usize) };
        let out_vec = out_ptr_slice
            .iter()
            .map(|str_ptr| unsafe { ffi::CStr::from_ptr(*str_ptr).to_str().unwrap().to_owned() })
            .collect();
        Ok(out_vec)
    }

    /// This method calculates the predicions from a given matrix.
    ///
    /// # Panics
    ///
    /// Will panic, if `XGBoost` cannot make predictions.
    pub fn predict_from_dmat(
        &self,
        dmat: &DMatrix,
        out_shape: &[u64; 2],
        out_dim: &mut u64,
    ) -> XGBResult<Vec<f32>> {
        let json_config = "{\"type\": 0,\"training\": false,\"iteration_begin\": 0,\"iteration_end\": 0,\"strict_shape\": true}".to_string();

        let mut out_result = ptr::null();

        let c_json_config = ffi::CString::new(json_config).unwrap();

        xgb_call!(xgboost_rs_sys::XGBoosterPredictFromDMatrix(
            self.handle,
            dmat.handle,
            c_json_config.as_ptr(),
            &mut out_shape.as_ptr(),
            out_dim,
            &mut out_result
        ))?;

        let out_len = out_shape[0];

        assert!(!out_result.is_null());
        let data = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };
        Ok(data)
    }

    /// Predict results for given data.
    ///
    /// Returns an array containing one entry per row in the given data.
    ///
    /// # Panics
    ///
    /// Will panic, if the predictions aren't possible for `XGBoost` or the results cannot be
    /// parsed.
    pub fn predict(&self, dmat: &DMatrix) -> XGBResult<Vec<f32>> {
        let option_mask = PredictOption::options_as_mask(&[]);
        let ntree_limit = 0;
        let mut out_len = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_rs_sys::XGBoosterPredict(
            self.handle,
            dmat.handle,
            option_mask,
            ntree_limit,
            0,
            &mut out_len,
            &mut out_result
        ))?;

        assert!(!out_result.is_null());
        let data = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };
        Ok(data)
    }

    /// Predict margin for given data.
    ///
    /// Returns an array containing one entry per row in the given data.
    ///
    /// # Panics
    ///
    /// Will panic, if the predictions aren't possible for `XGBoost` or the results cannot be
    /// parsed.
    pub fn predict_margin(&self, dmat: &DMatrix) -> XGBResult<Vec<f32>> {
        let option_mask = PredictOption::options_as_mask(&[PredictOption::OutputMargin]);
        let ntree_limit = 0;
        let mut out_len = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_rs_sys::XGBoosterPredict(
            self.handle,
            dmat.handle,
            option_mask,
            ntree_limit,
            1,
            &mut out_len,
            &mut out_result
        ))?;
        assert!(!out_result.is_null());
        let data = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };
        Ok(data)
    }

    /// Get predicted leaf index for each sample in given data.
    ///
    /// Returns an array of shape (number of samples, number of trees) as tuple of (data, `num_rows`).
    ///
    /// Note: the leaf index of a tree is unique per tree, so e.g. leaf 1 could be found in both tree 1 and tree 0.
    ///
    /// # Panics
    ///
    /// Will panic, if the prediction of a leave isn't possible for `XGBoost` or the data cannot be
    /// parsed.
    pub fn predict_leaf(&self, dmat: &DMatrix) -> XGBResult<(Vec<f32>, (usize, usize))> {
        let option_mask = PredictOption::options_as_mask(&[PredictOption::PredictLeaf]);
        let ntree_limit = 0;
        let mut out_len = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_rs_sys::XGBoosterPredict(
            self.handle,
            dmat.handle,
            option_mask,
            ntree_limit,
            0,
            &mut out_len,
            &mut out_result
        ))?;
        assert!(!out_result.is_null());

        let data = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };
        let num_rows = dmat.num_rows();
        let num_cols = data.len() / num_rows;
        Ok((data, (num_rows, num_cols)))
    }

    /// Get feature contributions (SHAP values) for each prediction.
    ///
    /// The sum of all feature contributions is equal to the run untransformed margin value of the
    /// prediction.
    ///
    /// Returns an array of shape (number of samples, number of features + 1) as a tuple of
    /// (data, `num_rows`). The final column contains the bias term.
    ///
    /// # Panics
    ///
    /// Will panic, if `XGBoost` cannot predict the data or parse the result.
    pub fn predict_contributions(&self, dmat: &DMatrix) -> XGBResult<(Vec<f32>, (usize, usize))> {
        let option_mask = PredictOption::options_as_mask(&[PredictOption::PredictContribitions]);
        let ntree_limit = 0;
        let mut out_len = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_rs_sys::XGBoosterPredict(
            self.handle,
            dmat.handle,
            option_mask,
            ntree_limit,
            0,
            &mut out_len,
            &mut out_result
        ))?;
        assert!(!out_result.is_null());

        let data = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };
        let num_rows = dmat.num_rows();
        let num_cols = data.len() / num_rows;
        Ok((data, (num_rows, num_cols)))
    }

    /// Get SHAP interaction values for each pair of features for each prediction.
    ///
    /// The sum of each row (or column) of the interaction values equals the corresponding SHAP
    /// value (from `predict_contributions`), and the sum of the entire matrix equals the raw
    /// untransformed margin value of the prediction.
    ///
    /// Returns an array of shape (number of samples, number of features + 1, number of features + 1).
    /// The final row and column contain the bias terms.
    ///
    /// # Panics
    ///
    /// Will panic, if `XGBoost` cannot predict the data or parse the result.
    pub fn predict_interactions(
        &self,
        dmat: &DMatrix,
    ) -> XGBResult<(Vec<f32>, (usize, usize, usize))> {
        let option_mask = PredictOption::options_as_mask(&[PredictOption::PredictInteractions]);
        let ntree_limit = 0;
        let mut out_len = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_rs_sys::XGBoosterPredict(
            self.handle,
            dmat.handle,
            option_mask,
            ntree_limit,
            0,
            &mut out_len,
            &mut out_result
        ))?;
        assert!(!out_result.is_null());

        let data = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };
        let num_rows = dmat.num_rows();

        let dim = ((data.len() / num_rows) as f64).sqrt() as usize;
        Ok((data, (num_rows, dim, dim)))
    }

    /// Get a dump of this model as a string.
    ///
    /// * `with_statistics` - whether to include statistics in output dump
    /// * `feature_map` - if given, map feature IDs to feature names from given map
    pub fn dump_model(
        &self,
        with_statistics: bool,
        feature_map: Option<&FeatureMap>,
    ) -> XGBResult<String> {
        if let Some(fmap) = feature_map {
            let tmp_dir = match tempfile::tempdir() {
                Ok(dir) => dir,
                Err(err) => return Err(XGBError::new(err.to_string())),
            };

            let file_path = tmp_dir.path().join("fmap.txt");
            let mut file: File = match File::create(&file_path) {
                Ok(f) => f,
                Err(err) => return Err(XGBError::new(err.to_string())),
            };

            for (feature_num, (feature_name, feature_type)) in &fmap.0 {
                writeln!(file, "{}\t{}\t{}", feature_num, feature_name, feature_type).unwrap();
            }

            self.dump_model_fmap(with_statistics, Some(&file_path))
        } else {
            self.dump_model_fmap(with_statistics, None)
        }
    }

    fn dump_model_fmap(
        &self,
        with_statistics: bool,
        feature_map_path: Option<&PathBuf>,
    ) -> XGBResult<String> {
        let fmap = if let Some(path) = feature_map_path {
            ffi::CString::new(path.as_os_str().as_bytes()).unwrap()
        } else {
            ffi::CString::new("").unwrap()
        };
        let format = ffi::CString::new("text").unwrap();
        let mut out_len = 0;
        let mut out_dump_array = ptr::null_mut();
        xgb_call!(xgboost_rs_sys::XGBoosterDumpModelEx(
            self.handle,
            fmap.as_ptr(),
            i32::from(with_statistics),
            format.as_ptr(),
            &mut out_len,
            &mut out_dump_array
        ))?;

        let out_ptr_slice = unsafe { slice::from_raw_parts(out_dump_array, out_len as usize) };
        let out_vec: Vec<String> = out_ptr_slice
            .iter()
            .map(|str_ptr| unsafe { ffi::CStr::from_ptr(*str_ptr).to_str().unwrap().to_owned() })
            .collect();

        assert_eq!(out_len as usize, out_vec.len());
        Ok(out_vec.join("\n"))
    }

    pub(crate) fn load_rabit_checkpoint(&self) -> XGBResult<i32> {
        let mut version = 0;
        xgb_call!(xgboost_rs_sys::XGBoosterLoadRabitCheckpoint(
            self.handle,
            &mut version
        ))?;
        Ok(version)
    }

    pub(crate) fn save_rabit_checkpoint(&self) -> XGBResult<()> {
        xgb_call!(xgboost_rs_sys::XGBoosterSaveRabitCheckpoint(self.handle))
    }

    /// Sets the parameters for `XGBoost` from a json file.
    ///
    /// # Panics
    ///
    /// Will panic, if `XGBoost` cannot set the values.
    fn set_param_from_json(&mut self, config: HashMap<&str, &str>) {
        for (k, v) in config {
            let name = ffi::CString::new(k).unwrap();
            let value = ffi::CString::new(v).unwrap();

            unsafe {
                xgboost_rs_sys::XGBoosterSetParam(self.handle, name.as_ptr(), value.as_ptr())
            };
        }
    }

    fn set_param(&mut self, name: &str, value: &str) -> XGBResult<()> {
        let name = ffi::CString::new(name).unwrap();
        let value = ffi::CString::new(value).unwrap();
        xgb_call!(xgboost_rs_sys::XGBoosterSetParam(
            self.handle,
            name.as_ptr(),
            value.as_ptr()
        ))
    }

    fn parse_eval_string(eval: &str, evnames: &[&str]) -> IndexMap<String, IndexMap<String, f32>> {
        let mut result: IndexMap<String, IndexMap<String, f32>> = IndexMap::new();

        debug!("Parsing evaluation line: {}", &eval);
        for part in eval.split('\t').skip(1) {
            for evname in evnames {
                if part.starts_with(evname) {
                    let metric_parts: Vec<&str> =
                        part[evname.len() + 1..].split(':').into_iter().collect();
                    assert_eq!(metric_parts.len(), 2);
                    let metric = metric_parts[0];
                    let score = metric_parts[1].parse::<f32>().unwrap_or_else(|_| {
                        panic!("Unable to parse XGBoost metrics output: {}", eval)
                    });

                    let metric_map = result
                        .entry(String::from(*evname))
                        .or_insert_with(IndexMap::new);
                    metric_map.insert(metric.to_owned(), score);
                }
            }
        }

        debug!("result: {:?}", &result);
        result
    }
}

impl Drop for Booster {
    fn drop(&mut self) {
        xgb_call!(xgboost_rs_sys::XGBoosterFree(self.handle)).unwrap();
    }
}

/// Maps a feature index to a name and type, used when dumping models as text.
///
/// See [`dump_model`](struct.Booster.html#method.dump_model) for usage.
pub struct FeatureMap(BTreeMap<u32, (String, FeatureType)>);

impl FeatureMap {
    /// Read a `FeatureMap` from a file at given path.
    ///
    /// File should contain one feature definition per line, and be of the form:
    /// ```text
    /// <number>\t<name>\t<type>\n
    /// ```
    ///
    /// Type should be one of:
    /// * `i` - binary feature
    /// * `q` - quantitative feature
    /// * `int` - integer features
    ///
    /// E.g.:
    /// ```text
    /// 0   age int
    /// 1   is-parent?=yes  i
    /// 2   is-parent?=no   i
    /// 3   income  int
    /// ```
    ///
    /// # Panics
    ///
    /// Will panic, if the given `FeatureMap` file cannot be loaded.
    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<FeatureMap> {
        let file = File::open(path)?;
        let mut features: FeatureMap = FeatureMap(BTreeMap::new());

        for (i, line) in BufReader::new(&file).lines().enumerate() {
            let line = line?;
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() != 3 {
                let msg = format!(
                    "Unable to parse features from line {}, expected 3 tab separated values",
                    i + 1
                );
                return Err(io::Error::new(io::ErrorKind::InvalidData, msg));
            }

            assert_eq!(parts.len(), 3);
            let feature_num: u32 = match parts[0].parse() {
                Ok(num) => num,
                Err(err) => {
                    let msg = format!(
                        "Unable to parse features from line {}, could not parse feature number: {}",
                        i + 1,
                        err
                    );
                    return Err(io::Error::new(io::ErrorKind::InvalidData, msg));
                }
            };

            let feature_name = parts[1];
            let feature_type = match FeatureType::from_str(parts[2]) {
                Ok(feature_type) => feature_type,
                Err(msg) => {
                    let msg = format!("Unable to parse features from line {}: {}", i + 1, msg);
                    return Err(io::Error::new(io::ErrorKind::InvalidData, msg));
                }
            };
            features
                .0
                .insert(feature_num, (feature_name.to_string(), feature_type));
        }
        Ok(features)
    }
}

/// Indicates the type of a feature, used when dumping models as text.
pub enum FeatureType {
    /// Binary indicator feature.
    Binary,

    /// Quantitative feature (e.g. age, time, etc.), can be missing.
    Quantitative,

    /// Integer feature (when hinted, decision boundary will be integer).
    Integer,
}

impl FromStr for FeatureType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "i" => Ok(FeatureType::Binary),
            "q" => Ok(FeatureType::Quantitative),
            "int" => Ok(FeatureType::Integer),
            _ => Err(format!(
                "unrecognised feature type '{}', must be one of: 'i', 'q', 'int'",
                s
            )),
        }
    }
}

impl fmt::Display for FeatureType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            FeatureType::Binary => "i",
            FeatureType::Quantitative => "q",
            FeatureType::Integer => "int",
        };
        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use indexmap::IndexMap;
    use ndarray::arr2;

    use crate::{
        parameters::{self, learning, tree, BoosterParameters},
        Booster, DMatrix, XGBResult,
    };

    fn read_train_matrix() -> XGBResult<DMatrix> {
        let data_path = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
        DMatrix::load(format!("{}/data.csv?format=csv", data_path))
    }

    fn load_test_booster() -> Booster {
        let dmat = read_train_matrix().expect("Reading train matrix failed");
        Booster::new_with_cached_dmats(&BoosterParameters::default(), &[&dmat])
            .expect("Creating Booster failed")
    }

    #[test]
    fn set_booster_parhm() {
        let mut booster = load_test_booster();
        let res = booster.set_param("key", "value");
        assert!(res.is_ok());
    }

    #[test]
    fn load_rabit_version() {
        let version = load_test_booster().load_rabit_checkpoint().unwrap();
        assert_eq!(version, 0);
    }

    #[test]
    fn get_set_attr() {
        let mut booster = load_test_booster();
        let attr = booster
            .get_attribute("foo")
            .expect("Getting attribute failed");
        assert_eq!(attr, None);

        booster
            .set_attribute("foo", "bar")
            .expect("Setting attribute failed");
        let attr = booster
            .get_attribute("foo")
            .expect("Getting attribute failed");
        assert_eq!(attr, Some("bar".to_owned()));
    }

    #[test]
    fn save_and_load_from_buffer() {
        let data_path = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
        let dmat_train =
            DMatrix::load(format!("{}/agaricus.txt.train?format=libsvm", data_path)).unwrap();
        let mut booster =
            Booster::new_with_cached_dmats(&BoosterParameters::default(), &[&dmat_train]).unwrap();
        let attr = booster
            .get_attribute("foo")
            .expect("Getting attribute failed");
        assert_eq!(attr, None);

        booster
            .set_attribute("foo", "bar")
            .expect("Setting attribute failed");
        let attr = booster
            .get_attribute("foo")
            .expect("Getting attribute failed");
        assert_eq!(attr, Some("bar".to_owned()));

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("test-xgboost-model");
        booster.save(&path).expect("saving booster failed");
        drop(booster);
        let bytes = std::fs::read(&path).expect("reading saved booster file failed");
        let booster = Booster::load_buffer(&bytes[..]).expect("loading booster from buffer failed");
        let attr = booster
            .get_attribute("foo")
            .expect("Getting attribute failed");
        assert_eq!(attr, Some("bar".to_owned()));
    }

    #[test]
    fn get_attribute_names() {
        let mut booster = load_test_booster();
        let attrs = booster
            .get_attribute_names()
            .expect("Getting attributes failed");
        assert_eq!(attrs, Vec::<String>::new());

        booster
            .set_attribute("foo", "bar")
            .expect("Setting attribute failed");
        booster
            .set_attribute("another", "another")
            .expect("Setting attribute failed");
        booster
            .set_attribute("4", "4")
            .expect("Setting attribute failed");
        booster
            .set_attribute("an even longer attribute name?", "")
            .expect("Setting attribute failed");

        let mut expected = vec!["foo", "another", "4", "an even longer attribute name?"];
        expected.sort_unstable();
        let mut attrs = booster
            .get_attribute_names()
            .expect("Getting attributes failed");
        attrs.sort();
        assert_eq!(attrs, expected);
    }

    #[test]
    fn predict() {
        let data_path = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
        let dmat_train =
            DMatrix::load(format!("{}/agaricus.txt.train?format=libsvm", data_path)).unwrap();
        let dmat_test =
            DMatrix::load(format!("{}/agaricus.txt.test?format=libsvm", data_path)).unwrap();

        let tree_params = tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build()
            .unwrap();
        let learning_params = learning::LearningTaskParametersBuilder::default()
            .objective(learning::Objective::BinaryLogistic)
            .eval_metrics(learning::Metrics::Custom(vec![
                learning::EvaluationMetric::MapCutNegative(4),
                learning::EvaluationMetric::LogLoss,
                learning::EvaluationMetric::BinaryErrorRate(0.5),
            ]))
            .build()
            .unwrap();
        let params = parameters::BoosterParametersBuilder::default()
            .booster_type(parameters::BoosterType::Tree(tree_params))
            .learning_params(learning_params)
            .verbose(false)
            .build()
            .unwrap();
        let mut booster =
            Booster::new_with_cached_dmats(&params, &[&dmat_train, &dmat_test]).unwrap();

        for i in 0..10 {
            booster.update(&dmat_train, i).expect("update failed");
        }

        let eps = 1e-6;

        let train_metrics = booster.evaluate(&dmat_train, "default").unwrap();
        assert!(*train_metrics.get("logloss").unwrap() - 0.006_634 < eps);
        assert!(*train_metrics.get("map@4-").unwrap() - 0.001_274 < eps);

        let test_metrics = booster.evaluate(&dmat_test, "default").unwrap();
        assert!(*test_metrics.get("logloss").unwrap() - 0.006_92 < eps);
        assert!(*test_metrics.get("map@4-").unwrap() - 0.005_155 < eps);

        let v = booster.predict(&dmat_test).unwrap();
        assert_eq!(v.len(), dmat_test.num_rows());

        // first 10 predictions
        let expected_start = [
            0.005_015_169_3,
            0.988_446_7,
            0.005_015_169_3,
            0.005_015_169_3,
            0.026_636_455,
            0.117_893_63,
            0.988_446_7,
            0.012_314_71,
            0.988_446_7,
            0.000_136_560_63,
        ];

        // last 10 predictions
        let expected_end = [
            0.002_520_344,
            0.000_609_179_26,
            0.998_810_05,
            0.000_609_179_26,
            0.000_609_179_26,
            0.000_609_179_26,
            0.000_609_179_26,
            0.998_110_2,
            0.002_855_195,
            0.998_110_2,
        ];

        for (pred, expected) in v.iter().zip(&expected_start) {
            assert!(pred - expected < eps);
        }

        for (pred, expected) in v[v.len() - 10..].iter().zip(&expected_end) {
            assert!(pred - expected < eps);
        }
    }

    #[test]
    fn predict_leaf() {
        let data_path = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
        let dmat_train =
            DMatrix::load(format!("{}/agaricus.txt.train?format=libsvm", data_path)).unwrap();
        let dmat_test =
            DMatrix::load(format!("{}/agaricus.txt.test?format=libsvm", data_path)).unwrap();

        let tree_params = tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build()
            .unwrap();
        let learning_params = learning::LearningTaskParametersBuilder::default()
            .objective(learning::Objective::BinaryLogistic)
            .eval_metrics(learning::Metrics::Custom(vec![
                learning::EvaluationMetric::LogLoss,
            ]))
            .build()
            .unwrap();
        let params = parameters::BoosterParametersBuilder::default()
            .booster_type(parameters::BoosterType::Tree(tree_params))
            .learning_params(learning_params)
            .verbose(false)
            .build()
            .unwrap();
        let mut booster =
            Booster::new_with_cached_dmats(&params, &[&dmat_train, &dmat_test]).unwrap();

        let num_rounds = 15;
        for i in 0..num_rounds {
            booster.update(&dmat_train, i).expect("update failed");
        }

        let (_preds, shape) = booster.predict_leaf(&dmat_test).unwrap();
        let num_samples = dmat_test.num_rows();
        assert_eq!(shape, (num_samples, num_rounds as usize));
    }

    #[test]
    fn predict_contributions() {
        let data_path = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
        let dmat_train =
            DMatrix::load(format!("{}/agaricus.txt.train?format=libsvm", data_path)).unwrap();
        let dmat_test =
            DMatrix::load(format!("{}/agaricus.txt.test?format=libsvm", data_path)).unwrap();

        let tree_params = tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build()
            .unwrap();
        let learning_params = learning::LearningTaskParametersBuilder::default()
            .objective(learning::Objective::BinaryLogistic)
            .eval_metrics(learning::Metrics::Custom(vec![
                learning::EvaluationMetric::LogLoss,
            ]))
            .build()
            .unwrap();
        let params = parameters::BoosterParametersBuilder::default()
            .booster_type(parameters::BoosterType::Tree(tree_params))
            .learning_params(learning_params)
            .verbose(false)
            .build()
            .unwrap();
        let mut booster =
            Booster::new_with_cached_dmats(&params, &[&dmat_train, &dmat_test]).unwrap();

        let num_rounds = 5;
        for i in 0..num_rounds {
            booster.update(&dmat_train, i).expect("update failed");
        }

        let (_preds, shape) = booster.predict_contributions(&dmat_test).unwrap();
        let num_samples = dmat_test.num_rows();
        let num_features = dmat_train.num_cols();
        assert_eq!(shape, (num_samples, num_features + 1));
    }

    #[test]
    fn predict_interactions() {
        let data_path = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
        let dmat_train =
            DMatrix::load(format!("{}/agaricus.txt.train?format=libsvm", data_path)).unwrap();
        let dmat_test =
            DMatrix::load(format!("{}/agaricus.txt.test?format=libsvm", data_path)).unwrap();

        let tree_params = tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build()
            .unwrap();
        let learning_params = learning::LearningTaskParametersBuilder::default()
            .objective(learning::Objective::BinaryLogistic)
            .eval_metrics(learning::Metrics::Custom(vec![
                learning::EvaluationMetric::LogLoss,
            ]))
            .build()
            .unwrap();
        let params = parameters::BoosterParametersBuilder::default()
            .booster_type(parameters::BoosterType::Tree(tree_params))
            .learning_params(learning_params)
            .verbose(false)
            .build()
            .unwrap();
        let mut booster =
            Booster::new_with_cached_dmats(&params, &[&dmat_train, &dmat_test]).unwrap();

        let num_rounds = 5;
        for i in 0..num_rounds {
            booster.update(&dmat_train, i).expect("update failed");
        }

        let (_preds, shape) = booster.predict_interactions(&dmat_test).unwrap();
        let num_samples = dmat_test.num_rows();
        let num_features = dmat_train.num_cols();
        assert_eq!(shape, (num_samples, num_features + 1, num_features + 1));
    }

    #[test]
    fn parse_eval_string() {
        let s = "[0]\ttrain-map@4-:0.5\ttrain-logloss:1.0\ttest-map@4-:0.25\ttest-logloss:0.75";
        let mut metrics = IndexMap::new();

        let mut train_metrics = IndexMap::new();
        train_metrics.insert("map@4-".to_owned(), 0.5);
        train_metrics.insert("logloss".to_owned(), 1.0);

        let mut test_metrics = IndexMap::new();
        test_metrics.insert("map@4-".to_owned(), 0.25);
        test_metrics.insert("logloss".to_owned(), 0.75);

        metrics.insert("train".to_owned(), train_metrics);
        metrics.insert("test".to_owned(), test_metrics);
        assert_eq!(Booster::parse_eval_string(s, &["train", "test"]), metrics);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn pred_from_dmat() {
        let data_arr_2d = arr2(&[
            [
                8.325_2,
                4.1e+01,
                6.984_127,
                1.023_809_6,
                3.22e+02,
                2.555_555_6,
                3.788e+01,
                -1.222_3e+02,
            ],
            [
                8.301_4,
                2.1e+01,
                6.238_137_2,
                9.718_805e-1,
                2.401e+03,
                2.109_841_8,
                3.786e+01,
                -1.222_2e+02,
            ],
            [
                7.257_4,
                5.2e+01,
                8.288_136,
                1.073_446_3,
                4.96e+02,
                2.802_26,
                3.785e+01,
                -1.222_4e+02,
            ],
            [
                5.643_1,
                5.2e+01,
                5.817_352,
                1.073_059_3,
                5.58e+02,
                2.547_945_3,
                3.785e+01,
                -1.222_5e+02,
            ],
            [
                3.846_2,
                5.2e+01,
                6.281_853,
                1.081_081,
                5.65e+02,
                2.181_467_3,
                3.785e+01,
                -1.222_5e+02,
            ],
            [
                4.036_8,
                5.2e+01,
                4.761_658,
                1.103_627,
                4.13e+02,
                2.139_896_4,
                3.785e+01,
                -1.222_5e+02,
            ],
            [
                3.659_1,
                5.2e+01,
                4.931_906_7,
                9.513_619e-1,
                1.094e+03,
                2.128_404_6,
                3.784e+01,
                -1.222_5e+02,
            ],
            [
                3.12,
                5.2e1,
                4.797_527,
                1.061_823_8,
                1.157e3,
                1.788_253_4,
                3.784e1,
                -1.222_5e2,
            ],
            [
                2.080_4,
                4.2e1,
                4.294_117_5,
                1.117_647,
                1.206e3,
                2.026_890_8,
                3.784e1,
                -1.222_6e2,
            ],
            [
                3.691_2,
                5.2e1,
                4.970_588,
                9.901_960_5e-1,
                1.551e3,
                2.172_268_9,
                3.784e1,
                -1.222_5e2,
            ],
            [
                3.203_1,
                5.2e1,
                5.477_612,
                1.079_602,
                9.1e2,
                2.263_681_7,
                3.785e1,
                -1.222_6e2,
            ],
            [
                3.270_5,
                5.2e1,
                4.772_479_5,
                1.024_523_1,
                1.504e3,
                2.049_046_3,
                3.785e1,
                -1.222_6e2,
            ],
            [
                3.075,
                5.2e1,
                5.322_649_5,
                1.012_820_5,
                1.098e+03,
                2.346_153_7,
                3.785e1,
                -1.222_6e2,
            ],
            [
                2.673_6,
                5.2e1,
                4.0,
                1.097_701_2,
                3.45e+02,
                1.982_758_6,
                3.784e1,
                -1.222_6e2,
            ],
            [
                1.916_7,
                5.2e1,
                4.262_903,
                1.009_677_4,
                1.212e+03,
                1.954_838_8,
                3.785e+01,
                -1.222_6e2,
            ],
            [
                2.125,
                5.0e+01,
                4.242_424,
                1.071_969_7,
                6.97e+02,
                2.640_151_5,
                3.785e+01,
                -1.222_6e2,
            ],
            [
                2.775,
                5.2e1,
                5.939_577,
                1.048_338_4,
                7.93e2,
                2.395_770_3,
                3.785e1,
                -1.222_7e2,
            ],
            [
                2.120_2,
                5.2e1,
                4.052_805_4,
                9.669_967e-1,
                6.48e2,
                2.138_614,
                3.785e1,
                -1.222_7e2,
            ],
            [
                1.991_1,
                5.0e1,
                5.343_675_6,
                1.085_918_9,
                9.9e2,
                2.362_768_4,
                3.784e1,
                -1.222_6e2,
            ],
            [
                2.603_3,
                5.2e1,
                5.465_454_6,
                1.083_636_4,
                6.9e2,
                2.509_091,
                3.784e1,
                -1.222_7e2,
            ],
            [
                1.357_8,
                4.0e1,
                4.524_096_5,
                1.108_433_7,
                4.09e2,
                2.463_855_5,
                3.785e1,
                -1.222_7e2,
            ],
            [
                1.713_5,
                4.2e1,
                4.478_142_3,
                1.002_732_3,
                9.29e2,
                2.538_251_4,
                3.785e1,
                -1.222_7e2,
            ],
            [
                1.725,
                5.2e1,
                5.096_234_3,
                1.131_799_1,
                1.015e3,
                2.123_431,
                3.784e1,
                -1.222_7e2,
            ],
            [
                2.180_6,
                5.2e1,
                5.193_846,
                1.036_923,
                8.53e2,
                2.624_615_4,
                3.784e1,
                -1.222_7e2,
            ],
            [
                2.6,
                5.2e1,
                5.270_142,
                1.035_545,
                1.006e3,
                2.383_886_3,
                3.784e1,
                -1.222_7e2,
            ],
            [
                2.403_8,
                4.1e1,
                4.495_798,
                1.033_613_4,
                3.17e2,
                2.663_865_6,
                3.785e1,
                -1.222_8e2,
            ],
            [
                2.459_7,
                4.9e+01,
                4.728_033_5,
                1.020_920_5,
                6.07e+02,
                2.539_749,
                3.785e1,
                -1.222_8e2,
            ],
            [
                1.808,
                5.2e1,
                4.780_856_6,
                1.060_453_4,
                1.102e3,
                2.775_818_6,
                3.785e+01,
                -1.222_8e2,
            ],
            [
                1.642_4,
                5.0e1,
                4.401_691_4,
                1.040_169_1,
                1.131e3,
                2.391_120_4,
                3.784e1,
                -1.222_8e2,
            ],
            [
                1.687_5,
                5.2e1,
                4.703_225_6,
                1.032_258,
                3.95e2,
                2.548_387,
                3.784e1,
                -1.222_8e2,
            ],
            [
                1.927_4,
                4.9e1,
                5.068_783_3,
                1.182_539_7,
                8.63e2,
                2.283_069,
                3.784e1,
                -1.222_8e2,
            ],
            [
                1.961_5,
                5.2e1,
                4.882_086_3,
                1.090_702_9,
                1.168e3,
                2.648_526_2,
                3.784e1,
                -1.222_8e2,
            ],
            [
                1.796_9,
                4.8e1,
                5.737_313_3,
                1.220_895_5,
                1.026e3,
                3.062_686_7,
                3.784e1,
                -1.222_7e2,
            ],
            [
                1.375,
                4.9e1,
                5.030_395,
                1.112_462,
                7.54e2,
                2.291_793_3,
                3.783e1,
                -1.222_7e2,
            ],
            [
                2.730_3,
                5.1e1,
                4.972_015,
                1.070_895_6,
                1.258e3,
                2.347_015,
                3.783e1,
                -1.222_7e2,
            ],
            [
                1.486_1,
                4.9e1,
                4.602_272_5,
                1.068_181_9,
                5.7e2,
                2.159_091,
                3.783e1,
                -1.222_7e2,
            ],
            [
                1.097_2,
                4.8e1,
                4.807_486_5,
                1.155_080_2,
                9.87e2,
                2.639_037_4,
                3.783e1,
                -1.222_7e2,
            ],
            [
                1.410_3,
                5.2e1,
                3.749_379_6,
                9.677_419e-1,
                9.01e2,
                2.235_732,
                3.783e1,
                -1.222_8e2,
            ],
            [
                3.48,
                5.2e1,
                4.757_282,
                1.067_961_2,
                6.89e2,
                2.229_773_5,
                3.783e1,
                -1.222_6e2,
            ],
            [
                2.589_8,
                5.2e1,
                3.494_253,
                1.027_298_8,
                1.377e3,
                1.978_448_3,
                3.783e1,
                -1.222_6e2,
            ],
            [
                2.097_8,
                5.2e1,
                4.215_19,
                1.060_759_5,
                9.46e2,
                2.394_936_8,
                3.783e1,
                -1.222_6e2,
            ],
            [
                1.285_2,
                5.1e1,
                3.759_036,
                1.248_996,
                5.17e2,
                2.076_305_2,
                3.783e1,
                -1.222_6e2,
            ],
            [
                1.025,
                4.9e1,
                3.772_486_7,
                1.068_783,
                4.62e2,
                2.444_444_4,
                3.784e1,
                -1.222_6e2,
            ],
            [
                3.964_3,
                5.2e1,
                4.797_98,
                1.020_202,
                4.67e2,
                2.358_585_8,
                3.784e1,
                -1.222_6e2,
            ],
            [
                3.012_5,
                5.2e1,
                4.941_781,
                1.065_068_5,
                6.6e2,
                2.260_274,
                3.783e1,
                -1.222_6e2,
            ],
            [
                2.676_8,
                5.2e1,
                4.335_078_7,
                1.099_476_5,
                7.18e2,
                1.879_581_1,
                3.783e1,
                -1.222_6e2,
            ],
            [
                2.026,
                5.0e+01,
                3.700_657_8,
                1.059_210_5,
                6.16e2,
                2.026_315_7,
                3.783e1,
                -1.222_6e2,
            ],
            [
                1.734_8,
                4.3e1,
                3.980_237_2,
                1.233_201_6,
                5.58e2,
                2.205_533_5,
                3.782e1,
                -1.222_7e2,
            ],
            [
                9.506e-1, 4.0e1, 3.9, 1.218_75, 4.23e2, 2.643_75, 3.782e1, -1.222_6e2,
            ],
            [
                1.775,
                4.0e1,
                2.687_5,
                1.065_340_9,
                7.0e2,
                1.988_636_4,
                3.782e1,
                -1.222_7e2,
            ],
            [
                9.218e-1,
                2.1e1,
                2.045_662_2,
                1.034_246_6,
                7.35e2,
                1.678_082_2,
                3.782e1,
                -1.222_7e2,
            ],
            [
                1.504_5,
                4.3e1,
                4.589_680_7,
                1.120_393_2,
                1.061e3,
                2.606_879_7,
                3.782e1,
                -1.222_7e2,
            ],
            [
                1.110_8,
                4.1e1,
                4.473_611,
                1.184_722_2,
                1.959e3,
                2.720_833_3,
                3.782e1,
                -1.222_7e2,
            ],
            [
                1.247_5,
                5.2e+1,
                4.075,
                1.14,
                1.162e+3,
                2.905,
                3.782e+1,
                -1.222_7e+2,
            ],
            [
                1.609_8,
                5.2e1,
                5.021_459,
                1.008_583_7,
                7.01e2,
                3.008_583_8,
                3.782e1,
                -1.222_8e2,
            ],
            [
                1.411_3,
                5.2e1,
                4.295_454_5,
                1.104_545_5,
                5.76e2,
                2.618_181_7,
                3.782e1,
                -1.222_8e2,
            ],
            [
                1.505_7,
                5.2e1,
                4.779_923,
                1.111_969_1,
                6.22e2,
                2.401_544_3,
                3.782e1,
                -1.222_8e2,
            ],
            [
                8.172e-1,
                5.2e1,
                6.102_459,
                1.372_950_8,
                7.28e+02,
                2.983_606_6,
                3.782e+01,
                -1.222_8e+02,
            ],
            [
                1.217_1e+00,
                5.2e+01,
                4.562_5e+00,
                1.121_710_5,
                1.074e+03,
                3.532_894_8,
                3.782e+01,
                -1.222_8e+02,
            ],
            [
                2.562_5e+00,
                2.0,
                2.771_929_7,
                7.543_859_5e-1,
                9.4e+01,
                1.649_122_8,
                3.782e+01,
                -1.222_9e+02,
            ],
            [
                3.392_9e+00,
                5.2e+01,
                5.994_652_3,
                1.128_342_3,
                5.54e+02,
                2.962_566_9,
                3.783e+01,
                -1.222_9e+02,
            ],
            [
                6.118_3e+00,
                4.9e+01,
                5.869_565,
                1.260_869_6,
                8.6e+01,
                3.739_130_5,
                3.782e+01,
                -1.222_9e+02,
            ],
            [
                9.011e-01,
                5.0e+01,
                6.229_508_4,
                1.557_377_1,
                3.77e+02,
                3.090_164,
                3.781e+01,
                -1.222_9e+02,
            ],
            [
                1.191e+00,
                5.2e+01,
                7.698_113_4,
                1.490_566,
                5.21e+02,
                3.276_729_6,
                3.781e+01,
                -1.223e+02,
            ],
            [
                2.593_8,
                4.8e+01,
                6.225_564,
                1.368_421_1,
                3.92e+02,
                2.947_368_4,
                3.781e+01,
                -1.223e+02,
            ],
            [
                1.166_7e+00,
                5.2e+01,
                5.401_069_6,
                1.117_647,
                6.04e+02,
                3.229_946_6,
                3.781e+01,
                -1.223e+02,
            ],
            [
                8.056e-01,
                4.8e+01,
                4.382_53,
                1.066_265_1,
                7.88e+02,
                2.373_494,
                3.781e+01,
                -1.223e+02,
            ],
            [
                2.609_4e+00,
                5.2e+01,
                6.986_394_4,
                1.659_864,
                4.92e+02,
                3.346_938_8,
                3.78e+01,
                -1.222_9e+02,
            ],
            [
                1.851_6e+00,
                5.2e+01,
                6.975_61,
                1.329_268_3,
                2.74e+02,
                3.341_463_3,
                3.781e+01,
                -1.223e+02,
            ],
            [
                9.802e-01,
                4.6e+01,
                4.584_288,
                1.054_009_8,
                1.823e+03,
                2.983_633_3,
                3.781e+01,
                -1.222_9e+02,
            ],
            [
                1.771_9,
                2.6e+01,
                6.047_244,
                1.196_850_4,
                3.92e+02,
                3.086_614_1,
                3.781e+01,
                -1.222_9e+02,
            ],
            [
                7.286e-1,
                4.6e+1,
                3.375_451_3,
                1.072_202_2,
                5.82e+02,
                2.101_083,
                3.781e+01,
                -1.222_9e+02,
            ],
            [
                1.75e+00,
                4.9e+01,
                5.552_631_4,
                1.342_105_3,
                5.6e+02,
                3.684_210_5,
                3.781_e+01,
                -1.222_9e+02,
            ],
            [
                4.999e-1,
                4.6e+1,
                1.714_285_7,
                5.714_286e-1,
                1.8e+01,
                2.571_428_5,
                3.781e+01,
                -1.222_9e+02,
            ],
            [
                2.483e+00,
                2.0e+01,
                6.278_195_4,
                1.210_526_3,
                2.9e+02,
                2.180_451_2,
                3.781e+01,
                -1.222_9e+02,
            ],
            [
                9.241e-01,
                1.7e+01,
                2.817_767_6,
                1.052_391_8,
                7.62e+02,
                1.735_763_1,
                3.781e+01,
                -1.222_8e+02,
            ],
            [
                2.446_4e+00,
                3.6e+01,
                5.724_951,
                1.104_125_7,
                1.236e+03,
                2.428_290_8,
                3.781e+01,
                -1.222_8e+02,
            ],
            [
                1.111_1e+00,
                1.9e+01,
                5.830_918,
                1.173_913,
                7.21e+02,
                3.483_091_8,
                3.781e+01,
                -1.222_8e+02,
            ],
            [
                8.026e-01,
                2.3e+01,
                5.369_230_7,
                1.150_769_2,
                1.054e+03,
                3.243_076_8,
                3.781e+01,
                -1.222_9e+02,
            ],
            [
                2.011_4e+00,
                3.8e+01,
                4.412_903_3,
                1.135_483_9,
                3.44e+02,
                2.219_354_9,
                3.78e+01,
                -1.222_8e+02,
            ],
            [
                1.5e+00,
                1.7e+01,
                3.197_231_8,
                1.0e+00,
                6.09e+02,
                2.107_266_4,
                3.781e+01,
                -1.222_8e+02,
            ],
            [
                1.166_7,
                5.2e+01,
                3.75e+00,
                1.0e+00,
                1.83e+02,
                3.267_857,
                3.781e+01,
                -1.222_7e+02,
            ],
            [
                1.520_8,
                5.2e+01,
                3.908_046,
                1.114_942_6,
                2.0e+02,
                2.298_850_5,
                3.781e+01,
                -1.222_8e+02,
            ],
            [
                8.075e-1,
                5.2e+01,
                2.490_322_6,
                1.058_064_5,
                3.46e+02,
                2.232_258,
                3.781e+01,
                -1.222_8e+02,
            ],
            [
                1.808_8e+00,
                3.5e+01,
                5.609_467_5,
                1.088_757_4,
                4.67e+02,
                2.763_313_5,
                3.781e+01,
                -1.222_8e+02,
            ],
            [
                2.408_3e+00,
                5.2e+01,
                6.721_739_3,
                1.243_478_3,
                3.77e+02,
                3.278_261,
                3.781e+01,
                -1.222_8e+02,
            ],
            [
                9.77e-01,
                4.0e+01,
                2.315_789_5,
                1.186_842_1,
                5.82e+02,
                1.531_578_9,
                3.781e+01,
                -1.222_7e+02,
            ],
            [
                7.6e-01,
                1.0e+01,
                2.651_515_2,
                1.054_545_4,
                5.46e+02,
                1.654_545_4,
                3.781e+01,
                -1.222_7e+02,
            ],
            [
                9.722e-01,
                1.0e+01,
                2.692_307_7,
                1.076_923_1,
                1.25e+02,
                3.205_128_2,
                3.78e+01,
                -1.222_7e+02,
            ],
            [
                1.243_4,
                5.2e+01,
                2.929_411_6,
                9.176_470_6e-1,
                3.96e+02,
                4.658_823_5,
                3.78e+01,
                -1.222_7e+02,
            ],
            [
                2.093_8,
                1.6e+01,
                2.745_856_3,
                1.082_873,
                8.0e+02,
                2.209_944_7,
                3.78e+01,
                -1.222_7e+02,
            ],
            [
                8.668e-1,
                5.2e+01,
                2.443_181_8,
                9.886_364e-1,
                9.04e+02,
                1.027_272_7e1,
                3.78e+01,
                -1.222_8e2,
            ],
            [
                7.5e-01,
                5.2e+01,
                2.823_529_5,
                9.117_647e-1,
                1.91e+02,
                5.617_647,
                3.78e+01,
                -1.222_8e+02,
            ],
            [
                2.635_4,
                2.7e+01,
                3.493_377_4,
                1.149_006_6,
                7.18e+02,
                2.377_483_4,
                3.779e+01,
                -1.222_7e+2,
            ],
            [
                1.847_7,
                3.9e+01,
                3.672_376_9,
                1.334_047_1,
                1.327e+03,
                2.841_541_8,
                3.78e+01,
                -1.222_7e+2,
            ],
            [
                2.009_6,
                3.6e+01,
                2.294_016_4,
                1.066_293_6,
                3.469e+03,
                1.493_327_6,
                3.78e+01,
                -1.222_6e+02,
            ],
            [
                2.834_5,
                3.1e+01,
                3.894_915_3,
                1.127_966,
                2.048e+03,
                1.735_593_2,
                3.782e+01,
                -1.222_6e+02,
            ],
            [
                2.006_2,
                2.9e+01,
                3.681_318_8,
                1.175_824_2,
                2.02e+02,
                2.219_780_2,
                3.781e+01,
                -1.222_6e+2,
            ],
            [
                1.218_5,
                2.2e+1,
                2.945_6,
                1.016,
                2.024e+3,
                1.619_2,
                3.782e+1,
                -1.222_6e+2,
            ],
            [
                2.610_4,
                3.7e+1,
                3.707_142_8,
                1.107_142_8,
                1.838e+3,
                1.875_510_2,
                3.782e+1,
                -1.222_6e+2,
            ],
        ]);

        let target_vec = [
            4.526, 3.585, 3.521, 3.413, 3.422, 2.697, 2.992, 2.414, 2.267, 2.611, 2.815, 2.418,
            2.135, 1.913, 1.592, 1.4, 1.525, 1.555, 1.587, 1.629, 1.475, 1.598, 1.139, 0.997,
            1.326, 1.075, 0.938, 1.055, 1.089, 1.32, 1.223, 1.152, 1.104, 1.049, 1.097, 0.972,
            1.045, 1.039, 1.914, 1.76, 1.554, 1.5, 1.188, 1.888, 1.844, 1.823, 1.425, 1.375, 1.875,
            1.125, 1.719, 0.938, 0.975, 1.042, 0.875, 0.831, 0.875, 0.853, 0.803, 0.6, 0.757, 0.75,
            0.861, 0.761, 0.735, 0.784, 0.844, 0.813, 0.85, 1.292, 0.825, 0.952, 0.75, 0.675,
            1.375, 1.775, 1.021, 1.083, 1.125, 1.313, 1.625, 1.125, 1.125, 1.375, 1.188, 0.982,
            1.188, 1.625, 1.375, 5.00001, 1.625, 1.375, 1.625, 1.875, 1.792, 1.3, 1.838, 1.25, 1.7,
            1.931,
        ];

        // define information needed for xgboost
        let strides_ax_0 = data_arr_2d.strides()[0] as usize;
        let strides_ax_1 = data_arr_2d.strides()[1] as usize;
        let byte_size_ax_0 = std::mem::size_of::<f32>() * strides_ax_0;
        let byte_size_ax_1 = std::mem::size_of::<f32>() * strides_ax_1;

        // get xgboost style matrices
        let mut xg_matrix = DMatrix::from_col_major_f32(
            data_arr_2d.as_slice_memory_order().unwrap(),
            byte_size_ax_0,
            byte_size_ax_1,
            100,
            9,
            -1,
            f32::NAN,
        )
        .unwrap();

        // set labels
        // TODO: make more generic

        let lbls: Vec<f32> = target_vec.iter().map(|elem| *elem as f32).collect();
        xg_matrix.set_labels(lbls.as_slice()).unwrap();

        // ------------------------------------------------------
        // start training

        let mut initial_training_config: HashMap<&str, &str> = HashMap::new();

        initial_training_config.insert("validate_parameters", "1");
        initial_training_config.insert("process_type", "default");
        initial_training_config.insert("tree_method", "hist");
        initial_training_config.insert("eval_metric", "rmse");
        initial_training_config.insert("max_depth", "3");

        let evals = &[(&xg_matrix, "train")];
        let bst = Booster::train(
            Some(evals),
            &xg_matrix,
            initial_training_config,
            None, // <- No old model yet
        )
        .unwrap();

        let test_data_arr_2d = arr2(&[
            [
                1.91,
                4.6e+01,
                5.0,
                1.004_132_3,
                5.23e+02,
                2.161_157_1,
                3.936e+01,
                -1.217e+02,
                6.39e-01,
            ],
            [
                2.047_4,
                3.7e+01,
                4.957_446_6,
                1.053_191_5,
                1.505e+03,
                3.202_127_7,
                3.936e+01,
                -1.217e+2,
                5.6e-01,
            ],
            [
                1.835_5,
                3.4e+01,
                5.103_03,
                1.127_272_7,
                6.35e+02,
                3.848_484_8,
                3.936e+01,
                -1.216_9e+2,
                6.3e-01,
            ],
            [
                2.324_3,
                2.7e+01,
                6.347_188_5,
                1.063_569_7,
                1.1e+03,
                2.689_486_5,
                3.938e+01,
                -1.217_4e+2,
                8.55e-01,
            ],
            [
                2.525_9,
                3.0e+01,
                5.508_108,
                1.037_837_9,
                5.01e+2,
                2.708_108_2,
                3.933e+1,
                -1.218e+2,
                8.13e-1,
            ],
            [
                2.281_3,
                2.1e+01,
                5.207_272_5,
                1.032_727_2,
                8.62e+02,
                3.134_545_6,
                3.942e+01,
                -1.217_1e+2,
                5.76e-01,
            ],
            [
                2.172_8,
                2.2e+01,
                5.616_099,
                1.058_823_6,
                9.41e+02,
                2.913_312_7,
                3.941e+01,
                -1.217_1e+2,
                5.94e-01,
            ],
            [
                2.494_3,
                2.9e+01,
                5.050_898,
                9.790_419_3e-1,
                8.64e+02,
                2.586_826_3,
                3.94e+01,
                -1.217_5e+2,
                8.19e-01,
            ],
            [
                3.392_9,
                3.9e+01,
                6.656_626_7,
                1.084_337_4,
                4.08e+02,
                2.457_831_4,
                3.948e+01,
                -1.217_9e+2,
                8.21e-01,
            ],
            [
                2.381_6,
                1.6e+01,
                6.055_954,
                1.120_516_5,
                1.516e+03,
                2.175_036,
                3.815e+01,
                -1.204_6e+2,
                1.16,
            ],
            [
                2.5,
                1.0e+01,
                5.381_443_3,
                1.116_838_5,
                7.85e+02,
                2.697_594_5,
                3.812e+01,
                -1.205_5e+2,
                1.161,
            ],
            [
                2.365_4,
                3.4e+01,
                5.590_631_5,
                1.138_492_8,
                1.15e+03,
                2.342_158_8,
                3.809e+01,
                -1.205_6e+2,
                9.49e-01,
            ],
            [
                2.906_3,
                2.7e+01,
                6.025_125_5,
                1.125_628_1,
                4.63e+02,
                2.326_633_2,
                3.807e+01,
                -1.205_5e+2,
                9.22e-1,
            ],
            [
                2.287_5,
                3.7e+01,
                5.257_143,
                1.057_142_9,
                3.39e+02,
                2.421_428_7,
                3.807e+01,
                -1.205_4e+2,
                7.99e-01,
            ],
            [
                2.652_8,
                9.0,
                8.010_753,
                1.586_021_5,
                2.233e+03,
                2.401_075_4,
                3.797e+01,
                -1.206_7e+2,
                1.33,
            ],
            [
                3.0,
                1.6e+01,
                6.110_569,
                1.162_601_6,
                1.777e+03,
                2.889_431,
                3.809e+01,
                -1.204_6e+2,
                1.226,
            ],
            [
                2.982_1,
                1.9e+01,
                5.278_947_4,
                1.236_842_2,
                5.38e+02,
                2.831_579,
                3.824e+01,
                -1.207_9e+2,
                9.04e-01,
            ],
            [
                2.047_2,
                1.6e+01,
                5.931_559,
                1.218_631_1,
                1.319e+3,
                2.507_604,
                3.82e+1,
                -1.209e+2,
                9.32e-1,
            ],
            [
                4.010_9,
                8.0,
                5.574_176,
                1.063_186_8,
                1.0e+03,
                2.747_252_7,
                3.816e+1,
                -1.208_8e+2,
                1.259,
            ],
            [
                3.636,
                9.0,
                5.994_983,
                1.137_123_7,
                1.8e+03,
                3.010_033_4,
                3.811e+1,
                -1.209_1e+2,
                1.331,
            ],
        ]);

        let strides_ax_0 = test_data_arr_2d.strides()[0] as usize;
        let strides_ax_1 = test_data_arr_2d.strides()[1] as usize;
        let byte_size_ax_0 = std::mem::size_of::<f32>() * strides_ax_0;
        let byte_size_ax_1 = std::mem::size_of::<f32>() * strides_ax_1;

        // get xgboost style matrices
        let test_data = DMatrix::from_col_major_f32(
            test_data_arr_2d.as_slice_memory_order().unwrap(),
            byte_size_ax_0,
            byte_size_ax_1,
            20,
            9,
            -1,
            f32::NAN,
        )
        .unwrap();

        let mut out_dim: u64 = 10;
        bst.predict_from_dmat(&test_data, &[20, 9], &mut out_dim)
            .unwrap();
    }
}
