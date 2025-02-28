use std::{ffi::CStr, fs::File};

use kaldi_fbank_rust::{FbankOptions, FrameExtractionOptions, MelBanksOptions, OnlineFbank};
use ndarray::{s, Array1, Array2, ArrayView1, Axis};

#[derive(Debug, Clone, Copy, Default)]
pub enum WindowType {
    #[default]
    Hanning,
    Sine,
    Hamming,
    Povey,
    Rectangular,
    Blackman,
}

#[derive(Debug)]
pub struct WavFrontendConfig {
    pub sample_rate: i32,
    pub frame_length_ms: f32,
    pub frame_shift_ms: f32,
    pub n_mels: usize,
    pub lfr_m: usize,
    pub lfr_n: usize,
    pub cmvn_file: Option<String>,
    pub window_type: WindowType,
}

impl Default for WavFrontendConfig {
    fn default() -> Self {
        WavFrontendConfig {
            // Todo: 其實我不知道正確的config是甚麼，這裡是根據我專案寫的
            sample_rate: 16000,
            frame_length_ms: 25.0,
            frame_shift_ms: 10.0,
            n_mels: 80,
            lfr_m: 7,
            lfr_n: 6,
            cmvn_file: None,
            window_type: WindowType::default(),
        }
    }
}

#[derive(Debug)]
pub struct WavFrontend {
    config: WavFrontendConfig,
    cmvn_means: Option<Array1<f32>>,
    cmvn_vars: Option<Array1<f32>>,
}

impl WavFrontend {
    pub fn new(config: WavFrontendConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let (cmvn_means, cmvn_vars) = if let Some(cmvn_path) = &config.cmvn_file {
            let (means, vars) = Self::load_cmvn(cmvn_path)?;
            (Some(means), Some(vars))
        } else {
            (None, None)
        };
        Ok(WavFrontend {
            config,
            cmvn_means,
            cmvn_vars,
        })
    }

    fn load_cmvn(path: &str) -> Result<(Array1<f32>, Array1<f32>), Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut lines = std::io::BufRead::lines(reader);

        let mut means = Vec::new();
        let mut vars = Vec::new();
        let mut is_means = false;
        let mut is_vars = false;

        while let Some(line) = lines.next() {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts[0] == "<AddShift>" {
                is_means = true;
                continue;
            } else if parts[0] == "<Rescale>" {
                is_vars = true;
                continue;
            } else if parts[0] == "<LearnRateCoef>" && is_means {
                means = parts[3..parts.len() - 1]
                    .iter()
                    .map(|x| x.parse::<f32>().unwrap())
                    .collect();
                is_means = false;
            } else if parts[0] == "<LearnRateCoef>" && is_vars {
                vars = parts[3..parts.len() - 1]
                    .iter()
                    .map(|x| x.parse::<f32>().unwrap())
                    .collect();
                is_vars = false;
            }
        }

        Ok((Array1::from_vec(means), Array1::from_vec(vars)))
    }

    fn compute_fbank_features(
        &self,
        waveform: &[f32],
    ) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        // 創建 FbankOptions，與 Python 版本一致
        let opt = FbankOptions {
            frame_opts: FrameExtractionOptions {
                samp_freq: self.config.sample_rate as f32, // 16000 Hz
                window_type: CStr::from_bytes_with_nul(b"hamming\0").unwrap().as_ptr(), // "hamming"
                dither: 0.0,                               // 無抖動
                frame_shift_ms: self.config.frame_shift_ms, // 10.0 ms
                frame_length_ms: self.config.frame_length_ms, // 25.0 ms
                snip_edges: true,
                ..Default::default()
            },
            mel_opts: MelBanksOptions {
                num_bins: self.config.n_mels as i32, // 80
                ..Default::default()
            },
            energy_floor: 0.0,
            ..Default::default()
        };

        // 創建 OnlineFbank 實例
        let mut fbank = OnlineFbank::new(opt);

        // 傳入波形數據
        fbank.accept_waveform(self.config.sample_rate as f32, waveform);

        // 獲取特徵幀數
        let frames = fbank.num_ready_frames();

        // 收集所有特徵幀
        let mut fbank_feats = Vec::with_capacity(frames as usize);
        for i in 0..frames {
            let frame = fbank.get_frame(i).expect("Should have frame"); // 獲取第 i 幀特徵
            fbank_feats.push(frame.to_vec());
        }

        // 轉換為 Array2<f32>，形狀為 (frames, n_mels)
        let fbank_array = Array2::from_shape_vec(
            (frames as usize, self.config.n_mels),
            fbank_feats.into_iter().flatten().collect(),
        )?;

        Ok(fbank_array)
    }

    fn apply_lfr(&self, fbank: &Array2<f32>, lfr_m: usize, lfr_n: usize) -> Array2<f32> {
        let t = fbank.shape()[0];
        let t_lfr = ((t as f32) / lfr_n as f32).ceil() as usize;
        let left_padding_rows = (lfr_m - 1) / 2;
        let mut padded_fbank = Array2::zeros((t + left_padding_rows, fbank.shape()[1]));

        for i in 0..left_padding_rows {
            padded_fbank
                .slice_mut(s![i, ..])
                .assign(&fbank.slice(s![0, ..]));
        }
        for i in 0..t {
            padded_fbank
                .slice_mut(s![i + left_padding_rows, ..])
                .assign(&fbank.slice(s![i, ..]));
        }

        let feat_dim = self.config.n_mels * lfr_m;
        let mut lfr_array = Array2::zeros((t_lfr, feat_dim));
        for i in 0..t_lfr {
            let start = i * lfr_n;
            let end = if lfr_m <= t + left_padding_rows - start {
                start + lfr_m
            } else {
                t + left_padding_rows
            };
            let frame = padded_fbank.slice(s![start..end, ..]);
            let mut target = lfr_array.slice_mut(s![i, ..frame.len()]);
            target.assign(&frame.into_shape(frame.len()).unwrap());
            if end < start + lfr_m {
                let last_row = padded_fbank.slice(s![-1, ..]);
                for j in end - start..lfr_m {
                    lfr_array
                        .slice_mut(s![i, j * self.config.n_mels..(j + 1) * self.config.n_mels])
                        .assign(&last_row);
                }
            }
        }
        lfr_array
    }

    fn apply_cmvn(&self, feats: &Array2<f32>) -> Array2<f32> {
        if let (Some(means), Some(vars)) = (&self.cmvn_means, &self.cmvn_vars) {
            let (frames, dim) = feats.dim();
            let means_expanded = means.broadcast((frames, dim)).unwrap();
            let vars_expanded = vars.broadcast((frames, dim)).unwrap();
            (feats + &means_expanded) * &vars_expanded
        } else {
            feats.to_owned()
        }
    }

    pub fn extract_features(
        &self,
        waveform: &[i16],
    ) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        let waveform: Vec<f32> = waveform.iter().map(|&s| s as f32).collect();
        let fbank = self.compute_fbank_features(&waveform)?;
        let lfr_feats = self.apply_lfr(&fbank, self.config.lfr_m, self.config.lfr_n);
        let feats = self.apply_cmvn(&lfr_feats);
        Ok(feats)
    }
}
