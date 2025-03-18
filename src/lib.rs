pub mod fsmn_vad;
pub mod wavfrontend;

use core::fmt;
use std::io::Read;
use std::sync::Mutex;
use std::{fs::File, io::BufReader};

use fsmn_vad::{FSMNVad, VADXOptions};
use hf_hub::api::sync::Api;
use hound::WavReader;
use ndarray::parallel::prelude::*;
use ndarray::{s, Array2, Array3, ArrayView3, Axis};
use ndarray_npy::ReadNpyExt;
use rayon::iter::IntoParallelIterator;
use regex::Regex;
use rknn_rs::prelude::{Rknn, RknnInput, RknnTensorFormat, RknnTensorType};
use sentencepiece::SentencePieceProcessor;
use wavfrontend::{WavFrontend, WavFrontendConfig};

/// Represents supported languages for speech recognition.
///
/// This enum defines the languages supported by the `SenseVoiceSmall` model.
#[derive(Debug, Copy, Clone)]
pub enum SenseVoiceLanguage {
    /// English
    En,
    /// Chinese (Mandarin)
    Zh,
    /// Cantonese
    Yue,
    /// Japanese
    Ja,
    /// Korean
    Ko,
}

/// Implementation of methods for `SenseVoiceLanguage`.
impl SenseVoiceLanguage {
    /// Converts a string to a `SenseVoiceLanguage` variant.
    ///
    /// This method parses a string (case-insensitive) and returns the corresponding language variant.
    ///
    /// # Arguments
    ///
    /// * `s` - The string to parse (e.g., "en", "ZH").
    ///
    /// # Returns
    ///
    /// An `Option<SenseVoiceLanguage>` where `None` indicates an unrecognized language string.
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "en" => Some(SenseVoiceLanguage::En),
            "zh" => Some(SenseVoiceLanguage::Zh),
            "yue" => Some(SenseVoiceLanguage::Yue),
            "ja" => Some(SenseVoiceLanguage::Ja),
            "ko" => Some(SenseVoiceLanguage::Ko),
            _ => None,
        }
    }
}

/// Represents possible emotions detected in speech.
///
/// This enum defines the emotional states that can be identified in audio segments.
#[derive(Debug, Copy, Clone)]
pub enum SenseVoiceEmo {
    /// Happy emotion
    Happy,
    /// Sad emotion
    Sad,
    /// Angry emotion
    Angry,
    /// Neutral emotion
    Neutral,
    /// Fearful emotion
    Fearful,
    /// Disgusted emotion
    Disgusted,
    /// Surprised emotion
    Surprised,
}

/// Implementation of methods for `SenseVoiceEmo`.
impl SenseVoiceEmo {
    /// Converts a string to a `SenseVoiceEmo` variant.
    ///
    /// This method parses a string (case-insensitive) and returns the corresponding emotion variant.
    ///
    /// # Arguments
    ///
    /// * `s` - The string to parse (e.g., "HAPPY", "sad").
    ///
    /// # Returns
    ///
    /// An `Option<SenseVoiceEmo>` where `None` indicates an unrecognized emotion string.
    fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "HAPPY" => Some(SenseVoiceEmo::Happy),
            "SAD" => Some(SenseVoiceEmo::Sad),
            "ANGRY" => Some(SenseVoiceEmo::Angry),
            "NEUTRAL" => Some(SenseVoiceEmo::Neutral),
            "FEARFUL" => Some(SenseVoiceEmo::Fearful),
            "DISGUSTED" => Some(SenseVoiceEmo::Disgusted),
            "SURPRISED" => Some(SenseVoiceEmo::Surprised),
            _ => None,
        }
    }
}

/// Represents types of audio events detected in speech.
///
/// This enum defines the categories of events that can occur within audio segments.
#[derive(Debug, Copy, Clone)]
pub enum SenseVoiceEvent {
    /// Background music
    Bgm,
    /// Speech content
    Speech,
    /// Applause sound
    Applause,
    /// Laughter sound
    Laughter,
    /// Crying sound
    Cry,
    /// Sneezing sound
    Sneeze,
    /// Breathing sound
    Breath,
    /// Coughing sound
    Cough,
}

/// Implementation of methods for `SenseVoiceEvent`.
impl SenseVoiceEvent {
    /// Converts a string to a `SenseVoiceEvent` variant.
    ///
    /// This method parses a string (case-insensitive) and returns the corresponding event variant.
    ///
    /// # Arguments
    ///
    /// * `s` - The string to parse (e.g., "BGM", "laughter").
    ///
    /// # Returns
    ///
    /// An `Option<SenseVoiceEvent>` where `None` indicates an unrecognized event string.
    fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "BGM" => Some(SenseVoiceEvent::Bgm),
            "SPEECH" => Some(SenseVoiceEvent::Speech),
            "APPLAUSE" => Some(SenseVoiceEvent::Applause),
            "LAUGHTER" => Some(SenseVoiceEvent::Laughter),
            "CRY" => Some(SenseVoiceEvent::Cry),
            "SNEEZE" => Some(SenseVoiceEvent::Sneeze),
            "BREATH" => Some(SenseVoiceEvent::Breath),
            "COUGH" => Some(SenseVoiceEvent::Cough),
            _ => None,
        }
    }
}

/// Represents options for punctuation normalization in transcribed text.
///
/// This enum defines whether punctuation is included or excluded in the output text.
#[derive(Debug, Copy, Clone)]
pub enum SenseVoicePunctuationNormalization {
    /// Include punctuation in the text
    With,
    /// Exclude punctuation from the text
    Woitn,
}

/// Implementation of methods for `SenseVoicePunctuationNormalization`.
impl SenseVoicePunctuationNormalization {
    /// Converts a string to a `SenseVoicePunctuationNormalization` variant.
    ///
    /// This method parses a string (case-insensitive) and returns the corresponding normalization variant.
    ///
    /// # Arguments
    ///
    /// * `s` - The string to parse (e.g., "with", "WOITN").
    ///
    /// # Returns
    ///
    /// An `Option<SenseVoicePunctuationNormalization>` where `None` indicates an unrecognized normalization string.
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "with" => Some(SenseVoicePunctuationNormalization::With),
            "woitn" => Some(SenseVoicePunctuationNormalization::Woitn),
            _ => None,
        }
    }
}

/// Represents a segment of audio with its transcribed text and associated metadata.
///
/// This structure holds the transcription result of an audio segment, including timing, language, emotion, event, and normalization details.
#[derive(Debug)]
pub struct VoiceText {
    /// The language of the transcribed text.
    pub language: SenseVoiceLanguage,
    /// The detected emotion in the audio segment.
    pub emotion: SenseVoiceEmo,
    /// The type of audio event in the segment.
    pub event: SenseVoiceEvent,
    /// Indicates whether punctuation is included in the transcribed text.
    pub punctuation_normalization: SenseVoicePunctuationNormalization,
    /// The transcribed text of the audio segment.
    pub content: String,
}

/// Parses a string line into a `VoiceText` instance based on a specific format.
///
/// The expected format is: `<|language|><|emotion|><|event|><|punctuation|><content>`
///
/// # Arguments
///
/// * `line` - The string to parse (e.g., "<|zh|><|HAPPY|><|BGM|><|woitn|>Hello").
/// * `start_ms` - Start time of the segment in milliseconds.
/// * `end_ms` - End time of the segment in milliseconds.
///
/// # Returns
///
/// An `Option<VoiceText>` where `None` indicates parsing failure due to invalid format or unrecognized tags.
fn parse_line(line: &str) -> Option<VoiceText> {
    let re = Regex::new(r"^<\|(.*?)\|><\|(.*?)\|><\|(.*?)\|><\|(.*?)\|>(.*)$").unwrap();
    if let Some(caps) = re.captures(line) {
        let lang_str = &caps[1];
        let emo_str = &caps[2];
        let event_str = &caps[3];
        let punct_str = &caps[4];
        let content = &caps[5];

        let language = SenseVoiceLanguage::from_str(lang_str)?;
        let emotion = SenseVoiceEmo::from_str(emo_str)?;
        let event = SenseVoiceEvent::from_str(event_str)?;
        let punctuation_normalization = SenseVoicePunctuationNormalization::from_str(punct_str)?;

        Some(VoiceText {
            language,
            emotion,
            event,
            punctuation_normalization,
            content: content.to_string(),
        })
    } else {
        None
    }
}

/// Represents an error specific to the `SenseVoiceSmall` system.
///
/// This structure encapsulates error messages related to initialization, inference, or resource management.
#[derive(Debug)]
struct SenseVoiceSmallError {
    /// The error message describing the issue.
    message: String,
}

/// Implements `Display` trait for `SenseVoiceSmallError` to format error messages.
impl fmt::Display for SenseVoiceSmallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SenseVoiceSmallError: {}", self.message)
    }
}

/// Implements `Error` trait for `SenseVoiceSmallError` to integrate with Rust's error handling system.
impl std::error::Error for SenseVoiceSmallError {}

/// Implementation of methods for `SenseVoiceSmallError`.
impl SenseVoiceSmallError {
    /// Creates a new `SenseVoiceSmallError` instance with the given message.
    ///
    /// # Arguments
    ///
    /// * `message` - The error message to encapsulate.
    ///
    /// # Returns
    ///
    /// A new `SenseVoiceSmallError` instance.
    pub fn new(message: &str) -> Self {
        SenseVoiceSmallError {
            message: message.to_owned(),
        }
    }
}

/// Represents the core structure for the SenseVoiceSmall speech recognition system.
///
/// This structure manages components such as voice activity detection (VAD), automatic speech recognition (ASR),
/// and RKNN model inference for processing audio data.
#[derive(Debug)]
pub struct SenseVoiceSmall {
    vad_frontend: WavFrontend,
    asr_frontend: WavFrontend,
    n_seq: usize,
    spp: SentencePieceProcessor,
    rknn: Rknn,
    fsmn: Mutex<FSMNVad>,
    embedding: Array2<f32>,
}

/// Implementation of methods for `SenseVoiceSmall`.
impl SenseVoiceSmall {
    /// Initializes a new `SenseVoiceSmall` instance by loading necessary models and configurations from Hugging Face Hub.
    ///
    /// This function downloads required models and configurations from the "happyme531/SenseVoiceSmall-RKNN2" repository
    /// on Hugging Face Hub, including VAD, embedding, RKNN, and sentencepiece models.
    ///
    /// # Errors
    ///
    /// Returns an error if any of the model files cannot be loaded or if there are issues initializing the components.
    ///
    /// # Example
    ///
    /// ```
    /// use sensevoice_rs::SenseVoiceSmall;
    ///
    /// let mut svs = SenseVoiceSmall::init().expect("Failed to initialize SenseVoiceSmall");
    /// ```
    pub fn init<P: AsRef<std::path::Path>>(
        model_path: P,
        vadconfig: VADXOptions,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // TODO: Maybe we should read a config file in the reop to read model.
        let api = Api::new().unwrap();
        let repo = api.model(model_path.as_ref().to_string_lossy().to_string());
        let fsmn_path = repo.get("fsmnvad-offline.onnx")?;
        let embedding_path = repo.get("embedding.npy")?;
        let rknn_path = repo.get("sense-voice-encoder.rknn")?;
        let sentence_path = repo.get("chn_jpn_yue_eng_ko_spectok.bpe.model")?;
        let fsmn_am_path = repo.get("fsmn-am.mvn")?;
        let am_path = repo.get("am.mvn")?;

        // TODO: Should read config
        let fsmn = Mutex::new(FSMNVad::new(fsmn_path, vadconfig).unwrap());

        // Load embedding.npy
        let embedding_file = File::open(embedding_path)?;
        let embedding_reader = BufReader::new(embedding_file);
        let embedding: Array2<f32> = Array2::read_npy(embedding_reader)?;
        assert_eq!(embedding.shape()[1], 560, "Embedding dimension must be 560");

        let rknn = Rknn::rknn_init(rknn_path)?;
        let spp = SentencePieceProcessor::open(sentence_path)?;

        //TODO: We should read from config
        let n_seq = 171; // 因為這個模型的向量已經被固定了，這是轉換成rknn的時候決定的，除非你將這個模型重新轉換成動態向量，否則這個數字不應該改變

        // 設定vad前端
        let vad_frontend = WavFrontend::new(WavFrontendConfig {
            lfr_m: 5, // 調整為 5，對應 400 維特徵（5 * 80）
            cmvn_file: Some(fsmn_am_path.to_str().unwrap().to_owned()),
            ..Default::default()
        })?;

        // 設定asr前端
        let asr_frontend = WavFrontend::new(WavFrontendConfig {
            lfr_m: 7, // 調整為 7，對應 560 維特徵（7 * 80）
            cmvn_file: Some(am_path.to_str().unwrap().to_owned()),
            ..Default::default()
        })?;

        Ok(SenseVoiceSmall {
            vad_frontend,
            asr_frontend,
            embedding,
            n_seq,
            spp,
            rknn,
            fsmn,
        })
    }

    /// Performs speech recognition on a vector of audio samples.
    ///
    /// This method processes raw audio samples, extracts features, detects speech segments using VAD,
    /// and transcribes each segment into text with associated metadata.
    ///
    /// # Arguments
    ///
    /// * `content` - A vector of 16-bit integer audio samples.
    /// * `sample_rate` - The sample rate of the audio (e.g., 8000 or 16000 Hz).
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of `VoiceText` instances, each representing a transcribed audio segment.
    ///
    /// # Errors
    ///
    /// Returns an error if feature extraction, VAD inference, or RKNN processing fails.
    pub fn infer_vec(
        &self,
        content: Vec<i16>,
        sample_rate: u32,
    ) -> Result<Vec<VoiceText>, Box<dyn std::error::Error>> {
        // 提取特徵
        let audio_feats = self.vad_frontend.extract_features(&content)?;

        // 進行 VAD 推理
        let segments = {
            let mut fsmn = self.fsmn.lock().unwrap();
            fsmn.infer_vad(audio_feats, &content, true)?
        };

        // 處理語音片段
        let mut ret = Vec::new();
        for (start_ms, end_ms) in segments {
            let start_sample = (start_ms as f32 / 1000.0 * sample_rate as f32) as usize;
            let end_sample = (end_ms as f32 / 1000.0 * sample_rate as f32) as usize;
            let segment = &content[start_sample..end_sample];
            let vt = self.recognition(segment)?;
            ret.push(vt);
        }
        Ok(ret)
    }

    pub fn recognition(&self, segment: &[i16]) -> Result<VoiceText, Box<dyn std::error::Error>> {
        // 提取特徵
        let audio_feats = self.asr_frontend.extract_features(segment)?;

        // 準備 RKNN 輸入
        self.prepare_rknn_input_advanced(&audio_feats, 0, false)?; // language=0 (auto), use_itn=false
        self.rknn.run()?;
        let mut asr_output = self.rknn.outputs_get_raw::<f32>()?;
        let asr_text = self.decode_asr_output(&asr_output.data)?;
        self.rknn.outputs_release(&mut asr_output)?; // 資料會被丟棄，不可再用asr_output
        match parse_line(&asr_text) {  // 處理輸出並解碼為文字
            Some(vt) => Ok(vt),
            None => Err(format!("Parse line failed, text is:{}, If u still get empty text, please check your vad config. This model only can infer 9 secs voice.",asr_text).into() ),
        }
    }

    /// Performs speech recognition on an audio file.
    ///
    /// This method reads a WAV file, validates its sample rate and format, and processes it to extract transcribed segments.
    ///
    /// # Arguments
    ///
    /// * `wav_path` - Path to the WAV file to process.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of `VoiceText` instances, each representing a transcribed audio segment.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The WAV file cannot be opened or read.
    /// - The sample rate is not 8000 or 16000 Hz.
    /// - The sample format is not 16-bit integer.
    /// - The audio content is empty or inference fails.
    ///
    /// # Example
    ///
    /// ```
    /// use sensevoice_rs::SenseVoiceSmall;
    /// use std::path::Path;
    ///
    /// let mut svs = SenseVoiceSmall::init().expect("Failed to initialize");
    /// let segments = svs.infer_file(Path::new("path/to/audio.wav"))
    ///     .expect("Failed to infer audio file");
    /// for seg in segments {
    ///     println!("{:?}", seg);
    /// }
    /// ```
    pub fn infer_file<P: AsRef<std::path::Path>>(
        &self,
        wav_path: P,
    ) -> Result<Vec<VoiceText>, Box<dyn std::error::Error>> {
        let mut wav_reader = WavReader::open(wav_path)?;
        match wav_reader.spec().sample_rate {
            8000 => (),
            16000 => (),
            _ => {
                return Err(Box::new(SenseVoiceSmallError::new(
                    "Unsupported sample rate. Expect 8 kHz or 16 kHz.",
                )))
            }
        };
        if wav_reader.spec().sample_format != hound::SampleFormat::Int {
            return Err(Box::new(SenseVoiceSmallError::new(
                "Unsupported sample format. Expect Int.",
            )));
        }

        let content = wav_reader
            .samples()
            .filter_map(|x| x.ok())
            .collect::<Vec<i16>>();
        if content.is_empty() {
            return Err(Box::new(SenseVoiceSmallError::new(
                "content is empty, check your audio file",
            )));
        }

        self.infer_vec(content, wav_reader.spec().sample_rate)
    }

    /// Decodes RKNN output into a transcribed text string.
    ///
    /// This method processes the raw float output from the RKNN model, converts it to token IDs,
    /// removes duplicates and blanks, and decodes it into human-readable text.
    ///
    /// # Arguments
    ///
    /// * `output` - A slice of floats representing the RKNN model's output.
    ///
    /// # Returns
    ///
    /// A `Result` containing the decoded text string.
    ///
    /// # Errors
    ///
    /// Returns an error if the output cannot be reshaped, token decoding fails, or sentencepiece processing encounters an issue.
    fn decode_asr_output(&self, output: &[f32]) -> Result<String, Box<dyn std::error::Error>> {
        // 解析為 [1, n_vocab, n_seq]
        let n_vocab = self.spp.len();
        let output_array = ArrayView3::from_shape((1, n_vocab, self.n_seq), output)?;

        // 在 n_vocab 維度（Axis(1)）上取 argmax
        let token_ids: Vec<i32> = output_array
            .axis_iter(Axis(2)) // 沿著 n_seq=171 維度迭代，得到 [1, 25055] 的視圖
            .into_par_iter()
            .map(|slice| {
                slice
                    .iter()
                    .enumerate()
                    .fold((0, f32::NEG_INFINITY), |(idx, max_val), (i, &val)| {
                        if val > max_val {
                            (i, val)
                        } else {
                            (idx, max_val)
                        }
                    })
                    .0 as i32 // 提取最大值的索引
            })
            .collect();

        // 移除連續重複的 token 和 blank_id=0
        let mut unique_ids = Vec::new();
        let mut prev_id = None;
        for &id in token_ids.iter() {
            if Some(id) != prev_id && id != 0 {
                unique_ids.push(id as u32);
                prev_id = Some(id);
            } else if Some(id) != prev_id {
                prev_id = Some(id);
            }
        }

        // 解碼為文本
        let decoded_text = self.spp.decode_piece_ids(&unique_ids)?;
        Ok(decoded_text)
    }

    /// Destroys the `SenseVoiceSmall` instance, releasing associated resources.
    ///
    /// This method ensures that the RKNN model resources are properly cleaned up.
    ///
    /// # Errors
    ///
    /// Returns an error if the RKNN model destruction fails.
    ///
    /// # Example
    ///
    /// ```
    /// use sensevoice_rs::SenseVoiceSmall;
    ///
    /// let svs = SenseVoiceSmall::init().expect("Failed to initialize");
    /// svs.destroy().expect("Failed to destroy SenseVoiceSmall");
    /// ```
    pub fn destroy(&self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(self.rknn.destroy()?)
    }

    /// Prepares input data for RKNN inference with advanced configuration.
    ///
    /// This method constructs the input tensor by combining language embeddings, event/emotion embeddings,
    /// text normalization embeddings, and scaled audio features, then pads or truncates it to match the expected shape.
    ///
    /// # Arguments
    ///
    /// * `feats` - A 2D array of audio features.
    /// * `language` - Index of the language embedding to use (0 for auto).
    /// * `use_itn` - Whether to use inverse text normalization (true) or not (false).
    ///
    /// # Errors
    ///
    /// Returns an error if tensor concatenation, padding, or RKNN input setting fails.
    fn prepare_rknn_input_advanced(
        &self,
        feats: &Array2<f32>,
        language: usize,
        use_itn: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // 提取嵌入向量
        let language_query = self.embedding.slice(s![language, ..]).insert_axis(Axis(0));
        let text_norm_idx = if use_itn { 14 } else { 15 };
        let text_norm_query = self
            .embedding
            .slice(s![text_norm_idx, ..])
            .insert_axis(Axis(0));
        let event_emo_query = self.embedding.slice(s![1..=2, ..]).to_owned();

        // 縮放語音特徵
        let speech = feats.mapv(|x| x * 0.5);

        // 沿著幀軸串接
        let input_content = ndarray::concatenate(
            Axis(0),
            &[
                language_query.view(),
                event_emo_query.view(),
                text_norm_query.view(),
                speech.view(),
            ],
        )?;

        // 填充或截斷至 [n_seq , 560]
        let total_frames = input_content.shape()[0];
        let padded_input = if total_frames < self.n_seq {
            let mut padded = Array2::zeros((self.n_seq, 560));
            padded
                .slice_mut(s![..total_frames, ..])
                .assign(&input_content);
            padded
        } else {
            input_content.slice(s![..self.n_seq, ..]).to_owned()
        };
        // Add batch dimension
        let input_3d: Array3<f32> = padded_input.insert_axis(Axis(0)); // [1, n_seq , 560]

        // Ensure contiguous memory and flatten to 1D
        let contiguous_input = input_3d.as_standard_layout(); // Row-major contiguous
        let flattened_input: Vec<f32> = contiguous_input
            .into_shape_with_order(1 * self.n_seq * 560)? // Flatten to [95760]
            .to_vec(); // Owned Vec<f32>

        self.rknn.input_set(&mut RknnInput {
            index: 0,             // 根據您的輸入索引設定
            buf: flattened_input, /* 您的數據 */
            pass_through: false,  // 通常設為 false，除非模型需要特殊處理
            type_: RknnTensorType::Float32,
            fmt: RknnTensorFormat::NCHW,
        })?;
        Ok(())
    }
}
