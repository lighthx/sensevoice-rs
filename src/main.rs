use sensevoice_rs::fsmn_vad::{VADXOptions, FSMNVad};
use hf_hub::api::sync::Api;
use hound::WavReader;
use ndarray::{s, Array2, Array3, ArrayView3, Axis};
use ndarray_npy::ReadNpyExt;
use rknn_rs::prelude::*;

use ndarray::parallel::prelude::*;
use sentencepiece::SentencePieceProcessor;
use sensevoice_rs::wavfrontend::{WavFrontend, WavFrontendConfig};
use std::{fs::File, io::BufReader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api = Api::new().unwrap();
    let repo = api.model("happyme531/SenseVoiceSmall-RKNN2".to_owned());
    let wav_path = repo.get("output.wav")?;
    let fsmn_path = repo.get("fsmnvad-offline.onnx")?;
    let embedding_path = repo.get("embedding.npy")?;
    let rknn_path = repo.get("sense-voice-encoder.rknn")?;
    let sentence_path = repo.get("chn_jpn_yue_eng_ko_spectok.bpe.model")?;
    let fsmn_am_path = repo.get("fsmn-am.mvn")?;
    let am_path = repo.get("am.mvn")?;

    let mut wav_reader = WavReader::open(wav_path)?;
    match wav_reader.spec().sample_rate {
        8000 => (),
        16000 => (),
        _ => panic!("Unsupported sample rate. Expect 8 kHz or 16 kHz."),
    };
    if wav_reader.spec().sample_format != hound::SampleFormat::Int {
        panic!("Unsupported sample format. Expect Int.");
    }

    let content = wav_reader
        .samples()
        .filter_map(|x| x.ok())
        .collect::<Vec<i16>>();
    assert!(!content.is_empty());

    let mut fsmn = FSMNVad::new(
        wav_reader.spec().sample_rate,
        fsmn_path,
        VADXOptions::default(),
    )
    .unwrap();

    // Load embedding.npy
    let embedding_file = File::open(embedding_path)?;
    let embedding_reader = BufReader::new(embedding_file);
    let embedding: Array2<f32> = Array2::read_npy(embedding_reader)?;
    assert_eq!(embedding.shape()[1], 560, "Embedding dimension must be 560");

    let mut rknn = Rknn::rknn_init(rknn_path)?;
    let spp = SentencePieceProcessor::open(sentence_path)?;

    let n_vocab = spp.len(); // 目前測試的大小是 25055
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

    // 提取特徵
    let audio_feats = vad_frontend.extract_features(&content)?;

    // 進行 VAD 推理
    let segments = fsmn.infer_vad(audio_feats, &content, true)?;

    // 處理語音片段
    for (start_ms, end_ms) in segments {
        println!("Speech segment: {}ms - {}ms", start_ms, end_ms);
        let start_sample =
            (start_ms as f32 / 1000.0 * wav_reader.spec().sample_rate as f32) as usize;
        let end_sample = (end_ms as f32 / 1000.0 * wav_reader.spec().sample_rate as f32) as usize;
        let segment = &content[start_sample..end_sample];
        // 提取特徵
        let audio_feats = asr_frontend.extract_features(segment)?;

        // 準備 RKNN 輸入
        prepare_rknn_input_advanced(&audio_feats, &embedding, &mut rknn, 0, false, n_seq)?; // language=0 (auto), use_itn=false
        rknn.run()?;
        let mut asr_output = rknn.outputs_get_raw::<f32>()?;
        let asr_text = decode_asr_output(&asr_output.data, &spp, n_seq, n_vocab)?;
        rknn.outputs_release(&mut asr_output)?; // 資料會被丟棄，不可再用asr_output
                                                // 處理輸出並解碼為文字
        println!("{}", asr_text);
    }
    rknn.destroy()?;
    Ok(())
}

fn decode_asr_output(
    output: &[f32],
    spp: &SentencePieceProcessor,
    n_seq: usize,   // 例如 171
    n_vocab: usize, // 例如 25055
) -> Result<String, Box<dyn std::error::Error>> {
    // 解析為 [1, n_vocab, n_seq]
    let output_array = ArrayView3::from_shape((1, n_vocab, n_seq), output)?;

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

    let decoded_text = spp.decode_piece_ids(&unique_ids)?;
    Ok(decoded_text)
}

fn prepare_rknn_input_advanced(
    feats: &Array2<f32>,
    embedding: &Array2<f32>,
    rknn: &mut Rknn,
    language: usize,
    use_itn: bool,
    n_seq: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // 提取嵌入向量
    let language_query = embedding.slice(s![language, ..]).insert_axis(Axis(0));
    let text_norm_idx = if use_itn { 14 } else { 15 };
    let text_norm_query = embedding.slice(s![text_norm_idx, ..]).insert_axis(Axis(0));
    let event_emo_query = embedding.slice(s![1..=2, ..]).to_owned();

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
    let padded_input = if total_frames < n_seq {
        let mut padded = Array2::zeros((n_seq, 560));
        padded
            .slice_mut(s![..total_frames, ..])
            .assign(&input_content);
        padded
    } else {
        input_content.slice(s![..n_seq, ..]).to_owned()
    };
    // Add batch dimension
    let input_3d: Array3<f32> = padded_input.insert_axis(Axis(0)); // [1, n_seq , 560]

    // Ensure contiguous memory and flatten to 1D
    let contiguous_input = input_3d.as_standard_layout(); // Row-major contiguous
    let flattened_input: Vec<f32> = contiguous_input
        .into_shape_with_order(1 * n_seq * 560)? // Flatten to [95760]
        .to_vec(); // Owned Vec<f32>

    rknn.input_set(&mut RknnInput {
        index: 0,             // 根據您的輸入索引設定
        buf: flattened_input, /* 您的數據 */
        pass_through: false,  // 通常設為 false，除非模型需要特殊處理
        type_: RknnTensorType::Float32,
        fmt: RknnTensorFormat::NCHW,
    })?;
    Ok(())
}
