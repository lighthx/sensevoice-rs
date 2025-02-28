use ndarray::Array2;
use ndarray::Ix3;
use ort::session::Session;
use std::path::Path;

use ndarray::Array3;

#[derive(Debug, PartialEq)]
enum VadStateMachine {
    StartPointNotDetected,
    InSpeechSegment,
    EndPointDetected,
}

// 幀狀態
#[derive(Debug, PartialEq, Clone, Copy)]
enum FrameState {
    Sil,    // 靜音
    Speech, // 語音
}

// 語音狀態變化
#[derive(Debug, PartialEq)]
enum AudioChangeState {
    Sil2Sil,       // 靜音保持
    Sil2Speech,    // 靜音到語音
    Speech2Speech, // 語音保持
    Speech2Sil,    // 語音到靜音
}

// 滑動窗口檢測器
#[derive(Debug)]
struct WindowDetector {
    win_size_frame: usize,             // 窗口大小（幀數）
    sil_to_speech_frmcnt_thres: usize, // 靜音到語音的幀數閾值
    speech_to_sil_frmcnt_thres: usize, // 語音到靜音的幀數閾值
    win_sum: usize,                    // 窗口內語音幀的總和
    win_state: Vec<usize>,             // 窗口內每幀的狀態（0 或 1）
    cur_win_pos: usize,                // 當前窗口位置
    pre_frame_state: FrameState,       // 前一狀態
}

impl WindowDetector {
    fn new(
        window_size_ms: usize,
        sil_to_speech_time: usize,
        speech_to_sil_time: usize,
        frame_size_ms: usize,
    ) -> Self {
        let win_size_frame = window_size_ms / frame_size_ms;
        WindowDetector {
            win_size_frame,
            sil_to_speech_frmcnt_thres: sil_to_speech_time / frame_size_ms,
            speech_to_sil_frmcnt_thres: speech_to_sil_time / frame_size_ms,
            win_sum: 0,
            win_state: vec![0; win_size_frame],
            cur_win_pos: 0,
            pre_frame_state: FrameState::Sil,
        }
    }

    fn detect_one_frame(&mut self, frame_state: FrameState) -> AudioChangeState {
        let cur_state = if frame_state == FrameState::Speech {
            1
        } else {
            0
        };
        self.win_sum -= self.win_state[self.cur_win_pos]; // 移除舊狀態
        self.win_sum += cur_state; // 添加新狀態
        self.win_state[self.cur_win_pos] = cur_state;
        self.cur_win_pos = (self.cur_win_pos + 1) % self.win_size_frame;

        if self.pre_frame_state == FrameState::Sil
            && self.win_sum >= self.sil_to_speech_frmcnt_thres
        {
            self.pre_frame_state = FrameState::Speech;
            AudioChangeState::Sil2Speech
        } else if self.pre_frame_state == FrameState::Speech
            && self.win_sum < self.speech_to_sil_frmcnt_thres
        {
            self.pre_frame_state = FrameState::Sil;
            AudioChangeState::Speech2Sil
        } else if self.pre_frame_state == FrameState::Sil {
            AudioChangeState::Sil2Sil
        } else {
            AudioChangeState::Speech2Speech
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VADXOptions {
    sample_rate: u32,                // 採樣率
    frame_in_ms: usize,              // 幀間隔（ms）
    frame_length_ms: usize,          // 幀長度（ms）
    decibel_thres: f32,              // 分貝閾值
    snr_thres: f32,                  // 信噪比閾值
    speech_noise_thres: f32,         // 語音與靜音概率差閾值
    max_end_silence_time: usize,     // 最大結束靜音時間（ms）
    window_size_ms: usize,           // 窗口大小（ms）
    sil_to_speech_time_thres: usize, // 靜音到語音閾值（ms）
    speech_to_sil_time_thres: usize, // 語音到靜音閾值（ms）
    max_single_segment_time: usize,  // 最大單段時間（ms）

    detect_mode: VadDetectMode, // 检测模式
    speech_2_noise_ratio: f32,  // 語音與噪音比
    noise_frame_num_used_for_snr: f32, // 用於信噪比計算的噪音幀數
                                // 可根據需要添加其他參數
}

impl VADXOptions {
    pub fn default() -> Self {
        Self {
            sample_rate: 16000,
            frame_in_ms: 60,
            frame_length_ms: 25,
            decibel_thres: -100.0,
            snr_thres: -100.0,
            speech_noise_thres: 0.6,
            max_end_silence_time: 800,
            window_size_ms: 200,
            sil_to_speech_time_thres: 150,
            speech_to_sil_time_thres: 150,
            max_single_segment_time: 6000,
            speech_2_noise_ratio: 1.0,
            noise_frame_num_used_for_snr: 100.0,
            detect_mode: VadDetectMode::default(),
        }
    }
}

#[derive(Debug, Clone)]
struct FrameData {
    start_ms: i32,
    end_ms: i32,
    contain_seg_start_point: bool,
    contain_seg_end_point: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
enum VadDetectMode {
    SingleUtterance,
    #[default]
    MutipleUtterance,
}

#[derive(Debug)]
pub struct FSMNVad {
    session: Session,
    decibel: Vec<f32>,
    output_data_buf: Vec<FrameData>,
    output_data_buf_offset: usize,
    opts: VADXOptions,
    vad_state: VadStateMachine,
    window_detector: WindowDetector,
    frm_cnt: usize, // 總幀數
    continous_silence_frame_count: usize,
    confirmed_start_frame: Option<usize>,
    confirmed_end_frame: Option<usize>,
    latest_confirmed_speech_frame: usize,
}

impl FSMNVad {
    pub fn new(
        sample_rate: u32,
        model_path: impl AsRef<Path>,
        opts: VADXOptions,
    ) -> Result<Self, ort::Error> {
        let session = ort::session::Session::builder()?.commit_from_file(model_path)?;
        Ok(Self {
            session,
            decibel: Vec::new(),
            output_data_buf: Vec::new(),
            output_data_buf_offset: 0,
            opts,
            vad_state: VadStateMachine::StartPointNotDetected,
            window_detector: WindowDetector::new(
                opts.window_size_ms,
                opts.sil_to_speech_time_thres,
                opts.speech_to_sil_time_thres,
                opts.frame_in_ms,
            ),
            frm_cnt: 0,
            continous_silence_frame_count: 0,
            confirmed_start_frame: None,
            confirmed_end_frame: None,
            latest_confirmed_speech_frame: 0,
        })
    }

    fn get_frame_state(
        &self,
        frame_idx: usize,
        noise_average_decibel: &mut f32,
        scores: &Array3<f32>,
    ) -> FrameState {
        let cur_decibel = self.decibel[frame_idx];
        let cur_snr = cur_decibel - *noise_average_decibel;
        if cur_decibel < self.opts.decibel_thres {
            return FrameState::Sil;
        }

        let sil_pdf_ids = &[0]; // 假設 sil_pdf_ids = [0]，可根據實際情況調整
        let sil_pdf_scores: Vec<f32> = sil_pdf_ids
            .iter()
            .map(|&id| scores[[0, frame_idx, id as usize]])
            .collect();
        let sum_score: f32 = sil_pdf_scores.iter().sum();

        let epsilon = 1e-10; // 避免 log(0)
        let noise_prob = (sum_score + epsilon).ln() * self.opts.speech_2_noise_ratio;
        let speech_prob = (1.0 - sum_score + epsilon).ln();

        if speech_prob.exp() >= noise_prob.exp() + self.opts.speech_noise_thres {
            if cur_snr >= self.opts.snr_thres && cur_decibel >= self.opts.decibel_thres {
                return FrameState::Speech;
            }
        } else {
            if *noise_average_decibel < -99.9 {
                *noise_average_decibel = cur_decibel;
            } else {
                *noise_average_decibel = (cur_decibel
                    + *noise_average_decibel * (self.opts.noise_frame_num_used_for_snr - 1.0))
                    / self.opts.noise_frame_num_used_for_snr;
            }
        }
        FrameState::Sil
    }

    pub fn infer_vad(
        &mut self,
        feats: Array2<f32>,
        waveform: &[i16],
        is_final: bool,
    ) -> Result<Vec<(i32, i32)>, Box<dyn std::error::Error>> {
        self.reset_detection(); // 確保重置所有狀態
        self.compute_decibel(waveform);

        let score = self.compute_scores(feats)?;

        if !is_final {
            self.detect_common_frames(&score)?;
        } else {
            self.detect_last_frames(&score)?;
        }

        let mut segments = Vec::new();
        let mut i = self.output_data_buf_offset;
        while i < self.output_data_buf.len() {
            let frame = &self.output_data_buf[i];
            if frame.contain_seg_start_point && frame.contain_seg_end_point {
                segments.push((frame.start_ms, frame.end_ms)); // 收集完整的語音段
                self.output_data_buf_offset += 1; // 更新偏移量
            }
            i += 1;
        }

        if is_final {
            self.reset_detection();
        }

        Ok(segments)
    }

    fn compute_decibel(&mut self, waveform: &[i16]) {
        let frame_size =
            (self.opts.frame_length_ms as f32 * self.opts.sample_rate as f32 / 1000.0) as usize;
        let frame_shift =
            (self.opts.frame_in_ms as f32 * self.opts.sample_rate as f32 / 1000.0) as usize;
        self.decibel = (0..(waveform.len() - frame_size + 1))
            .step_by(frame_shift)
            .map(|i| {
                let frame = &waveform[i..i + frame_size];
                // 正規化到 [-1.0, 1.0]
                let energy: f32 = frame.iter().map(|&x| (x as f32 / 32768.0).powi(2)).sum();
                10.0 * (energy + 1e-6).log10()
            })
            .collect();
    }

    fn compute_scores(
        &mut self,
        feats: Array2<f32>,
    ) -> Result<Array3<f32>, Box<dyn std::error::Error>> {
        // input:name: speech, tensor: float32[1,feats_length,400]
        // output:name: logits, tensor: float32[1,Softmaxlogits_dim_1,248] (不要再眼瞎看錯了，是3維，那不是逗點)

        let feats_3d = feats.insert_axis(ndarray::Axis(0));
        let inputs = ort::inputs![
            "speech" => feats_3d.clone()
        ]?;
        let outputs = self.session.run(inputs)?;
        let logits = outputs["logits"].try_extract_tensor::<f32>()?;
        let dyn_to_fix3array = logits.into_dimensionality::<Ix3>()?;

        Ok(dyn_to_fix3array.to_owned())
    }

    fn detect_common_frames(
        &mut self,
        scores: &Array3<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.vad_state == VadStateMachine::EndPointDetected {
            return Ok(());
        }

        let mut noise_average_decibel = -100.0; // 這裡需要動態計算，暫用默認值
        for i in (0..scores.dim().1).rev() {
            let frame_idx = self.frm_cnt + scores.dim().1 - 1 - i; // 從最新幀向前，不要再改了 適應realtime
            let frame_state = self.get_frame_state(i, &mut noise_average_decibel, scores); // 使用本地索引 i
            let state_change = self.window_detector.detect_one_frame(frame_state);

            match state_change {
                AudioChangeState::Sil2Speech => {
                    self.continous_silence_frame_count = 0; // 清空靜音計數
                    if self.vad_state == VadStateMachine::StartPointNotDetected {
                        let start_frame =
                            frame_idx.saturating_sub(self.window_detector.win_size_frame); // 計算語音開始幀
                        self.confirmed_start_frame = Some(start_frame); // 記錄開始幀
                        self.vad_state = VadStateMachine::InSpeechSegment; // 進入語音段狀態
                        self.output_data_buf.push(FrameData {
                            start_ms: (start_frame * self.opts.frame_in_ms) as i32,
                            end_ms: (frame_idx * self.opts.frame_in_ms) as i32, // 初始結束時間為當前幀
                            contain_seg_start_point: true,
                            contain_seg_end_point: false,
                        }); // 添加新的語音段到緩衝區
                    }
                }
                AudioChangeState::Speech2Speech => {
                    self.continous_silence_frame_count = 0;
                    if self.vad_state == VadStateMachine::InSpeechSegment {
                        if let Some(start) = self.confirmed_start_frame {
                            if frame_idx >= start
                                && frame_idx - start
                                    > self.opts.max_single_segment_time / self.opts.frame_in_ms
                            {
                                self.confirmed_end_frame = Some(frame_idx);
                                self.vad_state = VadStateMachine::EndPointDetected;
                                self.output_data_buf.last_mut().unwrap().end_ms =
                                    (frame_idx * self.opts.frame_in_ms) as i32;
                                self.output_data_buf
                                    .last_mut()
                                    .unwrap()
                                    .contain_seg_end_point = true;
                                // 多語句模式下重置
                                if self.opts.detect_mode == VadDetectMode::MutipleUtterance {
                                    self.reset_detection();
                                }
                            } else {
                                self.latest_confirmed_speech_frame = frame_idx;
                                self.output_data_buf.last_mut().unwrap().end_ms =
                                    (frame_idx * self.opts.frame_in_ms) as i32;
                            }
                        }
                    }
                }
                AudioChangeState::Speech2Sil => {
                    self.continous_silence_frame_count = 0; // 重置靜音計數
                    if self.vad_state == VadStateMachine::InSpeechSegment {
                        self.confirmed_end_frame = Some(frame_idx); // 記錄語音結束幀
                        self.vad_state = VadStateMachine::EndPointDetected; // 更新狀態
                        self.output_data_buf.last_mut().unwrap().end_ms =
                            (frame_idx * self.opts.frame_in_ms) as i32; // 更新結束時間
                        self.output_data_buf
                            .last_mut()
                            .unwrap()
                            .contain_seg_end_point = true; // 標記段落結束
                        if self.opts.detect_mode == VadDetectMode::MutipleUtterance {
                            self.reset_detection(); // 在多語句模式下重置狀態機
                        }
                    }
                }
                AudioChangeState::Sil2Sil => {
                    self.continous_silence_frame_count += 1;
                    if self.vad_state == VadStateMachine::InSpeechSegment
                        && self.continous_silence_frame_count * self.opts.frame_in_ms
                            >= self.opts.max_end_silence_time
                    {
                        self.confirmed_end_frame = Some(frame_idx);
                        self.vad_state = VadStateMachine::EndPointDetected;
                        self.output_data_buf.last_mut().unwrap().end_ms =
                            (frame_idx * self.opts.frame_in_ms) as i32;
                        self.output_data_buf
                            .last_mut()
                            .unwrap()
                            .contain_seg_end_point = true;
                    }
                }
            }
        }
        self.frm_cnt += scores.dim().1;

        Ok(())
    }

    fn detect_last_frames(
        &mut self,
        scores: &Array3<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.detect_common_frames(scores)?;
        if self.vad_state == VadStateMachine::InSpeechSegment {
            let last_frame = self.frm_cnt + scores.dim().1 - 1;
            self.confirmed_end_frame = Some(last_frame);
            self.output_data_buf.last_mut().unwrap().end_ms =
                (self.latest_confirmed_speech_frame * self.opts.frame_in_ms) as i32;
            self.output_data_buf
                .last_mut()
                .unwrap()
                .contain_seg_end_point = true;
            self.vad_state = VadStateMachine::EndPointDetected;
        }
        Ok(())
    }

    fn reset_detection(&mut self) {
        self.vad_state = VadStateMachine::StartPointNotDetected; // 重置狀態機
        self.continous_silence_frame_count = 0; // 重置靜音幀計數
        self.confirmed_start_frame = None; // 清空語音開始幀
        self.confirmed_end_frame = None; // 清空語音結束幀
        self.latest_confirmed_speech_frame = 0; // 重置最近確認的語音幀
        self.window_detector = WindowDetector::new(
            self.opts.window_size_ms,
            self.opts.sil_to_speech_time_thres,
            self.opts.speech_to_sil_time_thres,
            self.opts.frame_in_ms,
        ); // 重新創建 WindowDetector，確保其內部狀態清零
    }
}
