use hf_hub::api::sync::Api;
use sensevoice_rs::SenseVoiceSmall;

// 自定義 DelayedReader 模擬流式輸入，每次讀取延遲 0.5 秒
struct DelayedReader {
    file: File,
    chunk_size: usize,
    delay: Duration,
    buffer: Vec<u8>,
    pos: usize,
}

impl DelayedReader {
    fn new(mut file: File, chunk_size: usize, delay: Duration) -> Self {
        // 跳過 WAV 頭部（假設 44 字節）
        file.seek(SeekFrom::Start(44)).unwrap();
        DelayedReader {
            file,
            chunk_size,
            delay,
            buffer: Vec::new(),
            pos: 0,
        }
    }

    fn fill_buffer(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.buffer.clear();
        let mut temp_buf = vec![0u8; self.chunk_size];
        let bytes_read = self.file.read(&mut temp_buf)?;
        if bytes_read > 0 {
            self.buffer.extend_from_slice(&temp_buf[..bytes_read]);
            self.pos = 0;
            // 模擬流式延遲
            thread::sleep(self.delay);
        }
        Ok(())
    }
}

impl Read for DelayedReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.buffer.len() {
            self.fill_buffer()?;
            if self.buffer.is_empty() {
                return Ok(0); // 文件結束
            }
        }

        let remaining = self.buffer.len() - self.pos;
        let to_copy = std::cmp::min(remaining, buf.len());
        buf[..to_copy].copy_from_slice(&self.buffer[self.pos..self.pos + to_copy]);
        self.pos += to_copy;
        Ok(to_copy)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut svs = SenseVoiceSmall::init("happyme531/SenseVoiceSmall-RKNN2")?;

    let api = Api::new().unwrap();
    let repo = api.model("happyme531/SenseVoiceSmall-RKNN2".to_owned());
    let wav_path = repo.get("output.wav")?;
    let file_for_stream = File::open(&wav_path)?;
    let delayed_reader = DelayedReader::new(file_for_stream, 1024, Duration::from_millis(500));

    let allseg = svs.infer_stream(delayed_reader)?;
    for seg in allseg {
        println!("{:?}", seg);
    }

    Ok(svs.destroy()?)
}
