# SenseVoiceSmall

A Rust-based, Rknn as backend ASR. Running on the low cost SBC npu, fast and chep.

## Install

You should install rknn.so first.

```bash
sudo curl -L https://github.com/airockchip/rknn-toolkit2/raw/refs/heads/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so -o /lib/librknnrt.so
```

## Example

Also see [examples](examples) dictionary
```Rust
use hf_hub::api::sync::Api;
use sensevoice_rs::SenseVoiceSmall;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut svs = SenseVoiceSmall::init()?;
    
    let api = Api::new().unwrap();
    let repo = api.model("happyme531/SenseVoiceSmall-RKNN2".to_owned());
    let wav_path = repo.get("output.wav")?;
    let allseg = svs.infer_file(wav_path)?;
    for seg in allseg {
        println!("{:?}", seg);
    }
    
    Ok(svs.destroy()?)
}
```

## Output Example

```Rust
VoiceText { start_ms: 60, end_ms: 6120, language: Zh, emotion: Happy, event: Bgm, punctuation_normalization: Woitn, content: "大家好喵今天给大家分享的是在线一线语音生成网站的合集能够更加分富" }
VoiceText { start_ms: 6060, end_ms: 12120, language: Zh, emotion: Happy, event: Bgm, punctuation_normalization: Woitn, content: "方面大家选择自己想要生成的角色进入网站可以看到所有的删至" }
VoiceText { start_ms: 12060, end_ms: 18120, language: Zh, emotion: Neutral, event: Bgm, punctuation_normalization: Woitn, content: "模型都在这里选择你想要擅藏的角色点击进入就来到我" }
VoiceText { start_ms: 18060, end_ms: 24120, language: Zh, emotion: Happy, event: Bgm, punctuation_normalization: Woitn, content: "到了生成的页面在文本框内输入你想要生成的内容然后点击生成就好了" }
VoiceText { start_ms: 24060, end_ms: 30120, language: Zh, emotion: Neutral, event: Bgm, punctuation_normalization: Woitn, content: "另外呢因为每次的生成结果都会有一些不一样的地方如果您觉得第一次的生成结果" }
VoiceText { start_ms: 30060, end_ms: 36120, language: Zh, emotion: Neutral, event: Bgm, punctuation_normalization: Woitn, content: "生成效果不好的话可以尝试重新生成也可以稍微调取一下像的住址再生成试试" }
VoiceText { start_ms: 36060, end_ms: 39840, language: Zh, emotion: Neutral, event: Bgm, punctuation_normalization: Woitn, content: "使用时一定要遵守法律法规不可以损害刷害人的形象哦" }
```

## Use as rust library

Because the `kaldi-fbank-rust` library not upload to crates.io, so you should add this crate by yourself. Maybe we can breakaway from nonpublish library in the feature.

git clone my fork of `kaldi-fbank-rust`.

```bash
git clone https://github.com/darkautism/kaldi-fbank-rust/
```

In your Cargo.toml
```
[patch.crates-io]
kaldi-fbank-rust = {path = "./kaldi-fbank-rust"}
```

No you can write your code with sensevoice_rs.