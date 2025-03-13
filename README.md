# SenseVoiceSmall

A Rust-based, Rknn as backend ASR. Running on the low cost SBC npu, fast and chep.

## Install

You should install rknn.so first.

```bash
sudo curl -L https://github.com/airockchip/rknn-toolkit2/raw/refs/heads/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so -o /lib/librknnrt.so
```

## 範例

```
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

## 輸出範例

```
Speech segment: 60ms - 6120ms
<|zh|><|HAPPY|><|BGM|><|woitn|>大家好喵今天给大家分享的是在线一线语音生成网站的合集能够更加分富
Speech segment: 6060ms - 12120ms
<|zh|><|HAPPY|><|BGM|><|woitn|>方面大家选择自己想要生成的角色进入网站可以看到所有的删至
Speech segment: 12060ms - 18120ms
<|zh|><|NEUTRAL|><|BGM|><|woitn|>模型都在这里选择你想要擅藏的角色点击进入就来到我
Speech segment: 18060ms - 24120ms
<|zh|><|HAPPY|><|BGM|><|woitn|>到了生成的页面在文本框内输入你想要生成的内容然后点击生成就好了
Speech segment: 24060ms - 30120ms
<|zh|><|NEUTRAL|><|BGM|><|woitn|>另外呢因为每次的生成结果都会有一些不一样的地方如果您觉得第一次的生成结果
Speech segment: 30060ms - 36120ms
<|zh|><|NEUTRAL|><|BGM|><|woitn|>生成效果不好的话可以尝试重新生成也可以稍微调取一下像的住址再生成试试
Speech segment: 36060ms - 39840ms
<|zh|><|NEUTRAL|><|BGM|><|woitn|>使用时一定要遵守法律法规不可以损害刷害人的形象哦
```