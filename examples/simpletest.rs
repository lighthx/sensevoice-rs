use hf_hub::api::sync::Api;
use sensevoice_rs::SenseVoiceSmall;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut svs = SenseVoiceSmall::init("happyme531/SenseVoiceSmall-RKNN2")?;
    
    let api = Api::new().unwrap();
    let repo = api.model("happyme531/SenseVoiceSmall-RKNN2".to_owned());
    let wav_path = repo.get("output.wav")?;
    let allseg = svs.infer_file(wav_path)?;
    for seg in allseg {
        println!("{:?}", seg);
    }
    
    Ok(svs.destroy()?)
}
