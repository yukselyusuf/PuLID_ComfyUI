from eva_clip.pretrained import get_pretrained_cfg, download_pretrained

def main():
    # Get the configuration for EVA02-CLIP-L-14-336 with eva_clip tag
    model_name = "EVA02-CLIP-L-14-336"
    model_tag = "eva_clip"
    
    # Get the model configuration
    cfg = get_pretrained_cfg(model_name, model_tag)
    
    print(f"Downloading {model_name} ({model_tag})...")
    
    # Download the model
    # You can specify a custom cache_dir if needed
    model_path = download_pretrained(cfg, force_hf_hub=True)
    
    print(f"Model downloaded successfully to: {model_path}")

if __name__ == "__main__":
    main() 