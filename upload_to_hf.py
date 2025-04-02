#!/usr/bin/env python3
from huggingface_hub import HfApi, create_repo
import os
import argparse

def upload_models(hf_token, repo_id, model_dir):
    """
    Upload model files to Hugging Face Hub
    
    Args:
        hf_token (str): Hugging Face access token
        repo_id (str): Repository ID in format 'username/repo-name'
        model_dir (str): Path to directory containing model files
    """
    api = HfApi(token=hf_token)
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, token=hf_token, repo_type="model", exist_ok=True)
        print(f"Repository {repo_id} created/verified")
    except Exception as e:
        print(f"Error creating repository: {e}")
        raise

    # Upload each model file
    model_files = [
        'temp_model.joblib',
        'weather_model.joblib',
        'conditions_model.joblib',
        'scaler.joblib',
        'label_encoder.joblib',
        'README.md'
    ]

    for filename in model_files:
        file_path = os.path.join(model_dir, filename)
        if os.path.exists(file_path):
            print(f"Uploading {filename}...")
            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=filename,
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"Successfully uploaded {filename}")
            except Exception as e:
                print(f"Error uploading {filename}: {e}")
                raise
        else:
            print(f"Warning: {filename} not found in {model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upload weather prediction models to Hugging Face Hub')
    parser.add_argument('--hf-token', required=True, help='Hugging Face access token')
    parser.add_argument('--repo-id', required=True, help='Repository ID in format username/repo-name')
    parser.add_argument('--model-dir', default='weather_models', help='Path to model directory')
    
    args = parser.parse_args()
    
    upload_models(
        hf_token=args.hf_token,
        repo_id=args.repo_id,
        model_dir=args.model_dir
    )