import os
import gc
import torch
import pandas as pd
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,7"

from utils.curvature import compute_hessian_metrics
from utils.connectivity import train_bezier_curve
from utils.similarity import extract_and_compare

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_pythia_model_only(step, size="70m-seed1"):
    """Loads just the model weights since the tokenizer is now global"""
    model_name = f"EleutherAI/pythia-{size}"
    revision = f"step{step}" 
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        revision=revision,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        use_safetensors=False, 
    ).to(device)
    return model

def load_evaluation_data(tokenizer, benchmark="arc_challenge", batch_size=8, num_samples=256):
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    texts = [item["question"] for item in dataset][:num_samples]
            
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    class BenchmarkDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = item['input_ids'].clone()
            return item
        def __len__(self):
            return len(self.encodings.input_ids)
            
    hf_dataset = BenchmarkDataset(encoded)
    return DataLoader(hf_dataset, batch_size=batch_size, shuffle=True)


def run_full_study():
    print(f"Starting Full Phase Diagram Study on {device}...")
    
    # --- NEW: ONE-TIME DATA LOADING ---
    print("Pre-fetching Tokenizer and Dataset to prevent API Rate Limits...")
    base_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        
    global_dataloader = load_evaluation_data(base_tokenizer, num_samples=256)
    batch = next(iter(global_dataloader))
    static_inputs = {k: v.to(device) for k, v in batch.items()}
    # ----------------------------------
    
    model_sizes = ["14m", "31m", "70m", "160m", "410m"] 
    training_steps = [1000, 10000, 30000, 70000, 100000, 143000]
    
    results = []
    output_file = f"phase_diagram_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    for size in model_sizes:
        for step in training_steps:
            print(f"\n{'='*50}")
            print(f"EVALUATING: Size={size} | Step={step}")
            print(f"{'='*50}")
            
            model_seed1 = None
            model_seed2 = None
            curve_model = None
            
            try:
                print("Loading models...")
                model_seed1 = load_pythia_model_only(step, size=f"{size}-seed1")
                model_seed2 = load_pythia_model_only(step, size=f"{size}-seed2")
                
                print("Calculating Curvature...")
                eig_1, trace_1 = compute_hessian_metrics(model_seed1, static_inputs, device)
                eig_2, trace_2 = compute_hessian_metrics(model_seed2, static_inputs, device)
                
                print("Calculating CKA Similarity...")
                cka_score = extract_and_compare(model_seed1, model_seed2, static_inputs, device)
                
                print("Calculating Mode Connectivity (Bezier Curve)...")
                curve_model = train_bezier_curve(model_seed1, model_seed2, global_dataloader, epochs=3, device=device)
                
                with torch.no_grad():
                    curve_model.eval()
                    midpoint_loss = curve_model(0.5, static_inputs).item()
                    seed1_loss = model_seed1(**static_inputs).loss.item()
                    seed2_loss = model_seed2(**static_inputs).loss.item()
                
                avg_endpoint_loss = (seed1_loss + seed2_loss) / 2
                loss_barrier = midpoint_loss - avg_endpoint_loss
                
                row_data = {
                    "Model_Size": size,
                    "Training_Step": step,
                    "Trace_Seed1": trace_1,
                    "Trace_Seed2": trace_2,
                    "Avg_Trace": (trace_1 + trace_2) / 2,
                    "CKA_Similarity": cka_score,
                    "Midpoint_Loss": midpoint_loss,
                    "Avg_Endpoint_Loss": avg_endpoint_loss,
                    "Loss_Barrier": loss_barrier
                }
                results.append(row_data)
                print(f"SUCCESS -> Barrier: {loss_barrier:.4f} | CKA: {cka_score:.4f} | Trace: {row_data['Avg_Trace']:.4f}")
                
            except Exception as e:
                print(f"FAILED evaluating Size={size} at Step={step}. Error: {e}")
                
            finally:
                print("Flushing VRAM...")
                if model_seed1 is not None: del model_seed1
                if model_seed2 is not None: del model_seed2
                if curve_model is not None: del curve_model
                
                gc.collect()             
                torch.cuda.empty_cache() 
                
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            
    print(f"\nStudy Complete! Full dataset saved to {output_file}")

if __name__ == "__main__":
    run_full_study()