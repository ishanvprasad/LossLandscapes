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

def load_evaluation_data(tokenizer, benchmark="wikitext", batch_size=8, num_samples=256):
    print(f"Loading benchmark: {benchmark}...")
    
    if benchmark == "wikitext":
        # Load the data Pythia was actually trained on
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        # Filter out empty lines which are common in wikitext
        texts = [x["text"] for x in dataset if len(x["text"]) > 10][:num_samples]
    else:
        # Fallback to ARC
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
    
    # --- 1. PRE-FETCH DATA (SPLIT INTO TRAIN / TEST) ---
    print("Pre-fetching Tokenizer and Data Splits...")
    base_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        
    # Load a slightly larger slice of Wikitext (e.g., 512 samples)
    # We will use the first 256 for TRAINING the curve
    # We will use the last 256 for MEASURING the barrier
    full_dataset = load_evaluation_data(base_tokenizer, benchmark="wikitext", num_samples=512)
    
    # Manually split the batch
    all_inputs = next(iter(full_dataset)) # Get all 512 samples in one tensor dict
    
    # Split into two sets of 256
    train_inputs = {k: v[:256].to(device) for k, v in all_inputs.items()}
    eval_inputs  = {k: v[256:].to(device) for k, v in all_inputs.items()}
    
    # Create a mini dataloader just for the training inputs
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data): self.data = data
        def __getitem__(self, idx): return {k: v[idx] for k, v in self.data.items()}
        def __len__(self): return len(self.data['input_ids'])
        
    train_loader = DataLoader(SimpleDataset(train_inputs), batch_size=8, shuffle=True)
    # -------------------------------------------------------

    model_sizes = ["14m", "31m", "70m", "160m", "410m"] 
    training_steps = [1000, 10000, 30000, 70000, 100000, 143000]
    
    results = []
    output_file = f"phase_diagram_results_SPLIT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    for size in model_sizes:
        for step in training_steps:
            print(f"\nEvaluating {size} @ Step {step}...")
            
            try:
                # Load Models (No SafeTensors spam)
                model_seed1 = load_pythia_model_only(step, size=f"{size}-seed1")
                model_seed2 = load_pythia_model_only(step, size=f"{size}-seed2")
                
                # A. CALC CURVATURE (On Eval Set)
                # We measure sharpness on the HELD OUT set
                _, trace_1 = compute_hessian_metrics(model_seed1, eval_inputs, device)
                _, trace_2 = compute_hessian_metrics(model_seed2, eval_inputs, device)
                
                # B. CALC CKA (On Eval Set)
                cka_score = extract_and_compare(model_seed1, model_seed2, eval_inputs, device)
                
                # C. TRAIN BEZIER CURVE (On TRAIN Set)
                # The curve learns to connect the models using 'train_inputs'
                print("  Training Bezier Curve on TRAIN set...")
                curve_model = train_bezier_curve(model_seed1, model_seed2, train_loader, epochs=3, device=device)
                
                # D. MEASURE BARRIER (On EVAL Set)
                # We check if that connection generalizes to 'eval_inputs'
                print("  Measuring Barrier on EVAL set...")
                with torch.no_grad():
                    curve_model.eval()
                    midpoint_loss = curve_model(0.5, eval_inputs).item() # <--- Crucial: Evaluated on UNSEEN data
                    seed1_loss = model_seed1(**eval_inputs).loss.item()
                    seed2_loss = model_seed2(**eval_inputs).loss.item()
                
                avg_endpoint_loss = (seed1_loss + seed2_loss) / 2
                loss_barrier = midpoint_loss - avg_endpoint_loss
                
                row_data = {
                    "Model_Size": size,
                    "Training_Step": step,
                    "Trace_Seed1": trace_1,
                    "Avg_Trace": (trace_1 + trace_2) / 2,
                    "CKA_Similarity": cka_score,
                    "Midpoint_Loss": midpoint_loss,
                    "Avg_Endpoint_Loss": avg_endpoint_loss,
                    "Loss_Barrier": loss_barrier
                }
                results.append(row_data)
                print(f"  Result -> Barrier: {loss_barrier:.4f} (Mid: {midpoint_loss:.2f} vs End: {avg_endpoint_loss:.2f})")
                
            except Exception as e:
                print(f"FAILED: {e}")
                
            finally:
                # Cleanup
                if 'model_seed1' in locals(): del model_seed1
                if 'model_seed2' in locals(): del model_seed2
                if 'curve_model' in locals(): del curve_model
                gc.collect()             
                torch.cuda.empty_cache() 
            
            # Save constantly
            pd.DataFrame(results).to_csv(output_file, index=False)
            
    print(f"\nStudy Complete! Full dataset saved to {output_file}")

if __name__ == "__main__":
    run_full_study()