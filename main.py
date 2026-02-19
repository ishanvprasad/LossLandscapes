import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# Isolate this VRAM-heavy pipeline to specific GPUs on your multi-GPU server
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,7"

from utils.curvature import compute_hessian_metrics
from utils.connectivity import train_bezier_curve
from utils.similarity import extract_and_compare

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_pythia_checkpoint(step, size="70m-seed1"):
    model_name = f"EleutherAI/pythia-{size}"
    revision = f"step{step}" 
    
    print(f"Loading {model_name} at {revision}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # FIX 4: Loaded in float32 to prevent PyHessian NaN outputs
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)
    if device is not None:
        model.to(device)
    return model, tokenizer

def load_evaluation_data(tokenizer, benchmark="arc_challenge", batch_size=8, num_samples=200):
    """
    Loads text from the exact EleutherAI Pythia zero-shot and few-shot benchmarks.
    """
    print(f"Fetching {num_samples} samples from {benchmark}...")
    
    benchmark_configs = {
        "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
        "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test"),
        "blimp": ("blimp", "anaphor_agreement", "train"), 
        "lambada_openai": ("EleutherAI/lambada_openai", "default", "test"),
        "logiqa": ("lucasmccabe/logiqa", "default", "test"),
        "mmlu": ("cais/mmlu", "all", "test"),
        "piqa": ("piqa", "default", "test"),
        "sciq": ("sciq", "default", "test"),
        "wikitext": ("wikitext", "wikitext-2-raw-v1", "test"),
        "winogrande": ("winogrande", "winogrande_xl", "validation"),
        "wsc": ("super_glue", "wsc", "validation")
    }
    
    if benchmark not in benchmark_configs:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
        
    path, config, split = benchmark_configs[benchmark]
    dataset = load_dataset(path, config, split=split)
    
    texts = []
    for item in dataset:
        if benchmark in ["arc_challenge", "arc_easy", "mmlu", "sciq"]:
            texts.append(item["question"])
        elif benchmark == "blimp":
            texts.append(item["sentence_good"])
        elif benchmark == "logiqa":
            texts.append(item["context"] + " " + item["question"])
        elif benchmark == "piqa":
            texts.append(item["goal"])
        elif benchmark == "winogrande":
            texts.append(item["sentence"])
        elif benchmark in ["lambada_openai", "wikitext", "wsc"]:
            if item["text"].strip(): 
                texts.append(item["text"])
                
        if len(texts) >= num_samples:
            break
            
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

def run_landscape_evaluation():
    print(f"Executing pipeline on device: {device}")
    
    # FIX 1: Evaluate different seeds at the exact same step 
    eval_step = 143000 
    
    # 1. Load Models
    model_seed1, tokenizer = load_pythia_checkpoint(eval_step, size="70m-seed1")
    model_seed2, _ = load_pythia_checkpoint(eval_step, size="70m-seed2")
    
    # 2. Load Real Benchmark Data
    dataloader = load_evaluation_data(tokenizer, benchmark="arc_challenge", num_samples=256)
    
    batch = next(iter(dataloader))
    static_inputs = {k: v.to(device) for k, v in batch.items()}
    
    # --- A. CURVATURE (Local Structure) ---
    print(f"\n[1/3] Computing Seed 1 curvature (PyHessian) at Step {eval_step}...")
    eig_1, trace_1 = compute_hessian_metrics(model_seed1, static_inputs, device)
    
    print(f"[1/3] Computing Seed 2 curvature (PyHessian) at Step {eval_step}...")
    eig_2, trace_2 = compute_hessian_metrics(model_seed2, static_inputs, device)
    
    # --- B. SIMILARITY (Phase IV-B check) ---
    print("\n[2/3] Computing CKA Similarity between seeds...")
    cka_score = extract_and_compare(model_seed1, model_seed2, static_inputs, device)
    
    # --- C. CONNECTIVITY (Global Map) ---
    print("\n[3/3] Training Bezier curve for Mode Connectivity (Active Loop)...")
    curve_model = train_bezier_curve(model_seed1, model_seed2, dataloader, epochs=3, device=device)
    
    with torch.no_grad():
        curve_model.eval()
        midpoint_loss = curve_model(0.5, static_inputs).item()
        seed1_loss = model_seed1(**static_inputs).loss.item()
        seed2_loss = model_seed2(**static_inputs).loss.item()

    # --- SUMMARY ---
    print("\n==============================")
    print("       LANDSCAPE RESULTS      ")
    print("==============================")
    print(f"Step {eval_step} - Seed 1 Trace: {trace_1:.4f}")
    print(f"Step {eval_step} - Seed 2 Trace: {trace_2:.4f}")
    print(f"Cross-Seed CKA Similarity: {cka_score:.4f}")
    
    avg_endpoint_loss = (seed1_loss + seed2_loss) / 2
    barrier = midpoint_loss - avg_endpoint_loss
    print(f"Curve Midpoint Loss (t=0.5):   {midpoint_loss:.4f} (Endpoints: ~{avg_endpoint_loss:.4f})")
    print(f"Loss Barrier: {barrier:.4f}")

if __name__ == "__main__":
    run_landscape_evaluation()