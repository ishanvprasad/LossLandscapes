import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# Isolate this VRAM-heavy pipeline to specific GPUs on your multi-GPU server
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,7"

# Import from your modular files
from utils.curvature import compute_hessian_metrics
from utils.connectivity import train_bezier_curve
from utils.similarity import extract_and_compare

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_pythia_checkpoint(step, size="70m"):
    model_name = f"EleutherAI/pythia-{size}"
    revision = f"step{step}" 
    
    print(f"Loading {model_name} at {revision}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",
        revision=revision,
        torch_dtype=torch.float16)
    return model, tokenizer

def load_evaluation_data(tokenizer, benchmark="arc_challenge", batch_size=8, num_samples=200):
    """
    Loads text from the exact EleutherAI Pythia zero-shot and few-shot benchmarks.
    Supported: "arc_challenge", "arc_easy", "blimp", "lambada_openai", "logiqa", 
               "mmlu", "piqa", "sciq", "wikitext", "winogrande", "wsc"
    """
    print(f"Fetching {num_samples} samples from {benchmark}...")
    
    # Map benchmark names to their Hub paths, configs, and default evaluation splits
    benchmark_configs = {
        "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
        "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test"),
        "blimp": ("blimp", "anaphor_agreement", "train"), # One of the many BLiMP linguistic tasks
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
    
    # Each dataset uses wildly different column names for its core text
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
            # Wikitext contains empty newlines that we should skip
            if item["text"].strip(): 
                texts.append(item["text"])
                
        if len(texts) >= num_samples:
            break
            
    # Tokenize the batch (PyHessian needs standardized tensor shapes)
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    class BenchmarkDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            # Labels are required to calculate CausalLM cross-entropy loss
            item['labels'] = item['input_ids'].clone()
            return item
        def __len__(self):
            return len(self.encodings.input_ids)
            
    hf_dataset = BenchmarkDataset(encoded)
    return DataLoader(hf_dataset, batch_size=batch_size, shuffle=True)

def run_landscape_evaluation():
    print(f"Executing pipeline on device: {device}")
    
    step_early = 10000
    step_late = 143000 
    
    # 1. Load Models
    model_early, tokenizer = load_pythia_checkpoint(step_early, size="70m")
    model_late, _ = load_pythia_checkpoint(step_late, size="70m")
    
    # 2. Load Real Benchmark Data
    # You can loop this across ["arc_challenge", "hellaswag", "mmlu", "truthfulqa_mc1"]
    dataloader = load_evaluation_data(tokenizer, benchmark="arc_challenge", num_samples=256)
    
    # Grab a single batch formatted for the static evaluations (Hessian/CKA)
    batch = next(iter(dataloader))
    static_inputs = {k: v.to(device) for k, v in batch.items()}
    
    # --- A. CURVATURE (Local Structure) ---
    print("\n[1/3] Computing early checkpoint curvature (PyHessian)...")
    # pass the raw batch to the helper (we no longer need the list wrapper)
    eig_early, trace_early = compute_hessian_metrics(model_early, static_inputs, device)
    
    print("[1/3] Computing late checkpoint curvature (PyHessian)...")
    eig_late, trace_late = compute_hessian_metrics(model_late, static_inputs, device)
    
    # --- B. SIMILARITY (Phase IV-B check) ---
    print("\n[2/3] Computing CKA Similarity...")
    cka_score = extract_and_compare(model_early, model_late, static_inputs, device)
    
    # --- C. CONNECTIVITY (Global Map) ---
    print("\n[3/3] Training Bezier curve for Mode Connectivity (Active Loop)...")
    # This active loop leverages the full dataloader rather than a single static batch
    curve_model = train_bezier_curve(model_early, model_late, dataloader, epochs=3, device=device)
    
    with torch.no_grad():
        curve_model.eval()
        midpoint_loss = curve_model(0.5, static_inputs).item()
        early_loss = model_early(**static_inputs).loss.item()
        late_loss = model_late(**static_inputs).loss.item()

    # --- SUMMARY ---
    print("\n==============================")
    print("       LANDSCAPE RESULTS      ")
    print("==============================")
    print(f"Early Checkpoint (Step {step_early}): Trace: {trace_early:.4f}")
    print(f"Late Checkpoint (Step {step_late}):  Trace: {trace_late:.4f}")
    print(f"CKA Representation Similarity: {cka_score:.4f}")
    print(f"Curve Midpoint Loss (t=0.5):   {midpoint_loss:.4f} (Endpoints: ~{early_loss:.4f})")

if __name__ == "__main__":
    run_landscape_evaluation()