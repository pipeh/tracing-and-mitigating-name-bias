import json
import torch
import torch.nn as nn
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download



class JumpReLU(nn.Module):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = float(threshold)

    def forward(self, x):
        return torch.relu(x - self.threshold)


class LlamaScopeSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int, threshold: float):
        super().__init__()
        self.enc = nn.Linear(d_model, d_sae, bias=True)
        self.dec = nn.Linear(d_sae, d_model, bias=True)
        self.act = JumpReLU(threshold)

    def encode(self, x):
        return self.act(self.enc(x))


def llamascope_repo(site: str, expansion: str) -> str:
    mapping = {
        ("R", "8x"):  "OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x",
        ("A", "8x"):  "OpenMOSS-Team/Llama3_1-8B-Base-LXA-8x",
        ("M", "8x"):  "OpenMOSS-Team/Llama3_1-8B-Base-LXM-8x",
        ("TC","8x"):  "OpenMOSS-Team/Llama3_1-8B-Base-LXTC-8x",
        ("R", "32x"): "OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x",
        ("A", "32x"): "OpenMOSS-Team/Llama3_1-8B-Base-LXA-32x",
        ("M", "32x"): "OpenMOSS-Team/Llama3_1-8B-Base-LXM-32x",
        ("TC","32x"): "OpenMOSS-Team/Llama3_1-8B-Base-LXTC-32x",
    }
    return mapping[(site, expansion)]
    
def load_llamascope_sae(layer: int, site: str, expansion: str, device: torch.device, dtype: torch.dtype):
    repo = llamascope_repo(site, expansion)
    folder = f"Llama3_1-8B-Base-L{layer}{site}-{expansion}"

    hp_path = hf_hub_download(repo_id=repo, filename=f"{folder}/hyperparams.json")
    ckpt_path = hf_hub_download(repo_id=repo, filename=f"{folder}/checkpoints/final.safetensors")

    hp = json.load(open(hp_path, "r"))
    d_model = int(hp["d_model"])
    d_sae = int(hp["d_sae"])
    threshold = float(hp.get("jump_relu_threshold", 0.0))
    top_k = int(hp.get("top_k", 0))

    sd = load_file(ckpt_path, device=str(device))

    sae = LlamaScopeSAE(d_model=d_model, d_sae=d_sae, threshold=threshold).to(device).to(dtype)

    # Match checkpoint tensors by encoder/decoder dimensions.
    W_enc = W_dec = b_enc = b_dec = None
    for _, v in sd.items():
        if v.ndim == 2 and v.shape == (d_sae, d_model):
            W_enc = v
        elif v.ndim == 2 and v.shape == (d_model, d_sae):
            W_dec = v
        elif v.ndim == 1 and v.shape == (d_sae,):
            b_enc = v
        elif v.ndim == 1 and v.shape == (d_model,):
            b_dec = v

    if W_enc is None:
        raise RuntimeError("Could not find encoder weight by shape in checkpoint.")
    with torch.no_grad():
        sae.enc.weight.copy_(W_enc.to(dtype))
        if b_enc is not None:
            sae.enc.bias.copy_(b_enc.to(dtype))
        # Load decoder weights for feature interventions.
        if W_dec is not None:
            sae.dec.weight.copy_(W_dec.to(dtype))
        if b_dec is not None:
            sae.dec.bias.copy_(b_dec.to(dtype))

    sae.eval()
    return sae, hp, top_k
