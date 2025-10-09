# model/saab_model_run.py
"""
BiLSTM encoder + SVM classifier (+ optional SHAP) runner.

Train:
  python3 model/saab_model_run.py \
    --do_train \
    --train_data_file data/asterisk_ast_train.json \
    --output_dir runs/saab_bilstm_svm \
    --block_size 512 \
    --train_batch_size 32 \
    --epochs 6 \
    --lr 1e-3 \
    --hidden 128 \
    --emb_dim 128 \
    --use_shap  \
    --shap_nsamples 100 \
    --shap_background 50

Test (ADV):
  python3 model/saab_model_run.py \
    --do_test \
    --test_data_file data/asterisk_ast_test_ADV.json \
    --output_dir runs/saab_bilstm_svm \
    --block_size 512 \
    --eval_batch_size 64

Optional Top@k (if clean/ADV share idx):
  python3 model/saab_model_run.py \
    --do_test \
    --test_data_file data/asterisk_ast_test_ADV.json \
    --clean_test_file data/asterisk_ast_test.json \
    --output_dir runs/saab_bilstm_svm \
    --block_size 512 \
    --eval_batch_size 64
"""

import os, re, json, argparse, pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


try:
    import shap
except Exception:
    shap = None

# -------------------------
# IO helpers
# -------------------------
def load_json_any(path):
    s = open(path, "r", encoding="utf-8").read().strip()
    if not s: return []
    if s[0] == "[": return json.loads(s)
    out=[]
    for i,line in enumerate(s.splitlines(),1):
        line=line.strip()
        if not line: continue
        out.append(json.loads(line))
    return out

# -------------------------
# Tokenization / Vocab
# -------------------------
def simple_code_tokenize(text):
    # lightweight, robust splitter for code-like text
    return re.findall(r"[A-Za-z_]\w+|==|!=|<=|>=|&&|\|\||[^\s]", text or "")

class Vocab:
    def __init__(self):
        self.itos=["<pad>","<unk>"]
        self.stoi={t:i for i,t in enumerate(self.itos)}
    def add(self, tok):
        if tok not in self.stoi:
            self.stoi[tok]=len(self.itos)
            self.itos.append(tok)
    def build(self, texts, max_size=None):
        freq={}
        for t in texts:
            for tok in simple_code_tokenize(t):
                freq[tok]=freq.get(tok,0)+1
        for tok,_ in sorted(freq.items(), key=lambda x:(-x[1], x[0])):
            if max_size and len(self.itos)>=max_size: break
            self.add(tok)
    def save(self, path):
        with open(path,"w",encoding="utf-8") as f:
            for t in self.itos: f.write(t+"\n")
    @staticmethod
    def load(path):
        v=Vocab()
        v.itos=[l.strip() for l in open(path,encoding="utf-8")]
        v.stoi={t:i for i,t in enumerate(v.itos)}
        return v

def encode_batch(vocab, texts, max_len):
    ids=[]
    for t in texts:
        toks = simple_code_tokenize(t)[:max_len]
        arr  = [vocab.stoi.get(tok,1) for tok in toks]
        if len(arr)<max_len: arr += [0]*(max_len-len(arr))
        ids.append(arr)
    return np.array(ids, dtype=np.int64)

# -------------------------
# BiLSTM Encoder w/ Attention → feature vector
# -------------------------
class AttPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Linear(dim, 1, bias=False)
    def forward(self, H, mask):
        # H: [B,T,D], mask: [B,T] (1 for real, 0 for pad)
        scores = self.w(H).squeeze(-1)  # [B,T]
        scores = scores.masked_fill(mask==0, -1e9)
        att = torch.softmax(scores, dim=1)  # [B,T]
        pooled = torch.bmm(att.unsqueeze(1), H).squeeze(1)  # [B,D]
        return pooled, att

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden=128, pad_idx=0, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.pool = AttPool(2*hidden)
        self.out  = nn.Linear(2*hidden, 1)  # for training with BCE
        self.drop = nn.Dropout(dropout)
    def forward(self, ids):
        mask = (ids!=0).long()
        x = self.emb(ids)                  # [B,T,E]
        x,_ = self.lstm(x)                 # [B,T,2H]
        x = self.drop(x)
        pooled, att = self.pool(x, mask)   # [B,2H]
        logit = self.out(pooled).squeeze(-1)
        return logit, pooled, att
    def encode(self, ids, batch=64, device="cpu"):
        self.eval()
        feats=[]
        with torch.no_grad():
            for i in range(0,len(ids),batch):
                xb = torch.tensor(ids[i:i+batch], dtype=torch.long, device=device)
                _, pooled, _ = self.forward(xb)
                feats.append(pooled.detach().cpu().numpy())
        return np.concatenate(feats,axis=0) if feats else np.zeros((0, self.out.in_features),dtype=np.float32)

# -------------------------
# Metrics
# -------------------------
def accuracy(y_true, probs, thr=0.5):
    y = np.array(y_true, dtype=int)
    p = (np.array(probs)>=thr).astype(int)
    return float((y==p).sum())/max(1,len(y)), p

def file_adv_asr(y_true, preds):
    y = np.array(y_true, dtype=int)
    p = np.array(preds, dtype=int)
    adv_vuln = (y==1)
    return float(((adv_vuln)&(p==0)).sum())/max(1,adv_vuln.sum())

def compute_topk_from_clean(clean_probs, adv_pred_map, clean_idx, topk=(5,10,15,20)):
    """Compute Top@k attack success rate based on prediction changes.
    
    Args:
        clean_probs: Probability scores from clean samples
        adv_pred_map: Map of sample idx to adversarial prediction (0/1)
        clean_idx: Original indices of clean samples
        topk: Tuple of k values to compute
    
    Returns:
        Dictionary of Top@k success rates
    """
    pc = np.array(clean_probs)
    pc_pred = (pc >= 0.5).astype(int)
    
    # Sort ALL samples by confidence (absolute distance from decision boundary)
    confidence = np.abs(pc - 0.5)  # Higher value = more confident
    order = np.argsort(-confidence)  # Sort by confidence, highest first
    
    # Track successful attacks and matched samples
    success = []
    matched_samples = 0
    
    for j in order:
        cid = clean_idx[j]
        if cid not in adv_pred_map:
            continue
            
        # Get predictions for clean and adversarial versions
        clean_pred = pc_pred[j]
        adv_pred = adv_pred_map[cid]
        
        # Attack succeeds if prediction changed (0->1 or 1->0)
        changed = (clean_pred != adv_pred)
        success.append(1.0 if changed else 0.0)
        matched_samples += 1
    
    success = np.array(success, dtype=float)
    res = {}
    
    # Calculate Top@k metrics
    for k in topk:
        if matched_samples == 0:
            res[f"Top@{k}"] = 0.0
            continue
            
        # Only consider up to k or available samples
        n = min(k, matched_samples)
        # Success rate among top k most confident predictions
        res[f"Top@{k}"] = float(success[:n].sum()) / n
            
    return res

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    # Modes
    ap.add_argument("--do_train", action="store_true")
    ap.add_argument("--do_test", action="store_true")

    # Data
    ap.add_argument("--train_data_file", type=str, help="JSON/JSONL with func+target for training")
    ap.add_argument("--test_data_file",  type=str, help="JSON/JSONL for evaluation (adv or clean)")
    ap.add_argument("--clean_test_file", type=str, default=None, help="Optional clean file to compute Top@k via idx-matching")
    ap.add_argument("--text_field", type=str, default="func")
    ap.add_argument("--label_field", type=str, default="target")
    ap.add_argument("--block_size", type=int, default=512)

    # Model/Train
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train_batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=64)

    # Paths
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--load_predictions", type=str, help="Path to predictions.json file to evaluate")

    # SHAP options (optional)
    ap.add_argument("--use_shap", action="store_true", help="Run SHAP over SVM feature space")
    ap.add_argument("--shap_nsamples", type=int, default=50)
    ap.add_argument("--shap_background", type=int, default=50)

    # Device
    ap.add_argument("--device", type=str, default=None)  # auto if None

    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    # ----------------- TRAIN -----------------
    if args.do_train:
        if not args.train_data_file:
            raise SystemExit("--train_data_file is required for --do_train")

        train = load_json_any(args.train_data_file)
        Xtr=[r.get(args.text_field) for r in train]
        ytr=[int(r.get(args.label_field,0)) for r in train]

        # Build vocab & encode
        vocab = Vocab(); vocab.build(Xtr)
        vocab.save(outdir/"vocab.txt")
        Xtr_ids = encode_batch(vocab, Xtr, args.block_size)

        # Build encoder
        enc = BiLSTMEncoder(vocab_size=len(vocab.itos),
                            emb_dim=args.emb_dim, hidden=args.hidden,
                            dropout=args.dropout).to(device)

        # Train encoder with BCE on logits
        opt = optim.Adam(enc.parameters(), lr=args.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        enc.train()
        for ep in range(1, args.epochs+1):
            idx = np.arange(len(Xtr_ids))
            np.random.shuffle(idx)
            tot=0.0
            for i in range(0, len(idx), args.train_batch_size):
                sel = idx[i:i+args.train_batch_size]
                xb = torch.tensor(Xtr_ids[sel], dtype=torch.long, device=device)
                yb = torch.tensor(np.array(ytr)[sel], dtype=torch.float32, device=device)
                logit, _, _ = enc(xb)
                loss = loss_fn(logit, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach().cpu().item())
            print(f"epoch {ep}: loss={tot/max(1,(len(idx)//args.train_batch_size)):.4f}")

        # Save encoder weights
        torch.save(enc.state_dict(), outdir/"encoder.pt")

        # Extract features for SVM
        feats = enc.encode(Xtr_ids, batch=args.eval_batch_size, device=device)  # [N, 2*hidden]

        # SVM (with calibration → probabilities)
        base_svm = LinearSVC(class_weight="balanced", max_iter=5000)
        svm = CalibratedClassifierCV(base_svm, cv=3, method="sigmoid")
        clf = Pipeline([("scaler", StandardScaler(with_mean=False)), ("svm", svm)])
        clf.fit(feats, ytr)

        # Save SVM pipeline
        with open(outdir/"svm.pkl","wb") as f:
            pickle.dump(clf, f)

        # Optional: SHAP on feature space (fast-ish; still can be heavy)
        if args.use_shap:
            if shap is None:
                print("[WARN] shap not installed; skipping.")
            else:
                # Kernel SHAP on calibrated SVM probabilities; background subset
                bg_idx = np.random.choice(len(feats), size=min(args.shap_background, len(feats)), replace=False)
                background = feats[bg_idx]
                explainer = shap.KernelExplainer(lambda z: clf.predict_proba(z)[:,1], background)
                # explain a small set (e.g., 32)
                nsamp = min(32, len(feats))
                tgt_idx = np.random.choice(len(feats), size=nsamp, replace=False)
                shap_vals = explainer.shap_values(feats[tgt_idx], nsamples=args.shap_nsamples)
                np.save(outdir/"shap_values.npy", shap_vals)
                np.save(outdir/"shap_indices.npy", tgt_idx)
                print(f"[SHAP] saved shap_values.npy / shap_indices.npy (nsamples={args.shap_nsamples})")

        print("Training complete. Artifacts saved to:", outdir)

    # ----------------- TEST -----------------
    if args.do_test:
        if not args.test_data_file:
            raise SystemExit("--test_data_file is required for --do_test")

        # Load artifacts
        vocab = Vocab.load(outdir/"vocab.txt")
        enc = BiLSTMEncoder(vocab_size=len(vocab.itos),
                            emb_dim=args.emb_dim, hidden=args.hidden,
                            dropout=args.dropout).to(device)
        # tolerate shape diffs if hyperparams changed: user should match
        enc.load_state_dict(torch.load(outdir/"encoder.pt", map_location=device))
        enc.eval()
        clf = pickle.load(open(outdir/"svm.pkl","rb"))

        # Load ADV (or any) test
        test = load_json_any(args.test_data_file)
        X = [r.get(args.text_field) for r in test]
        y = [int(r.get(args.label_field,0)) if (args.label_field in r) else None for r in test]
        idxs = [r.get("idx",i) for i,r in enumerate(test)]

        X_ids = encode_batch(vocab, X, args.block_size)
        feats = enc.encode(X_ids, batch=args.eval_batch_size, device=device)
        probs = clf.predict_proba(feats)[:,1]
        preds = (probs>=0.5).astype(int)

        # Metrics
        if any(t is not None for t in y):
            y_filled = [int(t) if t is not None else 0 for t in y]
            acc, _ = accuracy(y_filled, probs)
        else:
            acc = None
        fad = file_adv_asr([int(t) if t is not None else 0 for t in y], preds)

        print("ADV/Test ACC (if labels present):", "N/A" if acc is None else f"{acc:.4f}")
        print(f"File ADV ASR (label=1 -> pred 0): {fad:.4f}")

        # Load or generate predictions
        if args.load_predictions:
            with open(args.load_predictions, "r") as f:
                out_preds = json.load(f)
                # Extract predictions from loaded file - use defense predictions directly
                if isinstance(out_preds, dict) and "detailed_results" in out_preds:
                    # Handle shap_defense.py format
                    probs = np.array([r.get("final_prediction", {}).get("probability", 0.5) 
                                    for r in out_preds.get("detailed_results", [])])
                    preds = np.array([r.get("final_prediction", {}).get("prediction", 0) 
                                    for r in out_preds.get("detailed_results", [])])
                else:
                    # Handle original format
                    probs = np.array([p["prob"] for p in out_preds])
                    preds = np.array([p["pred"] for p in out_preds])
        else:
            # Generate new predictions
            out_preds=[]
            for i,(ix,p,pr,rec) in enumerate(zip(idxs, probs.tolist(), preds.tolist(), test)):
                out_preds.append({"idx": ix, "prob": float(p), "pred": int(pr), "original": rec})
            json.dump(out_preds, open(outdir/"predictions.json","w"), indent=2)

        # Save summary metrics
        summary = {
            "accuracy": float(acc) if acc is not None else None,
            "adv_asr": float(fad)
        }
        json.dump(summary, open(outdir/"summary.json","w"), indent=2)
        print("Saved predictions.json and summary.json to", outdir)

        # Optional Top@k via clean/adv idx alignment
        topk = {}
        if args.clean_test_file:
            clean = load_json_any(args.clean_test_file)
            Xc = [r.get(args.text_field) for r in clean]
            idxc= [r.get("idx",i) for i,r in enumerate(clean)]
            Xc_ids = encode_batch(vocab, Xc, args.block_size)
            feats_c = enc.encode(Xc_ids, batch=args.eval_batch_size, device=device)
            pc = clf.predict_proba(feats_c)[:,1]
            adv_map = {ix:pr for ix,pr in zip(idxs, preds.tolist())}
            topk = compute_topk_from_clean(pc, adv_map, idxc, topk=(5,10,15,20))
            print("Top@k (via clean vs ADV idx alignment):")
            for k,v in topk.items(): print(f"  {k} = {v:.4f}")

        json.dump({
            "acc": None if acc is None else float(acc),
            "file_adv_asr": float(fad),
            "n": len(X),
            "topk": topk
        }, open(outdir/"summary.json","w"), indent=2)

        print("Saved predictions.json and summary.json to", outdir)

if __name__=="__main__":
    main()
