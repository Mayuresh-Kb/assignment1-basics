import torch
import numpy as np
import time
import json
import argparse
from pathlib import Path
from cs336_basics.transformer import Transformer_lm
from cs336_basics.train import cross_entropy, AdamW, learning_rate_schedule, gradient_clipping, get_batch, save_checkpoint, load_checkpoint

def train(train_data_path, val_data_path, vocab_size, context_length, d_model, d_ff, num_layers, num_heads, rope_theta, learning_rate, beta1, beta2, epsilon, weight_decay, batch_size, max_iterations, max_grad_norm, min_learning_rate, warmup_iterations, cosine_iterations, log_interval, eval_interval, eval_batches, checkpoint_dir, checkpoint_interval, device, metrics_path, resume_checkpoint=None):
    train_data = np.load(train_data_path, mmap_mode="r")
    val_data = np.load(val_data_path, mmap_mode="r")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    model = Transformer_lm(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay)
    
    start_iteration = 0
    if resume_checkpoint is not None:
        start_iteration = load_checkpoint(resume_checkpoint, model, optimizer)
        
    previous_elapsed_time = 0.0
    if resume_checkpoint is None: 
        metrics = []      
    else:
        if metrics_path.exists():
            with open(metrics_path, mode="r", encoding="utf-8") as f:
                metrics = json.load(f)
                metrics = [metric for metric in metrics if metric["iteration"] <= start_iteration]
                if metrics:
                    previous_elapsed_time = metrics[-1]["elapsed_time"]
        else:
            metrics = []
    

    start_time = time.perf_counter()
    for iteration in range(start_iteration, max_iterations):
        inputs, targets = get_batch(train_data, batch_size, context_length, device=device)
        logits = model(inputs)

        train_loss = cross_entropy(logits, targets)
        optimizer.zero_grad()
        train_loss.backward()
        gradient_clipping(model.parameters(), max_grad_norm)
        current_lr = learning_rate_schedule(iteration, learning_rate, min_learning_rate, warmup_iterations, cosine_iterations)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.step()

        completed_steps = iteration + 1
        
        if(completed_steps) % checkpoint_interval == 0:
            checkpoint_file = checkpoint_dir / f"checkpoint_{completed_steps}.pt"
            save_checkpoint(model, optimizer, completed_steps, checkpoint_file)

        if completed_steps % log_interval == 0:
            print("iteration: ", completed_steps)
            print("training loss: ", train_loss.item())
            print("current learning rate: ", current_lr)

        if completed_steps % eval_interval == 0:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for _ in range(eval_batches):
                    inputs, targets = get_batch(val_data, batch_size, context_length, device=device)
                    logits = model(inputs)
                    batch_val_loss = cross_entropy(logits, targets)
                    val_losses.append(batch_val_loss.item())
            
            val_loss = sum(val_losses) / len(val_losses)
            elapsed_time = (time.perf_counter() - start_time) + previous_elapsed_time
            
            print("validation loss: ", val_loss)
            print("elapsed time: ", elapsed_time)
            metrics.append({"iteration": completed_steps, "train_loss": train_loss.item(), "val_loss": val_loss, "learning_rate": current_lr, "elapsed_time": elapsed_time})
            with open(metrics_path, mode="w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)


            model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--val-data-path", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-iterations", type=int, default=100)
    parser.add_argument("--cosine-iterations", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--metrics-path", type=str, default="metrics.json")
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    args = parser.parse_args()
    train(**vars(args))

