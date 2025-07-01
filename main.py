import argparse
import torch
import os
import time
from datetime import datetime
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
import json
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import random
from model import DGMG

def generate_cycles_dataset(v_min: int, v_max: int, n_samples: int) -> list:
    """Gera o dataset de ciclos em tempo de execução."""
    dataset = []
    for _ in range(n_samples):
        size = random.randint(v_min, v_max)
        seq = []
        for i in range(size):
            seq.append(0)  # add node
            if i != 0:
                seq.append(0)  # add edge
                seq.append(i - 1)  # connect to previous node
            if i == size - 1:
                seq.append(0)  # add edge
                seq.append(0)  # close cycle (connect to first node)
            seq.append(1)  # stop adding edges
        seq.append(1)  # stop adding nodes
        dataset.append(seq)
    return dataset


def setup_logging(log_dir: str):
    """Configura diretório de logs e arquivo de métricas."""
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    return open(os.path.join(log_dir, "metrics.json"), "w")


def train(model, dataset, opts):
    """Rotina de treinamento do modelo."""
    model.train()
    optimizer = Adam(model.parameters(), lr=opts.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True)
    
    best_loss = float('inf')
    metrics = {"train_loss": [], "train_prob": []}

    for epoch in range(opts.nepochs):
        epoch_loss = 0
        epoch_prob = 0
        start_time = time.time()

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{opts.nepochs}"):
            actions = batch[0]  # batch_size=1 por padrão
            optimizer.zero_grad()
            
            log_prob = model(actions=actions)
            loss = -log_prob
            loss.backward()

            if opts.clip_grad:
                clip_grad_norm_(model.parameters(), opts.clip_bound)

            optimizer.step()

            epoch_loss += loss.item()
            epoch_prob += log_prob.exp().item()

        avg_loss = epoch_loss / len(dataloader)
        avg_prob = epoch_prob / len(dataloader)
        scheduler.step(avg_loss)
        
        metrics["train_loss"].append(avg_loss)
        metrics["train_prob"].append(avg_prob)
        
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}, Prob: {avg_prob:.4f}, Time: {time.time()-start_time:.1f}s")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(opts.log_dir, "checkpoints", "best_model.pth"))
        
        # Periodic saving
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(opts.log_dir, "checkpoints", f"model_epoch{epoch+1}.pth"))

    torch.save(model.state_dict(), os.path.join(opts.log_dir, "model_final.pth"))
    return metrics


def evaluate(model, num_samples: int, v_min: int, v_max: int, save_dir: str):
    """Avalia o modelo gerando amostras e calculando métricas."""
    model.eval()
    results = {
        "sizes": [],
        "valid_size": 0,
        "is_cycle": 0,
        "is_connected": 0,
        "avg_degree": []
    }

    os.makedirs(save_dir, exist_ok=True)
    
    for i in tqdm(range(num_samples), desc="Generating samples"):
        with torch.no_grad():
            data = model()

        g_nx = to_networkx(data, to_undirected=True)
        num_nodes = g_nx.number_of_nodes()
        degrees = [d for n, d in g_nx.degree()]
        
        results["sizes"].append(num_nodes)
        results["avg_degree"].append(sum(degrees)/num_nodes if num_nodes > 0 else 0)
        
        # Validação do tamanho
        is_valid_size = v_min <= num_nodes <= v_max
        results["valid_size"] += int(is_valid_size)
        
        # Validação de conectividade
        is_connected = nx.is_connected(g_nx) if num_nodes > 0 else False
        results["is_connected"] += int(is_connected)
        
        # Validação de ciclo
        is_cycle = is_connected and all(d == 2 for d in degrees)
        results["is_cycle"] += int(is_cycle)

        # Salva as primeiras amostras para visualização
        if i < 8:
            plt.figure(figsize=(6,6))
            nx.draw_circular(g_nx, with_labels=True, node_color='skyblue', node_size=500)
            plt.title(f"Sample {i+1}\nNodes: {num_nodes}, Cycle: {is_cycle}")
            plt.savefig(os.path.join(save_dir, f"sample_{i+1}.png"), bbox_inches='tight')
            plt.close()

    # Calcula métricas agregadas
    results["avg_size"] = sum(results["sizes"]) / num_samples
    results["valid_size_ratio"] = results["valid_size"] / num_samples
    results["cycle_ratio"] = results["is_cycle"] / num_samples
    results["connected_ratio"] = results["is_connected"] / num_samples
    results["avg_degree"] = sum(results["avg_degree"]) / num_samples
    
    print("\nEvaluation Results:")
    print(f"Average size: {results['avg_size']:.2f} (min: {min(results['sizes'])}, max: {max(results['sizes'])})")
    print(f"Valid size ratio: {results['valid_size_ratio']:.2f}")
    print(f"Cycle ratio: {results['cycle_ratio']:.2f}")
    print(f"Connected ratio: {results['connected_ratio']:.2f}")
    print(f"Average degree: {results['avg_degree']:.2f}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DGMG for Cycle Generation")
    parser.add_argument("--log-dir", type=str, default="results", help="Directory to save logs and models")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--nepochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--clip-grad", action="store_true", help="Enable gradient clipping")
    parser.add_argument("--clip-bound", type=float, default=0.25, help="Gradient clipping bound")
    parser.add_argument("--v-min", type=int, default=3, help="Minimum cycle size")
    parser.add_argument("--v-max", type=int, default=20, help="Maximum cycle size")
    parser.add_argument("--n", type=int, default=1000, help="Number of training samples to generate")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    opts = parser.parse_args()

    # Configuração inicial
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    opts.log_dir = os.path.join(opts.log_dir, timestamp)
    metrics_file = setup_logging(opts.log_dir)
    
    # Salva as configurações
    with open(os.path.join(opts.log_dir, "config.json"), "w") as f:
        json.dump(vars(opts), f, indent=2)

    # Gera dataset em memória
    dataset = generate_cycles_dataset(opts.v_min, opts.v_max, opts.n)
    model = DGMG(v_max=opts.v_max)
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using CUDA")

    # Treinamento e avaliação
    train_metrics = train(model, dataset, opts)
    eval_results = evaluate(model, opts.num_samples, opts.v_min, opts.v_max, 
                          os.path.join(opts.log_dir, "samples"))

    # Salva métricas
    all_metrics = {**vars(opts), **train_metrics, **eval_results}
    json.dump(all_metrics, metrics_file, indent=2)
    metrics_file.close()