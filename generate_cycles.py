import pickle
import random
import argparse
from typing import List

def get_decision_sequence(size: int) -> List[int]:
    """
    Gera a sequência de decisões para criar um ciclo com N nós.
    
    Args:
        size: Número de nós no ciclo (deve ser >= 3 para ser um ciclo válido)
    
    Returns:
        Lista de ações conforme especificação DGMG:
        0 = add node
        1 = stop adding nodes
        0 = add edge
        1 = stop adding edges
        k = índice do nó de destino (0-based)
    """
    if size < 3:
        raise ValueError("Um ciclo precisa de pelo menos 3 nós")
    
    sequence = []
    # Fase de adição de nós
    for i in range(size):
        sequence.append(0)  # add node
        
        # Fase de adição de arestas para cada nó
        if i > 0:  # Conecta ao nó anterior
            sequence.append(0)  # add edge
            sequence.append(i - 1)  # destination
            
            # No último nó, conecta de volta ao primeiro para fechar o ciclo
            if i == size - 1:
                sequence.append(0)  # add edge
                sequence.append(0)  # connect to first node
        
        sequence.append(1)  # stop adding edges
    
    sequence.append(1)  # stop adding nodes
    return sequence

def generate_dataset(v_min: int, v_max: int, n_samples: int, output_file: str) -> None:
    """
    Gera um dataset de sequências de construção de ciclos.
    
    Args:
        v_min: Tamanho mínimo do ciclo
        v_max: Tamanho máximo do ciclo
        n_samples: Número de amostras
        output_file: Arquivo de saída
    """
    dataset = []
    for _ in range(n_samples):
        size = random.randint(v_min, v_max)
        try:
            seq = get_decision_sequence(size)
            dataset.append(seq)
        except ValueError as e:
            print(f"Warning: {e}. Pulando amostra.")
    
    with open(output_file, "wb") as f:
        pickle.dump(dataset, f)
    
    print(f"Dataset salvo em {output_file} com {len(dataset)} amostras válidas.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerador de dataset para grafos cíclicos")
    parser.add_argument("--v-min", type=int, default=3, help="Tamanho mínimo do ciclo (>=3)")
    parser.add_argument("--v-max", type=int, default=20, help="Tamanho máximo do ciclo")
    parser.add_argument("--n", type=int, default=4000, help="Número de amostras")
    parser.add_argument("--output", type=str, default="cycles.p", help="Arquivo de saída")
    args = parser.parse_args()

    if args.v_min < 3:
        raise ValueError("--v-min deve ser >= 3 para formar ciclos válidos")
    
    generate_dataset(args.v_min, args.v_max, args.n, args.output)