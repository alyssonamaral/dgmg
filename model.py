import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical
from torch_geometric.data import Data
from typing import Optional, Tuple, List


def bernoulli_action_log_prob(logit: torch.Tensor, action: int) -> torch.Tensor:
    """Calcula o log prob de uma ação binária usando a distribuição de Bernoulli.
    
    Args:
        logit: Valor logit da distribuição
        action: Ação tomada (0 ou 1)
    
    Returns:
        Log probabilidade da ação
    """
    return F.logsigmoid(logit if action == 1 else -logit)


class GraphEmbed(nn.Module):
    """Módulo para calcular a representação do grafo completo."""
    def __init__(self, node_hidden_size: int):
        super().__init__()
        self.node_hidden_size = node_hidden_size
        self.graph_hidden_size = 2 * node_hidden_size
        
        # Gate para ponderar a contribuição de cada nó
        self.node_gating = nn.Sequential(
            nn.Linear(node_hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Transformação dos nós para o espaço do grafo
        self.node_to_graph = nn.Linear(node_hidden_size, self.graph_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calcula a representação do grafo.
        
        Args:
            x: Tensor de nós [num_nodes, node_hidden_size]
        
        Returns:
            Representação do grafo [1, graph_hidden_size]
        """
        # Aplica gate e transformação
        gated = self.node_gating(x) * self.node_to_graph(x)
        
        # Soma as contribuições ponderadas dos nós
        return gated.sum(dim=0, keepdim=True)


class GraphProp(nn.Module):
    """Módulo para propagação de mensagens entre nós."""
    def __init__(self, num_rounds: int, node_hidden_size: int):
        super().__init__()
        self.num_rounds = num_rounds
        self.node_hidden_size = node_hidden_size
        
        # Camadas para cálculo de mensagens
        self.msg_layers = nn.ModuleList([
            nn.Linear(2 * node_hidden_size + 1, node_hidden_size) 
            for _ in range(num_rounds)
        ])
        
        # GRU para atualização dos estados ocultos
        self.upd_layers = nn.ModuleList([
            nn.GRUCell(2 * node_hidden_size, node_hidden_size)
            for _ in range(num_rounds)
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
               edge_attr: torch.Tensor) -> torch.Tensor:
        """Propaga mensagens através do grafo.
        
        Args:
            x: Estados ocultos dos nós [num_nodes, node_hidden_size]
            edge_index: Índices das arestas [2, num_edges]
            edge_attr: Atributos das arestas [num_edges, edge_feat_dim]
        
        Returns:
            Estados ocultos atualizados [num_nodes, node_hidden_size]
        """
        for t in range(self.num_rounds):
            row, col = edge_index  # Índices de origem e destino
            
            # Obter estados dos nós de origem e destino
            src, dest = x[row], x[col]
            
            # Calcular mensagens
            m_input = torch.cat([dest, src, edge_attr], dim=1)
            messages = self.msg_layers[t](m_input)
            
            # Agregar mensagens por nó de destino
            agg = torch.zeros_like(x)
            agg.index_add_(0, row, messages)
            
            # Atualizar estados ocultos
            x = self.upd_layers[t](agg, x)
        
        return x


class AddNode(nn.Module):
    """Módulo para decisão de adicionar novo nó."""
    def __init__(self, graph_embed: GraphEmbed, node_hidden_size: int):
        super().__init__()
        self.log_probs: List[torch.Tensor] = []  # Tipo explícito para mypy
        self.graph_embed = graph_embed
        self.node_hidden_size = node_hidden_size
        
        # Classificador para decisão de adicionar nó
        self.add_node = nn.Linear(graph_embed.graph_hidden_size, 1)
        
        # Embedding para tipo de nó (poderia ser estendido para múltiplos tipos)
        self.node_type_embed = nn.Embedding(1, node_hidden_size)
        
        # Inicialização do estado oculto do novo nó
        self.initialize_hv = nn.Linear(
            node_hidden_size + graph_embed.graph_hidden_size, 
            node_hidden_size
        )
        
        # Armazena log probs para cálculo da loss
        self.log_probs: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor, action: Optional[int] = None) -> Tuple[bool, torch.Tensor]:
        """Toma decisão de adicionar nó e atualiza o grafo.
        
        Args:
            x: Estados ocultos dos nós existentes [num_nodes, node_hidden_size]
            action: Ação forçada (usada durante treino)
        
        Returns:
            stop: Se deve parar de adicionar nós
            x: Estados ocultos atualizados
        """
        graph_embed = self.graph_embed(x)
        logit = self.add_node(graph_embed)
        prob = torch.sigmoid(logit)

        # Amostra ação durante inferência
        if not self.training:
            action = Bernoulli(prob).sample().item()

        stop = bool(action == 1)
        
        # Adiciona novo nó se não parar
        if not stop:
            node_type = 0  # Assumindo único tipo de nó
            h_node = self.initialize_hv(
                torch.cat([
                    self.node_type_embed(torch.tensor([node_type], device=x.device)), 
                    graph_embed
                ], dim=1)
            )
            x = torch.cat([x, h_node], dim=0)

        # Armazena log prob para cálculo da loss
        if self.training and action is not None:
            self.log_probs.append(bernoulli_action_log_prob(logit, action))

        return stop, x


class AddEdge(nn.Module):
    """Módulo para decisão de adicionar nova aresta."""
    def __init__(self, graph_embed: GraphEmbed, node_hidden_size: int):
        super().__init__()
        self.log_probs: List[torch.Tensor] = []  # Tipo explícito para mypy

        self.graph_embed = graph_embed
        self.node_hidden_size = node_hidden_size
        
        # Classificador para decisão de adicionar aresta
        self.add_edge = nn.Linear(
            graph_embed.graph_hidden_size + node_hidden_size, 
            1
        )
        self.log_probs: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor, action: Optional[int] = None) -> bool:
        """Toma decisão de adicionar aresta.
        
        Args:
            x: Estados ocultos dos nós [num_nodes, node_hidden_size]
            action: Ação forçada (usada durante treino)
        
        Returns:
            to_add_edge: Se deve adicionar aresta
        """
        graph_embed = self.graph_embed(x)
        src = x[-1].unsqueeze(0)  # Último nó adicionado
        
        logit = self.add_edge(torch.cat([graph_embed, src], dim=1))
        prob = torch.sigmoid(logit)

        if not self.training:
            action = Bernoulli(prob).sample().item()

        to_add_edge = bool(action == 0)
        
        if self.training and action is not None:
            self.log_probs.append(bernoulli_action_log_prob(logit, action))

        return to_add_edge


class ChooseDestAndUpdate(nn.Module):
    """Módulo para escolher destino da aresta e atualizar o grafo."""
    def __init__(self, graph_prop: GraphProp, node_hidden_size: int):
        super().__init__()
        self.log_probs: List[torch.Tensor] = []  # Tipo explícito para mypy

        self.graph_prop = graph_prop
        self.node_hidden_size = node_hidden_size
        
        # Classificador para escolher nó de destino
        self.choose_dest = nn.Linear(2 * node_hidden_size, 1)
        self.log_probs: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
               edge_attr: torch.Tensor, dest: Optional[int] = None
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Escolhe destino para nova aresta e atualiza o grafo.
        
        Args:
            x: Estados ocultos dos nós [num_nodes, node_hidden_size]
            edge_index: Índices das arestas existentes [2, num_edges]
            edge_attr: Atributos das arestas existentes [num_edges, edge_feat_dim]
            dest: Destino forçado (usado durante treino)
        
        Returns:
            x: Estados ocultos atualizados
            edge_index: Índices das arestas atualizados
            edge_attr: Atributos das arestas atualizados
        """
        src_idx = x.size(0) - 1  # Índice do último nó adicionado
        
        # Calcula scores para todos os nós possíveis
        src = x[src_idx].expand(src_idx, -1)  # [N-1, hidden_size]
        dests = x[:src_idx]  # [N-1, hidden_size]
        
        scores = self.choose_dest(torch.cat([dests, src], dim=1)).view(1, -1)
        probs = F.softmax(scores, dim=1)

        # Amostra destino durante inferência
        if not self.training:
            dest = Categorical(probs).sample().item()

        # Adiciona arestas bidirecionais
        new_edges = torch.tensor(
            [[src_idx, dest], [dest, src_idx]], 
            dtype=torch.long,
            device=x.device
        )
        edge_index = torch.cat([edge_index, new_edges], dim=1)
        
        # Adiciona atributos das novas arestas (assumindo tipo único)
        edge_feat = torch.ones(2, 1, device=x.device)
        edge_attr = torch.cat([edge_attr, edge_feat], dim=0)
        
        # Propaga mensagens com o novo grafo
        x = self.graph_prop(x, edge_index, edge_attr)
        
        # Armazena log prob para cálculo da loss
        if self.training and dest is not None:
            self.log_probs.append(F.log_softmax(scores, dim=1)[0, dest:dest+1])

        return x, edge_index, edge_attr


class DGMG(nn.Module):
    """Modelo DGMG completo para geração de grafos."""
    def __init__(self, v_max: int, node_hidden_size: int = 16, num_prop_rounds: int = 2):
        super().__init__()
        self.v_max = v_max  # Número máximo de nós
        self.node_hidden_size = node_hidden_size
        
        # Módulos componentes
        self.graph_embed = GraphEmbed(node_hidden_size)
        self.graph_prop = GraphProp(num_prop_rounds, node_hidden_size)
        self.add_node_agent = AddNode(self.graph_embed, node_hidden_size)
        self.add_edge_agent = AddEdge(self.graph_embed, node_hidden_size)
        self.choose_dest_agent = ChooseDestAndUpdate(self.graph_prop, node_hidden_size)
    
    def device(self):
        return next(self.parameters()).device

    def forward(self, actions: Optional[List[int]] = None) -> torch.Tensor:
        """Executa o modelo no modo treino (com ações) ou geração (sem ações).
        
        Args:
            actions: Sequência de ações forçadas (para treino)
        
        Returns:
            No treino: Log prob da sequência de ações
            Na geração: Objeto Data com o grafo gerado
        """
        # Inicializa grafo vazio
        device = next(self.parameters()).device
        x = torch.zeros((0, self.node_hidden_size), device=device)
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.zeros((0, 1), device=device)
        
        # Limpa buffers de log prob
        self.add_node_agent.log_probs.clear()
        self.add_edge_agent.log_probs.clear()
        self.choose_dest_agent.log_probs.clear()
        
        # Processa sequência de ações
        t = 0
        stop, x = self.add_node_agent(x, actions[t] if actions else None)
        t += 1
        
        while not stop and x.size(0) <= self.v_max:
            # Fase de adição de arestas
            to_add_edge = self.add_edge_agent(x, actions[t] if actions else None)
            t += 1
            
            while to_add_edge:
                x, edge_index, edge_attr = self.choose_dest_agent(
                    x, edge_index, edge_attr, 
                    actions[t] if actions else None
                )
                t += 1
                to_add_edge = self.add_edge_agent(x, actions[t] if actions else None)
                t += 1
            
            # Fase de adição de nó
            stop, x = self.add_node_agent(x, actions[t] if actions else None)
            t += 1
        
        if self.training:
            device = self.device()
            node_log_prob = torch.cat(self.add_node_agent.log_probs).sum() if self.add_node_agent.log_probs else torch.zeros([], device=device, requires_grad=True)
            edge_log_prob = torch.cat(self.add_edge_agent.log_probs).sum() if self.add_edge_agent.log_probs else torch.zeros([], device=device, requires_grad=True)
            dest_log_prob = torch.cat(self.choose_dest_agent.log_probs).sum() if self.choose_dest_agent.log_probs else torch.zeros([], device=device, requires_grad=True)
            return node_log_prob + edge_log_prob + dest_log_prob

        else:
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)