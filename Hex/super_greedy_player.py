import random
import numpy as np
import heapq
from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.game.light_action import LightAction

class MyPlayer(PlayerHex):
    """
    Player class for Hex game that makes super greedy moves - amélioration de l'agent glouton.
    """

    def __init__(self, piece_type: str, name: str = "super_greedy",*args) -> None:
        """
        Initialize the SuperGreedy PlayerHex instance.
        """
        super().__init__(piece_type,name,*args)
        self._moves_made = 0

    def compute_action(self, current_state: GameState, **kwargs) -> Action:
        """
        Fonction améliorée qui copie l'agent glouton avec des optimisations.
        """
        self._moves_made += 1
        possible_actions = current_state.get_possible_light_actions()

        # Algorithme glouton amélioré - exactement comme l'original mais avec des tweaks
        env = current_state.rep.env
        dist = np.full((current_state.rep.dimensions[0], current_state.rep.dimensions[1]), np.inf)
        preds = np.full((current_state.rep.dimensions[0], current_state.rep.dimensions[1]), None, dtype=tuple)
        objectives = []
        pq = []
        
        if self.piece_type == "R":
            for j in range(current_state.rep.dimensions[1]):
                if env.get((0,j)) == "R":
                    dist[0, j] = 0
                else:
                    dist[0, j] = 1
                heapq.heappush(pq, (dist[0, j], (0, j), None))
                objectives.append((current_state.rep.dimensions[0]-1, j))

        else:
            for i in range(current_state.rep.dimensions[0]):
                if env.get((i,0)) == "B":
                    dist[i, 0] = 0
                else:
                    dist[i, 0] = 1
                heapq.heappush(pq, (dist[i, 0], (i, 0), None))
                objectives.append((i, current_state.rep.dimensions[1]-1))
        
        path=[]
        while len(pq) != 0:
            d, (i, j), pred = heapq.heappop(pq)
            if d > dist[i, j]:
                continue
            preds[i,j] = pred
            if (i,j) in objectives:
                path = retrace_path(preds, (i,j))
                break
            for n_type, (ni, nj) in current_state.rep.get_neighbours(i, j).values():
                if n_type == "EMPTY":
                    new_dist = d + 1
                elif n_type == self.piece_type:
                    new_dist = d 
                else:
                    continue
                if new_dist < dist[ni, nj]:
                    dist[ni, nj] = new_dist
                    heapq.heappush(pq, (new_dist, (ni, nj), (i, j)))
        
        # Amélioration : sélection plus intelligente du point sur le chemin
        hq = []
        for pos in path:
            if env.get(pos) == None:
                # Score amélioré : on favorise les positions selon la phase du jeu
                center_score = abs(pos[0]-6.5) + abs(pos[1]-6.5)
                
                # Début de partie : préférer le centre
                if self._moves_made < 10:
                    final_score = center_score
                else:
                    # Milieu/fin de partie : préférer les positions qui complètent notre stratégie
                    if self.piece_type == "R":
                        # Rouge : favoriser les positions qui créent des connexions verticales
                        strategic_bonus = min(abs(pos[0] - 3), abs(pos[0] - 10)) * 0.5
                    else:
                        # Bleu : favoriser les positions qui créent des connexions horizontales  
                        strategic_bonus = min(abs(pos[1] - 3), abs(pos[1] - 10)) * 0.5
                    
                    final_score = center_score - strategic_bonus
                
                heapq.heappush(hq, (final_score, pos))
        
        if hq:
            _ , pos = heapq.heappop(hq)
            return LightAction({"piece": self.piece_type, "position": pos})
        else:
            # Fallback : prendre n'importe quelle action disponible
            return list(possible_actions)[0]

def retrace_path(preds, end):
    """
    Recreate the path from the start to the end position using the predecessors.
    """
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = preds[current]
    return path
