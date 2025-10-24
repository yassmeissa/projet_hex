import time
import random
import math
from typing import List, Tuple, Optional, Set
from collections import deque
from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_hex import GameStateHex
from seahorse.game.light_action import LightAction

class MyPlayer(PlayerHex):
    """
    Intelligent Hex player using Minimax with Alpha-Beta pruning and sophisticated heuristics.
    
    Attributes:
        piece_type (str): "R" for red player, "B" for blue player
        opponent_type (str): opponent's piece type
        _start_time (float): timestamp when compute_action started
        _time_limit (float): maximum time allowed for this move
        _moves_made (int): number of moves made so far
        _total_time_used (float): total time used so far
    """

    def __init__(self, piece_type: str, name: str = "HexMaster"):
        """
        Initialize the Hex player.

        Args:
            piece_type (str): Type of the player's game piece ("R" or "B")
            name (str): Name of the player
        """
        super().__init__(piece_type, name)
        self.opponent_type = "B" if piece_type == "R" else "R"
        self._start_time = 0
        self._time_limit = 0
        self._moves_made = 0
        self._total_time_used = 0

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Choose the best action using a strong greedy-path policy, with minimax fallback.

        Args:
            current_state (GameState): The current game state
            remaining_time (int): Remaining time in milliseconds

        Returns:
            Action: The best action found
        """
        self._start_time = time.time()
        
        # Convert remaining time from milliseconds to seconds and apply safety margin
        remaining_seconds = remaining_time / 1000.0
        self._time_limit = min(remaining_seconds * 0.95, 60.0)  # Use 95% of remaining time, max 60s
        
        # Aggressive time management - invest heavily in critical moves
        moves_left_estimate = max(1, (14 * 14) // 2 - self._moves_made)
        time_per_move = remaining_seconds / moves_left_estimate
        
        # Allocate maximum time for deep analysis against human players
        if self._moves_made < 8:
            self._time_limit = min(self._time_limit, time_per_move * 6)  # 6x average for opening
        elif self._moves_made < 20:
            self._time_limit = min(self._time_limit, time_per_move * 5)  # 5x average for mid-game
        else:
            self._time_limit = min(self._time_limit, time_per_move * 8)  # 8x average for endgame
        
        possible_actions = list(current_state.get_possible_light_actions())
        
        if not possible_actions:
            return None
            
        if len(possible_actions) == 1:
            self._moves_made += 1
            elapsed_time = time.time() - self._start_time
            self._total_time_used += elapsed_time
            return possible_actions[0]
        
        # Special handling for opening moves - be more aggressive
        if self._moves_made < 2:
            action = self._get_opening_move(current_state, possible_actions)
            self._moves_made += 1
            elapsed_time = time.time() - self._start_time
            self._total_time_used += elapsed_time
            return action
        
        # Maximum candidate analysis for human-level play
        max_candidates = 20 if self._moves_made < 10 else 18  # More candidates for thorough analysis
        candidate_actions = self._select_best_moves(current_state, possible_actions, max_candidates=max_candidates)
        
        # Urgent tactical check (use full set to never miss a win/block)
        urgent_action = self._check_urgent_moves(current_state, possible_actions)
        if urgent_action:
            self._moves_made += 1
            elapsed_time = time.time() - self._start_time
            self._total_time_used += elapsed_time
            return urgent_action
        
        # Enhanced proactive blocking: be more aggressive about blocking
        try:
            opp_len, opp_path = self._shortest_path_with_path(current_state, self.opponent_type)
            self_len, _ = self._shortest_path_with_path(current_state, self.piece_type)
        except Exception:
            opp_len, opp_path, self_len = None, None, None
        if opp_len is not None and opp_path:
            # Ultra-aggressive blocking against human players - detect and counter threats
            should_block = (
                opp_len <= 4 or  # Block if opponent needs 4 or fewer moves
                (self_len is not None and opp_len <= self_len + 1) or  # Block if opponent is close to our progress
                (self_len is not None and self_len > 6 and opp_len <= 6) or  # Block if we're behind and they're advancing
                (opp_len <= 5 and self._moves_made > 10)  # Late-game aggressive blocking
            )
            if should_block:
                best_block = None
                best_gain = -math.inf
                opp_positions = set(opp_path)
                for a in possible_actions:
                    pos = a.data["position"]
                    if pos not in opp_positions:
                        continue
                    ns = current_state.apply_action(a)
                    new_opp = self._shortest_path_length(ns, self.opponent_type)
                    new_self = self._shortest_path_length(ns, self.piece_type)
                    # Enhanced blocking score calculation
                    gain = (new_opp - opp_len) * 2 + 0.2 * (new_opp - new_self)
                    if gain > best_gain:
                        best_gain = gain
                        best_block = a
                if best_block is not None and best_gain > 0.5:  # Only block if significant gain
                    self._moves_made += 1
                    elapsed_time = time.time() - self._start_time
                    self._total_time_used += elapsed_time
                    return best_block
        
        # Anti-greedy strategy: prioritize path differential over greedy path following
        sp_action = self._choose_by_shortest_path_diff(current_state, candidate_actions)
        if sp_action is not None:
            # Deep multi-ply analysis of top candidates to beat human players
            quick_alternatives = candidate_actions[:8]  # Top 8 candidates for more thorough analysis
            best_action = sp_action
            best_score = float('-inf')
            
            for action in quick_alternatives:
                if self._is_time_up():
                    break
                new_state = current_state.apply_action(action)
                
                # Multi-level evaluation: immediate + 2-ply lookahead
                immediate_score = 0
                try:
                    our_path = self._shortest_path_length(new_state, self.piece_type)
                    opp_path = self._shortest_path_length(new_state, self.opponent_type)
                    immediate_score = (opp_path - our_path) * 15 + self._evaluate_state(new_state, True) * 0.02
                    
                    # Add 2-ply lookahead for critical moves
                    if not self._is_time_up() and len(quick_alternatives) <= 5:
                        future_actions = list(new_state.get_possible_light_actions())[:5]
                        future_score = 0
                        for future_action in future_actions:
                            if self._is_time_up():
                                break
                            future_state = new_state.apply_action(future_action)
                            future_our = self._shortest_path_length(future_state, self.piece_type)
                            future_opp = self._shortest_path_length(future_state, self.opponent_type)
                            future_score += (future_opp - future_our) * 2
                        immediate_score += future_score / len(future_actions) if future_actions else 0
                    
                    if immediate_score > best_score:
                        best_score = immediate_score
                        best_action = action
                except:
                    continue
            
            self._moves_made += 1
            elapsed_time = time.time() - self._start_time
            self._total_time_used += elapsed_time
            return best_action
        
        # Fallback: enhanced greedy if path differential fails
        greedy_action = self._get_enhanced_greedy_move(current_state, possible_actions)
        if greedy_action:
            self._moves_made += 1
            elapsed_time = time.time() - self._start_time
            self._total_time_used += elapsed_time
            return greedy_action
        
        # Minimax fallback with iterative deepening (rarely reached against baselines)
        best_action = None
        best_value = float('-inf')
        # Maximum search depth - very deep to beat experienced human players
        if self._moves_made < 5:
            max_depth = 10  # Very deep opening analysis
        elif self._moves_made < 15:
            max_depth = 12 if len(candidate_actions) < 10 else 10  # Deep mid-game
        else:
            max_depth = 15 if len(candidate_actions) < 6 else 12  # Extremely deep endgame
        
        for depth in range(1, max_depth + 1):
            if self._is_time_up():
                break
            try:
                action, value = self._minimax_root(current_state, depth, candidate_actions)
                if action is not None:
                    best_action = action
                    best_value = value
                    if value >= 9000:
                        break
            except TimeoutError:
                break
        
        self._moves_made += 1
        elapsed_time = time.time() - self._start_time
        self._total_time_used += elapsed_time
        return best_action if best_action else random.choice(possible_actions)

    def _evaluate_board_control(self, state: GameState) -> float:
        """
        Evaluate overall board control and positioning.
        """
        dimensions = state.rep.get_dimensions()
        env = state.rep.get_env()
        
        our_pieces = set()
        opponent_pieces = set()
        
        for pos, piece in env.items():
            if piece is not None:
                if piece.piece_type == self.piece_type:
                    our_pieces.add(pos)
                else:
                    opponent_pieces.add(pos)
        
        # Calculate territory control
        our_territory = self._calculate_territory_influence(our_pieces, dimensions)
        opponent_territory = self._calculate_territory_influence(opponent_pieces, dimensions)
        
        return (our_territory - opponent_territory) * 2

    def _calculate_territory_influence(self, pieces: Set[Tuple[int, int]], dimensions: Tuple[int, int]) -> float:
        """
        Calculate how much territory/influence a set of pieces controls.
        """
        if not pieces:
            return 0
        
        rows, cols = dimensions
        influence = 0
        
        # Each piece influences nearby empty squares
        for piece in pieces:
            for distance in range(1, 4):  # Check influence up to 3 squares away
                for dr in range(-distance, distance + 1):
                    for dc in range(-distance, distance + 1):
                        if abs(dr) + abs(dc) <= distance:
                            new_pos = (piece[0] + dr, piece[1] + dc)
                            if (0 <= new_pos[0] < rows and 0 <= new_pos[1] < cols):
                                # Weight decreases with distance
                                weight = 1 / (distance + 1)
                                influence += weight
        
        return influence

    def _evaluate_threats(self, state: GameState) -> float:
        """
        Evaluate immediate threats and opportunities.
        """
        # Check if opponent is close to winning
        opponent_eval = self._evaluate_player_position(state, self.opponent_type)
        our_eval = self._evaluate_player_position(state, self.piece_type)
        
        # If opponent has a strong position, prioritize blocking
        if opponent_eval > 800:  # Opponent close to winning
            return -50
        elif our_eval > 800:  # We're close to winning
            return 50
        
        return 0

    def _get_opening_move(self, current_state: GameState, possible_actions: List[LightAction]) -> LightAction:
        """
        Get a strong opening move using opening theory.
        
        Args:
            current_state: Current game state
            possible_actions: Available actions
            
        Returns:
            LightAction: A strong opening move
        """
        dimensions = current_state.rep.get_dimensions()
        rows, cols = dimensions
        
        # First move: take center or near-center
        if self._moves_made == 0:
            center_moves = []
            center_row, center_col = rows // 2, cols // 2
            
            # Look for positions near center
            for action in possible_actions:
                pos = action.data["position"]
                distance = abs(pos[0] - center_row) + abs(pos[1] - center_col)
                if distance <= 2:
                    center_moves.append((distance, action))
            
            if center_moves:
                # Sort by distance to center and pick the closest
                center_moves.sort(key=lambda x: x[0])
                return center_moves[0][1]
        
        # Second move: establish connection or respond to threat
        elif self._moves_made == 1:
            env = current_state.rep.get_env()
            
            # Find opponent's first move
            opponent_pieces = []
            for pos, piece in env.items():
                if piece is not None and piece.piece_type == self.opponent_type:
                    opponent_pieces.append(pos)
            
            if opponent_pieces:
                opponent_pos = opponent_pieces[0]
                
                # Play near our strategy edges while staying away from opponent
                good_moves = []
                for action in possible_actions:
                    pos = action.data["position"]
                    
                    # Distance from opponent (prefer further)
                    opp_distance = abs(pos[0] - opponent_pos[0]) + abs(pos[1] - opponent_pos[1])
                    
                    # Distance from our target edges
                    if self.piece_type == "R":
                        edge_distance = min(pos[0], rows - 1 - pos[0])  # Distance to top or bottom
                    else:
                        edge_distance = min(pos[1], cols - 1 - pos[1])  # Distance to left or right
                    
                    # Prefer moves that are not too close to opponent but near our edges
                    if opp_distance >= 3 and edge_distance <= 4:
                        good_moves.append(action)
                
                if good_moves:
                    return random.choice(good_moves)
        
        # Default: pick a reasonable move
        return random.choice(possible_actions)

    def _minimax_root(self, state: GameState, max_depth: int, possible_actions: List[LightAction]) -> Tuple[Optional[LightAction], float]:
        """
        Root call for minimax algorithm.
        
        Args:
            state: Current game state
            max_depth: Maximum search depth
            possible_actions: Available actions
            
        Returns:
            Tuple of (best_action, best_value)
        """
        best_action = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        # Sort actions by heuristic value for better pruning
        action_values = []
        for action in possible_actions:
            if self._is_time_up():
                raise TimeoutError()
            new_state = state.apply_action(action)
            value = self._evaluate_state(new_state, True)
            action_values.append((value, action))
        
        # Sort in descending order (best actions first)
        action_values.sort(reverse=True)
        
        for _, action in action_values:
            if self._is_time_up():
                raise TimeoutError()
                
            new_state = state.apply_action(action)
            value = self._minimax(new_state, max_depth - 1, alpha, beta, False)
            
            if value > best_value:
                best_value = value
                best_action = action
            
            alpha = max(alpha, value)
            if beta <= alpha:
                break
                
        return best_action, best_value

    def _minimax(self, state: GameState, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        """
        Minimax algorithm with alpha-beta pruning.
        
        Args:
            state: Current game state
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing: True if maximizing player's turn
            
        Returns:
            float: Evaluation value of the state
        """
        if self._is_time_up():
            raise TimeoutError()
            
        if depth == 0 or state.is_done():
            return self._evaluate_state(state, maximizing)
        
        possible_actions = list(state.get_possible_light_actions())
        
        if maximizing:
            max_eval = float('-inf')
            for action in possible_actions:
                if self._is_time_up():
                    raise TimeoutError()
                new_state = state.apply_action(action)
                eval_score = self._minimax(new_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for action in possible_actions:
                if self._is_time_up():
                    raise TimeoutError()
                new_state = state.apply_action(action)
                eval_score = self._minimax(new_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def _evaluate_state(self, state: GameState, from_our_perspective: bool) -> float:
        """
        Evaluate the game state using multiple heuristics with improved weighting.
        
        Args:
            state: Game state to evaluate
            from_our_perspective: True if evaluating from our player's perspective
            
        Returns:
            float: Evaluation score (higher is better for us)
        """
        if state.is_done():
            winner_score = state.scores[self.get_id()]
            if winner_score == 1:
                return 10000  # We won
            else:
                return -10000  # We lost
        
        our_eval = self._evaluate_player_position(state, self.piece_type)
        opponent_eval = self._evaluate_player_position(state, self.opponent_type)
        
        # Calculate relative advantage with improved scaling
        score_diff = our_eval - opponent_eval
        
        # Add positional bonuses
        board_control = self._evaluate_board_control(state)
        threat_assessment = self._evaluate_threats(state)
        
        total_score = score_diff + board_control + threat_assessment
        
        return total_score if from_our_perspective else -total_score

    def _evaluate_player_position(self, state: GameState, player_type: str) -> float:
        """
        Evaluate position strength for a specific player using improved heuristics.
        
        Args:
            state: Game state
            player_type: "R" or "B"
            
        Returns:
            float: Position evaluation for the player
        """
        dimensions = state.rep.get_dimensions()
        env = state.rep.get_env()
        
        # Get player pieces
        player_pieces = set()
        for pos, piece in env.items():
            if piece is not None and piece.piece_type == player_type:
                player_pieces.add(pos)
        
        if not player_pieces:
            return 0
        
        # Calculate shortest path distance (most important)
        path_score = self._calculate_shortest_path_improved(state, player_type, player_pieces)
        
        # Calculate connectivity and bridge potential
        connectivity_score = self._calculate_connectivity_score(player_pieces, dimensions)
        
        # Calculate edge control
        edge_score = self._calculate_edge_control(state, player_type, player_pieces)
        
        # Calculate opponent blocking potential
        blocking_score = self._calculate_blocking_potential(state, player_type)
        
        # Calculate strategic position value
        strategic_score = self._calculate_strategic_positions(player_pieces, dimensions, player_type)
        
        # Maximum weighted combination for human-level play
        total_score = (
            path_score * 500 +       # Highest priority: path to victory (maximized)
            edge_score * 450 +       # Critical: edge control dominance (maximized)
            connectivity_score * 300 + # Very high: piece connectivity (maximized)
            strategic_score * 250 +   # Strategic positions (maximized)
            blocking_score * 200      # High: blocking opponent (maximized)
        )
        
        return total_score

    def _calculate_shortest_path_improved(self, state: GameState, player_type: str, player_pieces: Set[Tuple[int, int]]) -> float:
        """
        Improved shortest path calculation using a more robust approach.
        """
        dimensions = state.rep.get_dimensions()
        rows, cols = dimensions
        env = state.rep.get_env()
        
        # Use Union-Find to detect connected components
        if player_type == "R":
            # Red: check connection from top edge to bottom edge
            top_connected = False
            bottom_connected = False
            
            # Check if we have pieces on target edges
            for j in range(cols):
                if (0, j) in player_pieces:
                    top_connected = True
                if (rows-1, j) in player_pieces:
                    bottom_connected = True
            
            if top_connected and bottom_connected:
                # Check if there's a path connecting them
                if self._has_connected_path(player_pieces, dimensions, "R"):
                    return 1000  # We have a winning path!
            
            # Calculate minimum spanning distance using BFS
            return self._calculate_connection_potential(state, player_type, player_pieces)
            
        else:  # Blue player
            # Blue: check connection from left edge to right edge
            left_connected = False
            right_connected = False
            
            for i in range(rows):
                if (i, 0) in player_pieces:
                    left_connected = True
                if (i, cols-1) in player_pieces:
                    right_connected = True
            
            if left_connected and right_connected:
                if self._has_connected_path(player_pieces, dimensions, "B"):
                    return 1000  # We have a winning path!
            
            return self._calculate_connection_potential(state, player_type, player_pieces)

    def _has_connected_path(self, pieces: Set[Tuple[int, int]], dimensions: Tuple[int, int], player_type: str) -> bool:
        """
        Check if pieces form a connected path from one edge to the opposite edge.
        """
        if not pieces:
            return False
        
        rows, cols = dimensions
        
        # Find starting pieces (on the source edge)
        if player_type == "R":
            start_pieces = [p for p in pieces if p[0] == 0]  # Top edge
            target_row = rows - 1
        else:
            start_pieces = [p for p in pieces if p[1] == 0]  # Left edge
            target_col = cols - 1
        
        if not start_pieces:
            return False
        
        # BFS to see if we can reach the target edge
        visited = set()
        queue = deque(start_pieces)
        visited.update(start_pieces)
        
        while queue:
            current = queue.popleft()
            
            # Check if we reached target edge
            if player_type == "R" and current[0] == target_row:
                return True
            elif player_type == "B" and current[1] == target_col:
                return True
            
            # Explore connected pieces
            for neighbor in self._get_neighbors(current, dimensions):
                if neighbor in pieces and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return False

    def _calculate_connection_potential(self, state: GameState, player_type: str, player_pieces: Set[Tuple[int, int]]) -> float:
        """
        Calculate the potential for creating a winning connection.
        """
        dimensions = state.rep.get_dimensions()
        rows, cols = dimensions
        env = state.rep.get_env()
        
        # Use a flood-fill approach to find shortest virtual path
        visited = set()
        distances = {}
        
        if player_type == "R":
            # Start from top edge (row 0)
            sources = []
            for j in range(cols):
                pos = (0, j)
                if pos in player_pieces:
                    distances[pos] = 0
                    sources.append(pos)
                elif env.get(pos) is None:  # Empty position
                    distances[pos] = 1
                    sources.append(pos)
            
            # BFS to find shortest path to bottom
            queue = deque(sources)
            
            while queue:
                current = queue.popleft()
                current_dist = distances[current]
                
                # Check if reached bottom edge
                if current[0] == rows - 1:
                    return max(0, 200 - current_dist * 8)
                
                # Explore neighbors
                for neighbor in self._get_neighbors(current, dimensions):
                    if neighbor in visited:
                        continue
                    
                    piece_at_pos = env.get(neighbor)
                    cost = 0 if neighbor in player_pieces else (1 if piece_at_pos is None else 999)
                    
                    if cost < 999:  # Can move here
                        new_dist = current_dist + cost
                        if neighbor not in distances or new_dist < distances[neighbor]:
                            distances[neighbor] = new_dist
                            queue.append(neighbor)
                    
                    visited.add(neighbor)
            
        else:  # Blue player
            sources = []
            for i in range(rows):
                pos = (i, 0)
                if pos in player_pieces:
                    distances[pos] = 0
                    sources.append(pos)
                elif env.get(pos) is None:
                    distances[pos] = 1
                    sources.append(pos)
            
            queue = deque(sources)
            
            while queue:
                current = queue.popleft()
                current_dist = distances[current]
                
                if current[1] == cols - 1:
                    return max(0, 200 - current_dist * 8)
                
                for neighbor in self._get_neighbors(current, dimensions):
                    if neighbor in visited:
                        continue
                    piece_at_pos = env.get(neighbor)
                    cost = 0 if neighbor in player_pieces else (1 if piece_at_pos is None else 999)
                    if cost < 999:
                        new_dist = current_dist + cost
                        if neighbor not in distances or new_dist < distances[neighbor]:
                            distances[neighbor] = new_dist
                            queue.append(neighbor)
                    visited.add(neighbor)
            
        # If no path found, return minimal score
        return 0

    def _calculate_edge_control(self, state: GameState, player_type: str, player_pieces: Set[Tuple[int, int]]) -> float:
        """
        Calculate how well the player controls their target edges.
        """
        dimensions = state.rep.get_dimensions()
        rows, cols = dimensions
        edge_score = 0
        if player_type == "R":
            top_pieces = sum(1 for j in range(cols) if (0, j) in player_pieces)
            bottom_pieces = sum(1 for j in range(cols) if (rows-1, j) in player_pieces)
            edge_score = (top_pieces + bottom_pieces) * 15
            if (0, 0) in player_pieces: edge_score += 5
            if (0, cols-1) in player_pieces: edge_score += 5
            if (rows-1, 0) in player_pieces: edge_score += 5
            if (rows-1, cols-1) in player_pieces: edge_score += 5
        else:
            left_pieces = sum(1 for i in range(rows) if (i, 0) in player_pieces)
            right_pieces = sum(1 for i in range(rows) if (i, cols-1) in player_pieces)
            edge_score = (left_pieces + right_pieces) * 15
            if (0, 0) in player_pieces: edge_score += 5
            if (0, cols-1) in player_pieces: edge_score += 5
            if (rows-1, 0) in player_pieces: edge_score += 5
            if (rows-1, cols-1) in player_pieces: edge_score += 5
        return edge_score

    def _shortest_path_length(self, state: GameState, player_type: str) -> int:
        """Minimal empty stones needed to connect sides using Dijkstra-like costs."""
        import heapq
        rows, cols = state.rep.get_dimensions()
        env = state.rep.get_env()
        INF = 10**9
        dist = [[INF]*cols for _ in range(rows)]
        pq = []
        if player_type == "R":
            for j in range(cols):
                n_type = env.get((0, j)).get_type() if env.get((0, j)) is not None else "EMPTY"
                cost = 0 if n_type == "R" else (1 if n_type == "EMPTY" else INF)
                if cost < INF:
                    dist[0][j] = cost
                    heapq.heappush(pq, (cost, 0, j))
            targets = {(rows-1, j) for j in range(cols)}
        else:
            for i in range(rows):
                n_type = env.get((i, 0)).get_type() if env.get((i, 0)) is not None else "EMPTY"
                cost = 0 if n_type == "B" else (1 if n_type == "EMPTY" else INF)
                if cost < INF:
                    dist[i][0] = cost
                    heapq.heappush(pq, (cost, i, 0))
            targets = {(i, cols-1) for i in range(rows)}
        while pq:
            d, i, j = heapq.heappop(pq)
            if d != dist[i][j]:
                continue
            if (i, j) in targets:
                return d
            for n_type, (ni, nj) in state.rep.get_neighbours(i, j).values():
                if n_type == "EMPTY":
                    nd = d + 1
                elif n_type == player_type:
                    nd = d
                else:
                    continue
                if 0 <= ni < rows and 0 <= nj < cols and nd < dist[ni][nj]:
                    dist[ni][nj] = nd
                    heapq.heappush(pq, (nd, ni, nj))
        return INF

    def _shortest_path_with_path(self, state: GameState, player_type: str) -> Tuple[int, List[Tuple[int,int]]]:
        """Dijkstra that returns both length and one shortest path nodes (as grid coords)."""
        import heapq
        rows, cols = state.rep.get_dimensions()
        env = state.rep.get_env()
        INF = 10**9
        dist = [[INF]*cols for _ in range(rows)]
        pred: List[List[Optional[Tuple[int,int]]]] = [[None]*cols for _ in range(rows)]
        pq = []
        targets: Set[Tuple[int,int]] = set()
        if player_type == 'R':
            for j in range(cols):
                piece = env.get((0, j))
                cost = 0 if (piece is not None and piece.get_type() == 'R') else 1
                dist[0][j] = cost
                heapq.heappush(pq, (cost, 0, j))
                targets.add((rows-1, j))
        else:
            for i in range(rows):
                piece = env.get((i, 0))
                cost = 0 if (piece is not None and piece.get_type() == 'B') else 1
                dist[i][0] = cost
                heapq.heappush(pq, (cost, i, 0))
                targets.add((i, cols-1))
        end: Optional[Tuple[int,int]] = None
        while pq:
            d, i, j = heapq.heappop(pq)
            if d != dist[i][j]:
                continue
            if (i, j) in targets:
                end = (i, j)
                break
            for n_type, (ni, nj) in state.rep.get_neighbours(i, j).values():
                if n_type == 'EMPTY':
                    nd = d + 1
                elif n_type == player_type:
                    nd = d
                else:
                    continue
                if 0 <= ni < rows and 0 <= nj < cols and nd < dist[ni][nj]:
                    dist[ni][nj] = nd
                    pred[ni][nj] = (i, j)
                    heapq.heappush(pq, (nd, ni, nj))
        if end is None:
            return (INF, [])
        # Reconstruct path
        path: List[Tuple[int,int]] = []
        cur = end
        while cur is not None:
            path.append(cur)
            ci, cj = cur
            cur = pred[ci][cj]
        path.reverse()
        return (dist[end[0]][end[1]], path)

    def _choose_by_shortest_path_diff(self, state: GameState, actions: List[LightAction]) -> Optional[LightAction]:
        """Pick move maximizing (opponent path - our path) after the move."""
        best = None
        best_score = -math.inf
        for a in actions:
            new_state = state.apply_action(a)
            d_self = self._shortest_path_length(new_state, self.piece_type)
            d_opp = self._shortest_path_length(new_state, self.opponent_type)
            score = d_opp - d_self
            pos = a.data["position"]
            rows, cols = state.rep.get_dimensions()
            center = abs(pos[0] - rows//2) + abs(pos[1] - cols//2)
            score -= 0.01 * center
            if score > best_score:
                best_score = score
                best = a
        return best

    # ===== Added helpers to stabilize agent =====
    def _is_time_up(self) -> bool:
        return (time.time() - self._start_time) >= (self._time_limit - 0.01)

    def _get_neighbors(self, pos: Tuple[int, int], dimensions: Tuple[int, int]) -> List[Tuple[int, int]]:
        i, j = pos
        rows, cols = dimensions
        candidates = [(i-1, j+1), (i-1, j), (i+1, j-1), (i+1, j), (i, j-1), (i, j+1)]
        return [(r, c) for (r, c) in candidates if 0 <= r < rows and 0 <= c < cols]

    def _select_best_moves(self, state: GameState, actions: List[LightAction], max_candidates: int = 15) -> List[LightAction]:
        scored = []
        for a in actions:
            ns = state.apply_action(a)
            d_self = self._shortest_path_length(ns, self.piece_type)
            d_opp = self._shortest_path_length(ns, self.opponent_type)
            # Larger (d_opp - d_self) is better
            score = d_opp - d_self
            scored.append((score, a))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [a for _, a in scored[:max_candidates]] if len(scored) > max_candidates else [a for _, a in scored]

    def _check_urgent_moves(self, state: GameState, actions: List[LightAction]) -> Optional[LightAction]:
        # 1) If we can win immediately, do it
        for a in actions:
            ns = state.apply_action(a)
            if ns.is_done() and ns.scores.get(self.get_id(), 0) == 1:
                return a
        # 2) If opponent can win next move, block the winning position if possible
        # Find any opponent immediate win from the current state
        opp_actions = list(state.get_possible_light_actions())
        for oa in opp_actions:
            if oa.data["piece"] == self.piece_type:
                # Ensure actions are for opponent
                oa = LightAction({"piece": self.opponent_type, "position": oa.data["position"]})
            ns = state.apply_action(oa)
            if ns.is_done():
                block_pos = oa.data["position"]
                for a in actions:
                    if a.data["position"] == block_pos:
                        return a
        return None

    def _get_enhanced_greedy_move(self, state: GameState, actions: List[LightAction]) -> Optional[LightAction]:
        import heapq
        rows, cols = state.rep.get_dimensions()
        env = state.rep.get_env()
        INF = 10**9
        dist = [[INF]*cols for _ in range(rows)]
        pred: List[List[Optional[Tuple[int,int]]]] = [[None]*cols for _ in range(rows)]
        pq = []
        objectives: Set[Tuple[int,int]] = set()
        if self.piece_type == "R":
            for j in range(cols):
                piece = env.get((0, j))
                cost = 0 if (piece is not None and getattr(piece, 'piece_type', None) == 'R') else 1
                dist[0][j] = cost
                heapq.heappush(pq, (cost, (0, j)))
                objectives.add((rows-1, j))
        else:
            for i in range(rows):
                piece = env.get((i, 0))
                cost = 0 if (piece is not None and getattr(piece, 'piece_type', None) == 'B') else 1
                dist[i][0] = cost
                heapq.heappush(pq, (cost, (i, 0)))
                objectives.add((i, cols-1))
        end: Optional[Tuple[int,int]] = None
        while pq:
            d, (i, j) = heapq.heappop(pq)
            if d != dist[i][j]:
                continue
            if (i, j) in objectives:
                end = (i, j)
                break
            for (ni, nj) in self._get_neighbors((i, j), (rows, cols)):
                piece = env.get((ni, nj))
                if piece is None:
                    nd = d + 1
                elif getattr(piece, 'piece_type', None) == self.piece_type:
                    nd = d
                else:
                    continue
                if nd < dist[ni][nj]:
                    dist[ni][nj] = nd
                    pred[ni][nj] = (i, j)
                    heapq.heappush(pq, (nd, (ni, nj)))
        if end is None:
            return None
        # Reconstruct path
        path: List[Tuple[int,int]] = []
        cur = end
        while cur is not None:
            path.append(cur)
            ci, cj = cur
            cur = pred[ci][cj]
        path.reverse()
        # Select best empty along path among allowed actions
        action_positions = {a.data["position"] for a in actions}
        candidates: List[Tuple[float, Tuple[int,int]]] = []
        for pos in path:
            if pos in action_positions and env.get(pos) is None:
                # Phase-aware scoring: early prefer center, later prefer alignment
                center_score = abs(pos[0]-rows/2) + abs(pos[1]-cols/2)
                if self._moves_made < 10:
                    score = -center_score
                else:
                    if self.piece_type == 'R':
                        align = -abs(pos[1] - cols/2)
                    else:
                        align = -abs(pos[0] - rows/2)
                    score = -0.7*center_score + 0.3*align
                candidates.append((score, pos))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        # Early game: add small randomness among top-2 best path cells to reduce predictability
        top_k = 2 if self._moves_made < 8 and len(candidates) >= 2 else 1
        choice = random.choice(candidates[:top_k]) if top_k > 1 else candidates[0]
        _, best_pos = choice
        # Return the matching LightAction from actions to keep consistency
        for a in actions:
            if a.data["position"] == best_pos:
                return a
        return None

    def _calculate_connectivity_score(self, pieces: Set[Tuple[int, int]], dimensions: Tuple[int, int]) -> float:
        score = 0.0
        for p in pieces:
            for n in self._get_neighbors(p, dimensions):
                if n in pieces:
                    score += 1.0
        # Each edge in connectivity counted twice; normalize
        return score / 2.0

    def _calculate_blocking_potential(self, state: GameState, player_type: str) -> float:
        # Higher opponent shortest path length => better blocking for player_type
        opponent = 'B' if player_type == 'R' else 'R'
        d_opp = self._shortest_path_length(state, opponent)
        # Cap to avoid extreme weights
        return float(min(d_opp, 20))

    def _calculate_strategic_positions(self, pieces: Set[Tuple[int, int]], dimensions: Tuple[int, int], player_type: str) -> float:
        rows, cols = dimensions
        center_r, center_c = rows/2, cols/2
        score = 0.0
        for (r, c) in pieces:
            # Prefer center
            score += max(0.0, 6.0 - (abs(r-center_r) + abs(c-center_c)) * 0.5)
            # Prefer progress towards target edges
            if player_type == 'R':
                # Encourage vertical spread
                score += (min(r, rows-1-r)) * 0.1
            else:
                # Encourage horizontal spread
                score += (min(c, cols-1-c)) * 0.1
        return score
