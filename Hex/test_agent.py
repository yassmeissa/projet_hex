#!/usr/bin/env python3
"""
Script de test pour évaluer la performance de notre agent Hex
"""

import subprocess
import sys
import re
import time
from pathlib import Path

def run_game(player1, player2, timeout=120):
    """
    Lance une partie entre deux joueurs et retourne le résultat.
    
    Args:
        player1: Fichier du premier joueur
        player2: Fichier du second joueur
        timeout: Timeout en secondes
        
    Returns:
        tuple: (gagnant, log_complet)
    """
    cmd = [
        "/Users/yassmeissa/Downloads/Projet_Hex_A2025/.venv/bin/python",
        "main_hex.py",
        "-t", "local",
        player1, player2,
        "-g"  # Sans interface graphique
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/Users/yassmeissa/Downloads/Projet_Hex_A2025/Hex"
        )
        
        # Analyser le résultat
        output = result.stderr  # Les logs sont dans stderr
        
        # Chercher le gagnant
        winner_match = re.search(r'(\w+) has won the game', output)
        if winner_match:
            winner = winner_match.group(1)
            return winner, output
        else:
            # Sans modifier greedy: si l'adversaire greedy crashe (heappop sur tas vide),
            # ne pas compter cette partie dans les statistiques.
            greedy_crash_signatures = [
                'greedy_player_hex.py',
                'IndexError: index out of range',
                'heapq.heappop',
                'Traceback'
            ]
            if (
                player2 == 'greedy_player_hex.py' and
                all(sig in output for sig in greedy_crash_signatures)
            ):
                return 'IGNORED', output
            
            return "UNKNOWN", output
            
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "Game timed out"
    except Exception as e:
        return "ERROR", str(e)

def test_agent_performance():
    """
    Teste la performance de notre agent contre différents adversaires.
    """
    print("🎮 Test de performance de HexMaster")
    print("=" * 50)
    
    # Adversaires à tester
    opponents = {
        "random_player_hex.py": "Agent Aléatoire",
        "greedy_player_hex.py": "Agent Glouton"
    }
    
    our_agent = "my_player.py"
    total_games = 0
    total_wins = 0
    total_ignored = 0
    
    for opponent_file, opponent_name in opponents.items():
        print(f"\n🤖 Test contre {opponent_name}")
        print("-" * 30)
        
        games_to_play = 5  # Nombre de parties à jouer
        wins = 0
        losses = 0
        errors = 0
        ignored = 0
        
        for game_num in range(games_to_play):
            print(f"Partie {game_num + 1}/{games_to_play}... ", end="", flush=True)
            
            # Relance automatique si la partie est ignorée (crash adverse)
            retries = 0
            while True:
                winner, log = run_game(our_agent, opponent_file)
                if winner == 'IGNORED':
                    retries += 1
                    if retries >= 3:
                        ignored += 1
                        print("⏭️  Ignorée (crash adverse) → abandon après 3 essais")
                        break
                    else:
                        print(f"⏭️  Ignorée (crash adverse) → relance ({retries}/3)", end="", flush=True)
                        time.sleep(0.2)
                        continue
                break
            
            # Si toujours ignorée après 3 essais, on passe à la partie suivante
            if winner == 'IGNORED':
                continue
            
            if winner.startswith("my_player"):
                wins += 1
                print("✅ Victoire")
            elif winner.startswith(opponent_file.split('.')[0]):
                losses += 1
                print("❌ Défaite")
            else:
                errors += 1
                print(f"⚠️  Erreur: {winner}")
            
            time.sleep(0.5)  # Petite pause entre les parties
        
        # Statistiques pour cet adversaire
        counted_games = games_to_play - ignored
        win_rate = (wins / counted_games) * 100 if counted_games > 0 else 0
        print(f"\nRésultats contre {opponent_name}:")
        print(f"  Victoires: {wins}")
        print(f"  Défaites: {losses}")
        print(f"  Erreurs: {errors}")
        print(f"  Ignorées (crash adverse): {ignored}")
        print(f"  Taux de victoire: {win_rate:.1f}% (sur {counted_games} parties comptées)")
        
        total_games += counted_games
        total_wins += wins
        total_ignored += ignored
    
    # Statistiques globales
    overall_win_rate = (total_wins / total_games) * 100 if total_games > 0 else 0
    print(f"\n🏆 RÉSULTATS GLOBAUX")
    print("=" * 30)
    print(f"Total parties comptées: {total_games}")
    print(f"Total victoires: {total_wins}")
    print(f"Parties ignorées (crash adverse): {total_ignored}")
    print(f"Taux de victoire global: {overall_win_rate:.1f}%")
    
    # Évaluation
    if overall_win_rate >= 80:
        print("🌟 Excellent ! Votre agent est très performant.")
    elif overall_win_rate >= 60:
        print("👍 Bon travail ! Votre agent est solide.")
    elif overall_win_rate >= 40:
        print("📈 Correct. Il y a de la place pour l'amélioration.")
    else:
        print("🔧 Votre agent a besoin d'améliorations importantes.")

def analyze_single_game(opponent="random_player_hex.py"):
    """
    Analyse une partie unique en détail.
    """
    print(f"\n🔍 Analyse détaillée d'une partie contre {opponent}")
    print("-" * 50)
    
    winner, log = run_game("my_player.py", opponent)
    
    print(f"Gagnant: {winner}")
    
    # Extraire des statistiques du log
    move_times = re.findall(r'time : (\d+(?:\.\d+)?)', log)
    if move_times:
        move_times = [float(t) for t in move_times]
        remaining_times = [900 - (900 - t) for t in move_times[::2]]  # Temps restant pour notre agent
        
        print(f"Nombre de coups: {len(move_times) // 2}")
        print(f"Temps moyen par coup: {(900 - move_times[-2]) / (len(move_times) // 2):.3f}s")
        print(f"Temps total utilisé: {900 - move_times[-2]:.3f}s")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        opponent = sys.argv[2] if len(sys.argv) > 2 else "random_player_hex.py"
        analyze_single_game(opponent)
    else:
        test_agent_performance()
