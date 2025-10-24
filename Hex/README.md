# HexMaster - Agent Intelligent pour le Jeu Hex

## 📋 Description

HexMaster est un agent intelligent développé pour jouer au jeu de plateau Hex dans le cadre du projet INF8175. L'agent utilise l'algorithme Minimax avec élagage Alpha-Beta et des heuristiques sophistiquées pour prendre des décisions optimales.

## 🎯 Objectif du Jeu

Le jeu Hex se joue sur un plateau hexagonal où deux joueurs (Rouge et Bleu) tentent de créer un chemin continu :
- **Joueur Rouge** : doit relier le côté supérieur au côté inférieur
- **Joueur Bleu** : doit relier le côté gauche au côté droit

## 🛠️ Installation et Configuration

### Prérequis

- Python 3.11.x ou supérieur
- Environnement virtuel recommandé

### Installation

1. **Créer un environnement virtuel**
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   # ou
   venv\Scripts\activate     # Windows
   ```

2. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

Pour lancer une partie, on peut utiliser les commandes présentées dans l'énoncé:

Partie humain contre humain:
```bash
python main_hex.py -t human_vs_human
```

Pour faire affronter 2 agents:
```bash
python main_hex.py -t local agent1.py agent2.py
```

Un agent aléatoire et un greedy sont fournis:
```bash
python main_hex.py -t local .\random_player_hex.py .\greedy_player_hex.py
```

Pour affronter humain contre agent:
```bash
python main_hex.py -t human_vs_computer agent.py
```

Pour affronter un agent d'un autre groupe:
Sur le PC hébergant le match:
```bash
python main_hex.py -t host_game -a <ip_adress> agent.py
```
Et l'équipe qu'on affronte devra lancer, avec l'IP de l'équipe qui héberge:
```bash
python main_hex.py -t connect -a <ip_adress> agent.py
```

## Règles
Le premier joueur joue les pions rouges, le second les bleus. Le joueur rouge doit relier par un chemin continu le haut et le bas du plateau, tandis que le bleu doit relier la gauche et la droite. 

A chaque tour, un joueur va poser une pièce de sa couleur sur une case vide du plateau.