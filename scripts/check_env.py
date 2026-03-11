"""
Script : check_env
Rôle : Vérifier que l’environnement est correct (venv actif, Python 3.10+, dépendances, CUDA si disponible).

Structure :
-----------
1. Imports : sys, subprocess ou importlib pour vérifier les modules.
2. Vérifier la version de Python (>= 3.10).
3. Vérifier que l’exécution se fait bien dans un venv (sys.prefix != sys.base_prefix ou présence de VIRTUAL_ENV).
4. Vérifier les packages : numpy, pandas, scipy, sklearn, yaml, tqdm ; optionnel torch.
5. Si torch présent : afficher torch.__version__ et torch.cuda.is_available() (et device count) pour informer l’utilisateur (CUDA utilisé ou non).
6. Afficher un résumé lisible (OK / avertissements) en français.
"""

import sys

def main():
    print("=== Vérification de l'environnement (INF6083 P2) ===\n")
    # Version Python
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"  Python : {sys.version.split()[0]} (OK)")
    else:
        print(f"  Python : {sys.version.split()[0]} (recommandé : 3.10+)")
    # Venv
    in_venv = getattr(sys, "prefix", None) != getattr(sys, "base_prefix", None) or bool(__import__("os").environ.get("VIRTUAL_ENV"))
    print(f"  Venv actif : {'Oui' if in_venv else 'Non (recommandé : activer .venv)'}")
    # Dépendances
    deps = ["numpy", "pandas", "scipy", "sklearn", "yaml", "tqdm"]
    for name in deps:
        mod = "sklearn" if name == "sklearn" else name
        try:
            __import__(mod)
            print(f"  {name} : OK")
        except ImportError:
            print(f"  {name} : manquant")
    try:
        import torch
        print(f"  torch : {torch.__version__}")
        print(f"  CUDA disponible : {torch.cuda.is_available()} ({torch.cuda.device_count()} device(s))")
    except ImportError:
        print("  torch : non installé (optionnel)")
    print("\n=== Fin de la vérification ===")

if __name__ == "__main__":
    main()
