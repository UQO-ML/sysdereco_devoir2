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
import subprocess
import os
import platform



def main():
    print("=== Vérification de l'environnement (INF6083 P2) ===\n")
    # Plateforme et OS
    os_name, rapids_capable, rapids_msg = _platform_info()
    print(f"  Plateforme : {os_name}")
    # Version Python
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10 and version.minor < 14:
        print(f"  Python : {sys.version.split()[0]} (OK)")
    else:
        print(f"  Python : {sys.version.split()[0]} (recommandé : 3.10+ <= 3.13 )")
    # Venv
    in_venv = getattr(sys, "prefix", None) != getattr(sys, "base_prefix", None) or bool(__import__("os").environ.get("VIRTUAL_ENV"))
    print(f"  Venv actif : {'Oui' if in_venv else 'Non (recommandé : activer .venv)'}")
    # Dépendances
    deps = ["numpy", "pandas", "scipy", "sklearn", "yaml", "tqdm", "polars"]
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
        i = 0
        while (i < torch.cuda.device_count()):
            print(f"  Device n°{i} : {torch.cuda.get_device_name(i)}")
            i += 1
        # GPU présent : vérifier OS et éventuellement installer RAPIDS
        if torch.cuda.device_count() > 0:
            # RAPIDS officiellement supporté surtout sous Linux (et WSL2)
            if rapids_capable:
                print("  OS : Linux (compatible RAPIDS)")
                # Optionnel : installer RAPIDS si pas déjà installés
                try:
                    import cudf
                    import cupy
                    import rmm
                    print("  RAPIDS (cudf, cupy, rmm) : déjà installés")
                except ImportError:
                    req_path = os.path.join(os.path.dirname(__file__), "..", "requirements-rapids.txt")
                    if os.path.isfile(req_path):
                        print("  RAPIDS : manquants, installation...")
                        r = subprocess.run(
                            [sys.executable, "-m", "pip", "install", "-r", os.path.normpath(req_path)],
                            cwd=os.path.normpath(os.path.join(os.path.dirname(__file__), "..")),
                        )
                        if r.returncode == 0:
                            print("  RAPIDS : installation terminée")
                        else:
                            print("  RAPIDS : échec pip install (vérifier CUDA 12 et requirements-rapids.txt)")
                    else:
                        print("  RAPIDS : manquants (créer requirements-rapids.txt pour installation auto)")
            else:
                print(f"  OS : {rapids_msg}")
    except ImportError:
        print("  torch : non installé (optionnel)")

    print("\n=== Fin de la vérification ===")


def _platform_info():
    """Retourne (os_name, is_rapids_capable, message)."""
    plat = platform.system()
    if plat == "Linux":
        return "Linux", True, "Compatible RAPIDS (NVIDIA GPU requis)"
    if plat == "Darwin":
        return "macOS (Apple Silicon possible)", False, "RAPIDS non disponible - utiliser mode CPU (pandas) ou échantillons pré-générés"
    if plat == "Windows":
        return "Windows", False, "RAPIDS non disponible en natif - utiliser mode CPU ou WSL2 + NVIDIA pour RAPIDS"
    return plat, False, "Plateforme non testée"

if __name__ == "__main__":
    main()
