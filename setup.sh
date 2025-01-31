python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly