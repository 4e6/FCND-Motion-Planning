with import <nixpkgs> {};

stdenv.mkDerivation {
  name = "motion-planning-env";
  buildInputs = [
    # these packages are required for virtualenv and pip to work:
    python3Full
    python3Packages.virtualenv
    python3Packages.pip
    # python libs
    python3Packages.numpy
    # udacidrone deps
    python3Packages.lxml
    # system dependencies
    git
    stdenv
    which
  ];

  src = null;

  LANG = "en_US.UTF-8";

  shellHook = ''
    # set SOURCE_DATE_EPOCH so that we can use python wheels
    export SOURCE_DATE_EPOCH=$(date +%s)
    virtualenv --no-setuptools venv
    export PATH=$PWD/venv/bin:$PATH
    export PYTHONPATH=venv/lib/python3.6/site-packages/:$PYTHONPATH
    pip install -r requirements.txt
  '';
}
