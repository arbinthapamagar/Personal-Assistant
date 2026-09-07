#!/usr/bin/env bash
# Install the personal assistant on this machine.
#
#   git clone <repo> && cd <repo> && ./install.sh
# or, without cloning first:
#   curl -fsSL https://raw.githubusercontent.com/<you>/<repo>/main/install.sh | bash
#
# Creates a private virtualenv (so PEP 668 / "externally managed environment"
# on Debian, Ubuntu, and Fedora never blocks the install), links a `pa` binary
# onto PATH, and leaves the heavy optional packages for first use.

set -euo pipefail

REPO_URL="${PA_REPO_URL:-https://github.com/CHANGEME/personal-assistant.git}"
PREFIX="${PA_PREFIX:-$HOME/.local/share/personal-assistant}"
BIN_DIR="${PA_BIN_DIR:-$HOME/.local/bin}"
VENV="$PREFIX/venv"
EXTRAS="${PA_EXTRAS:-}"

info()  { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn()  { printf '\033[33m!!\033[0m  %s\n' "$*" >&2; }
die()   { printf '\033[31mxx\033[0m  %s\n' "$*" >&2; exit 1; }

# --- 1. find a usable Python ------------------------------------------------
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
      PYTHON="$candidate"; break
    fi
  fi
done
[ -n "$PYTHON" ] || die "need Python 3.10 or newer. Install it, e.g.: sudo apt install python3 python3-venv"
info "using $($PYTHON --version) at $(command -v "$PYTHON")"

# venv is a separate package on Debian/Ubuntu and its absence is the single most
# common install failure, so check for it explicitly rather than failing later.
if ! "$PYTHON" -c 'import venv, ensurepip' 2>/dev/null; then
  die "Python venv support is missing. Install it:
  Debian/Ubuntu:  sudo apt install python3-venv python3-pip
  Fedora:         sudo dnf install python3-pip
  Arch:           sudo pacman -S python-pip"
fi

# --- 2. get the source ------------------------------------------------------
if [ -f "pyproject.toml" ] && grep -q 'name = "personal-assistant"' pyproject.toml 2>/dev/null; then
  SRC="$(pwd)"
  info "installing from the current checkout: $SRC"
else
  command -v git >/dev/null 2>&1 || die "git is required to fetch the source"
  SRC="$PREFIX/src"
  if [ -d "$SRC/.git" ]; then
    info "updating existing checkout at $SRC"
    git -C "$SRC" pull --ff-only
  else
    case "$REPO_URL" in
      *CHANGEME*) die "set the repository first: PA_REPO_URL=https://github.com/you/repo.git bash install.sh" ;;
    esac
    info "cloning $REPO_URL"
    mkdir -p "$PREFIX"
    git clone --depth 1 "$REPO_URL" "$SRC"
  fi
fi

# --- 3. build the virtualenv ------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  info "creating virtualenv at $VENV"
  mkdir -p "$PREFIX"
  "$PYTHON" -m venv "$VENV"
fi
info "installing the package"
"$VENV/bin/python" -m pip install --quiet --upgrade pip setuptools wheel
if [ -n "$EXTRAS" ]; then
  info "including extras: $EXTRAS"
  "$VENV/bin/python" -m pip install --quiet -e "$SRC[$EXTRAS]"
else
  "$VENV/bin/python" -m pip install --quiet -e "$SRC"
fi

# --- 4. put `pa` on PATH ----------------------------------------------------
mkdir -p "$BIN_DIR"
ln -sf "$VENV/bin/pa" "$BIN_DIR/pa"
info "linked $BIN_DIR/pa"

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *)
    warn "$BIN_DIR is not on your PATH. Add it:"
    printf '\n    echo '\''export PATH="%s:$PATH"'\'' >> ~/.bashrc && exec bash\n\n' "$BIN_DIR"
    ;;
esac

# --- 5. optional system packages (best-effort, never fatal) -----------------
# These make desktop/browser control and semantic memory work out of the box.
# Skipped silently if apt is absent or the user cannot sudo.
if command -v apt-get >/dev/null 2>&1 && [ "${PA_SKIP_APT:-}" != "1" ]; then
  MISSING=""
  for pkg in xdotool wmctrl ydotool gnome-screenshot ripgrep redis-server alsa-utils espeak-ng; do
    dpkg -s "$pkg" >/dev/null 2>&1 || MISSING="$MISSING $pkg"
  done
  if [ -n "$MISSING" ]; then
    warn "optional system packages not installed:$MISSING"
    warn "install them for full desktop/browser/search support:"
    printf '\n    sudo apt install%s\n\n' "$MISSING"
  fi
fi

# --- 6. report what this machine can do -------------------------------------
info "checking the environment"
"$VENV/bin/pa" --doctor || true

cat <<'NEXT'

Next steps
  1. Give it a model. Any one of these is enough:
       export ANTHROPIC_API_KEY=sk-ant-...      # then: pa
       export OPENAI_API_KEY=sk-...             # then: pa -p gpt
       export GEMINI_API_KEY=...                # then: pa -p gemini
       ollama serve                             # then: pa -p local   (no key, fully offline)
  2. Write a config if you want to change defaults:
       pa           then type   /config
  3. Try it:
       pa "what is using the most disk space in my home directory?"

Optional extras, installed on demand the first time you use them, or up front:
  PA_EXTRAS=all bash install.sh
NEXT
