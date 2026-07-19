# ---------------------------------------------------------------------------
# BASE IMAGE
# ---------------------------------------------------------------------------
# This is the starting point - a pre-built image that already has Python
# 3.11 installed on a minimal Linux system. "slim" means it's a stripped-
# down version (smaller download, faster builds) that still has everything
# Python itself needs - we're not using the full, heavier image since we
# don't need extra OS tools we won't use.
#
# IMPORTANT: this MUST match the Python version your project actually
# needs (3.11), for the exact same reason we had to fix your local
# Python version earlier - the package versions in requirements.txt were
# tested against 3.11, not any other version.
FROM python:3.11-slim

# ---------------------------------------------------------------------------
# WORKING DIRECTORY
# ---------------------------------------------------------------------------
# This sets /app as the "current folder" for every instruction that
# follows - similar to doing `cd /app` inside the container. It also
# creates that folder if it doesn't exist yet. Everything we copy in and
# every command we run happens relative to this folder.
WORKDIR /app

# ---------------------------------------------------------------------------
# INSTALL DEPENDENCIES FIRST (before copying the rest of your code)
# ---------------------------------------------------------------------------
# This ordering is deliberate, not arbitrary. Docker builds images in
# LAYERS, and caches each layer - if a layer's inputs haven't changed
# since the last build, Docker reuses the cached result instead of
# redoing the work. Your requirements.txt changes rarely; your actual
# Python code (agents, graph, etc.) changes constantly while you develop.
# By copying and installing requirements FIRST, Docker can reuse the
# "install everything" layer (which takes minutes) on every rebuild,
# as long as requirements.txt itself hasn't changed - even if you've
# edited your agent code 50 times since. This can turn a multi-minute
# rebuild into a few seconds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# COPY YOUR ACTUAL PROJECT CODE
# ---------------------------------------------------------------------------
# The "." means "everything in the current folder on my machine" and the
# second "." means "into the current WORKDIR inside the container" (/app).
# Combined with .dockerignore, this copies your agents/, graph/,
# mcp_servers/, requirements.txt, etc. - but never venv/ or .env.
COPY . .

# ---------------------------------------------------------------------------
# THE ACTUAL COMMAND THIS CONTAINER RUNS
# ---------------------------------------------------------------------------
# This is what executes when the container starts - identical to you
# typing this command in your own terminal. We use the "-m" module form
# for the exact same reason we used it locally: it ensures Python treats
# /app as the base for resolving your agents.*, graph.*, and
# mcp_servers.* package imports correctly.
CMD ["python", "-m", "graph.run_pipeline"]