## 1) Scope & Assumptions

### 1.1 Scope

This README describes how to set up a reproducible Conda/Python environment and workspace on **CSCS bristen**, targeting **interactive Slurm GPU sessions** for TTS benchmarking and data generation.

### 1.2 Included

- Miniconda/Conda setup on bristen (x86)
- Common CSCS-specific issues (e.g., Conda Terms of Service, architecture selection)
- Recommended workspace layout using `$HOME` (code) and `$SCRATCH` (large artifacts)
- Basic Slurm interactive GPU usage and sanity checks

### 1.3 Not included

- Model-specific installation and runtime details (see model docs such as `docs/chattts.md`, `docs/fishspeech.md`)
- Long-running production job submission and scheduling strategy (see `docs/benchmark.md` and/or `run/benchmark` docs)

### 1.4 Assumptions / Prerequisites

Before following this guide, you should have:

- SSH access to **CSCS bristen**
- A valid CSCS account and knowledge of your project allocation/account (e.g., `infra01`)
- A working `$SCRATCH` directory available on the system

---

## 2) Quickstart

This section provides the shortest path to a working interactive GPU environment on **CSCS bristen**. It is intentionally minimal; detailed explanations are in later sections.

### 2.1 Connect to CSCS (SSH)

Use your existing CSCS SSH setup. The exact key name, username, and host may differ.

```bash
# (Optional) ensure your private key has correct permissions
chmod 600 ~/.ssh/<your_cscs_key>

# (Optional) if your key is passphrase-protected and you want to cache it temporarily
ssh-add -t <duration> ~/.ssh/<your_cscs_key>

# Connect to CSCS login / gateway host (example)
ssh -A <cscs_username>@<cscs_login_host>
````

### 2.2 Start an interactive GPU shell (Slurm)

Request a single GPU interactively (account/memory/walltime may vary):

```bash
srun -A <project_account> --gres=gpu:1 --mem=<memory> --time=<walltime> --pty bash
```

### 2.3 Load Conda and activate your environment

Inside the allocated GPU shell:

```bash
# If your .bashrc already initializes conda, this may be enough:
source ~/.bashrc

# Otherwise, explicitly load conda (x86 example):
export PATH=/users/$USER/miniconda3_x86/bin:$PATH
source /users/$USER/miniconda3_x86/etc/profile.d/conda.sh

# Activate your project environment (name may differ)
conda activate <your_env>
```

### 2.4 Sanity checks

Verify that the GPU is visible and Python works:

```bash
nvidia-smi
python -c "print('python ok')"
```

If you have already installed the QC stack:

```bash
python -c "import whisper, sentence_transformers, librosa, soundfile; print('imports ok')"
```

### 2.5 Go to your project workspace

```bash
cd ~/<your_path>/<model_or_repo_dir>
ls
```

---

## 3) Conda Installation & Architecture Switch

This section installs Miniconda (x86) on bristen and ensures the correct conda is loaded depending on node architecture.

### 3.1 Install Miniconda (x86)

Download and install Miniconda:

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

When prompted for the install path, set:

```text
/users/$USER/miniconda3_x86
```

When asked whether to initialize conda automatically, choose **no** (we will manage shell init explicitly).

(Optional) remove the installer after successful installation:

```bash
rm -f ~/Miniconda3-latest-Linux-x86_64.sh
```

### 3.2 Configure `.bashrc` to select conda by architecture

Back up your `.bashrc` first:

```bash
cp ~/.bashrc ~/.bashrc.bak.$(date +%Y%m%d_%H%M%S)
```

Append the following block to `~/.bashrc`. It selects the correct conda installation based on `uname -m`:

```bash
# Conda init with architecture switch (aarch64 vs x86_64)
if [[ $(uname -m) == "aarch64" ]]; then
    # ARM conda (example path; adjust if you have a separate ARM install)
    __conda_setup="$('/users/$USER/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/users/$USER/miniconda3/etc/profile.d/conda.sh" ]; then
            . "/users/$USER/miniconda3/etc/profile.d/conda.sh"
        else
            export PATH="/users/$USER/miniconda3/bin:$PATH"
        fi
    fi
    unset __conda_setup
else
    # x86 conda
    __conda_setup="$('/users/$USER/miniconda3_x86/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/users/$USER/miniconda3_x86/etc/profile.d/conda.sh" ]; then
            . "/users/$USER/miniconda3_x86/etc/profile.d/conda.sh"
        else
            export PATH="/users/$USER/miniconda3_x86/bin:$PATH"
        fi
    fi
    unset __conda_setup
fi
```

Reload your shell:

```bash
source ~/.bashrc
```

Verify the conda you are using:

```bash
uname -m
which conda
conda --version
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
```

### 3.3 Disable auto-activating `base` (recommended)

To avoid automatically entering `base` every time you log in:

```bash
conda config --set auto_activate_base false
```

If you previously added `conda activate` to your `.bashrc`, remove it (or replace it with your target env once it exists).

### Why the architecture switch is necessary

On CSCS systems it is possible to encounter different node architectures (or legacy environment remnants). Selecting conda by `uname -m` reduces the risk of loading the wrong conda installation and avoids hard-to-debug dependency issues.

---

## 4) Conda ToS & Channels (CSCS-specific)

On CSCS, conda may require explicit acceptance of the Terms of Service for the default channels. If you encounter errors during `conda create` / `conda install`, handle them as follows.

### 4.1 Symptom: `CondaToSNonInteractiveError`

You may see an error similar to:

```text
CondaToSNonInteractiveError: ...
```

### 4.2 Fix: accept ToS for the default channels

Run the following commands once:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 4.3 Recommendation

Treat this as a one-time setup step. You typically only need to do it the first time conda complains about ToS.

---

## 5) Slurm GPU Usage & Sanity Checks

This section covers how to request an interactive GPU shell on bristen and perform minimal checks to confirm the environment is usable.

### 5.1 Start an interactive GPU session

From a login node:

```bash
srun -A <project_account> --gres=gpu:1 --mem=<memory> --time=<walltime> --pty bash
```

### 5.2 Initialize your environment inside the GPU shell

After you land on the allocated node, always re-load your shell init and activate your environment:

```bash
source ~/.bashrc
conda activate <your_env>
```

### 5.3 GPU visibility check (recommended)

```bash
nvidia-smi
```

### 5.4 Python / CUDA sanity check (optional)

Whether this works depends on whether you installed PyTorch/CUDA in your env:

```bash
python -c "import torch; print('cuda?', torch.cuda.is_available()); print('torch', torch.__version__)"
```

### 5.5 Common pitfalls

* Do not install heavy dependencies or run inference on the login node. Use an allocated Slurm session (interactive or batch), otherwise you may hit performance throttling, permission limits, or policy enforcement.
* If `conda activate` fails inside the GPU session, re-run `source ~/.bashrc` and confirm `which conda` points to the intended installation.

