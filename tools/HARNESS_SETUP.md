# DeepSeek Harness: installation and LLM connection

This guide sets up the **local Harness runtime** used by GenAI-Net and connects
it either to hosted DeepSeek or to a local OpenAI-compatible model server.
The Python client starts Harness automatically for each Decider/Writer role;
you do **not** need to leave the Harness web UI running. A local model server,
if used, must remain running separately.

These instructions target **Linux with Bash**. They use the paper's pinned
Harness `@deepseek-ai/dsh@0.1.0-rc.8`, rather than the latest release. The
recorded installation used Node 24.19.0 and pnpm 11.7.0. Native Windows/macOS
execution is not covered: our integration uses Linux Bubblewrap and POSIX locks.

## Quick installation and GUI

From the repository root on Linux, run:

```bash
bash tools/install_harness.sh
```

The script downloads the pinned Harness, installs Node 24 locally if needed,
initializes the GUI and headless profiles, and sets Flash v4 with high reasoning
as the default. It optionally asks for your DeepSeek API key with hidden input.
Press **Enter** to skip it and configure the provider credential in the GUI
instead. Existing configuration and credentials are preserved on subsequent runs.
The script needs `python3`, `curl`, `tar`, `xz` and `sha256sum`; install these with
your distribution's package manager if missing. Bubblewrap is needed for agent
tools, as described below.

The GUI opens at **http://127.0.0.1:3080** and stays in the foreground; **Ctrl-C**
stops it. Reopen it later with:

```bash
"$HOME/.local/share/genainet/bin/harness-gui"
```

In a separate terminal, load the environment for the Python integration:

```bash
source "$HOME/.local/share/genainet/env.sh"
```

With a custom `--prefix`, use that directory's `bin/harness-gui` and `env.sh`
instead; the installer prints the exact commands when it finishes.

For a remote machine, start with `bash tools/install_harness.sh --no-open`, then
forward the GUI port from your laptop:

```bash
ssh -N -L 3080:127.0.0.1:3080 your-user@your-server
```

Open `http://127.0.0.1:3080` on the laptop. The GUI listens on loopback only.
Use `--port 3081` if the default port is occupied, and adjust the tunnel accordingly.

Other useful options:

```bash
# Set up now, enter credentials and open the GUI later.
bash tools/install_harness.sh --no-gui --skip-api-key

# Choose installation and configuration locations on a different disk.
bash tools/install_harness.sh --prefix /path/to/harness-install \
  --config-dir /path/to/harness-config --no-open

bash tools/install_harness.sh --help
```

The installer creates a private `workspace` inside the configuration directory
for GUI sessions. Keys stay in that directory's `.credentials.yaml`, with
owner-only access, and are never embedded in the generated launchers. GUI model
and credential changes are shared through this Harness home; GenAI-Net's Python
client still selects its model and provider explicitly. For local inference,
start an OpenAI-compatible model server separately and follow section 4B.

The installer handles the Harness only. For numerical evaluation, prepare the
Python environment in section 1, source the generated `env.sh`, and continue to
the offline/live smoke tests in section 4A or the local endpoint in section 4B.
**Skip the manual installation, profile initialization and hosted credential/config
creation blocks in sections 2–4A when using the installer.** Those blocks below
are an alternative manual setup. Keep the runtime's `pnpm-lock.yaml` to reproduce
its resolved dependencies. The installer uses the tested dependency graph in
[`harness-runtime/`](harness-runtime/): pinning only Harness rc.8 can resolve
newer Cordis dependencies that fail during GUI startup. The bundled build policy
enables the required native helpers explicitly. If prebuilt binaries are unavailable
for your platform, install a C/C++ toolchain (`build-essential` on Debian/Ubuntu).

The smoke test below runs one RPA Decider/Writer batch and evaluates its output.
It does not train RL, launch a campaign, run Julia/AdaGrad, or upload to Comet.
The installer and offline smoke test make no model requests. The smoke test's
`--live` option and messages sent in the GUI do; hosted requests are billable.

## 1. Prepare the repository and Python environment

Open a Bash terminal in your checkout of GenAI-Net. Use the same terminal for
the following sections, so the exported paths remain available.

```bash
export GENAI_ROOT="$PWD"
test -f "$GENAI_ROOT/RL4CRN/llm/harness_client.py"

# Debian/Ubuntu; use your distribution's equivalents on other Linux systems.
sudo apt-get update
sudo apt-get install -y git curl xz-utils bubblewrap python3-venv build-essential

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/genainet-mpl"
chmod u+x tools/dsh_bwrap_resolved_tmp.sh
```

Python 3.10.12 is used by the current working environment. The Python
installation above resolves dependencies
from `setup.py`; it is a setup recipe, not a complete reproduction lockfile.
A GPU is unnecessary for the hosted-model smoke test and its CVODE evaluation.

Check that Bubblewrap can create a sandbox on this machine:

```bash
bwrap --ro-bind / / --dev /dev --proc /proc --unshare-user --unshare-pid /bin/true
```

An `Operation not permitted` error here means that the host/container's namespace
policy needs to allow Bubblewrap. Resolve this with the machine administrator
before running the live test; do not disable the integration's sandbox.

## 2. Install Node and the pinned Harness

If Node 24 is already on `PATH`, skip the optional Node installation block.
Otherwise, this installs the tested version in your own home directory without
changing the system Node installation:

```bash
export GENAI_NODE_VERSION=v24.19.0
case "$(uname -m)" in
  x86_64) GENAI_NODE_ARCH=x64 ;;
  aarch64|arm64) GENAI_NODE_ARCH=arm64 ;;
  *) echo "Use a Node 24 build for your architecture"; return 1 2>/dev/null || exit 1 ;;
esac
export GENAI_NODE_DIR="$HOME/.local/share/genainet/node-${GENAI_NODE_VERSION}-linux-${GENAI_NODE_ARCH}"
GENAI_NODE_DOWNLOAD="$(mktemp -d)"
curl -fL "https://nodejs.org/dist/${GENAI_NODE_VERSION}/node-${GENAI_NODE_VERSION}-linux-${GENAI_NODE_ARCH}.tar.xz" \
  -o "$GENAI_NODE_DOWNLOAD/node.tar.xz"
curl -fL "https://nodejs.org/dist/${GENAI_NODE_VERSION}/SHASUMS256.txt" \
  -o "$GENAI_NODE_DOWNLOAD/SHASUMS256.txt"
GENAI_NODE_SHA="$(awk -v name="node-${GENAI_NODE_VERSION}-linux-${GENAI_NODE_ARCH}.tar.xz" '$2 == name {print $1}' "$GENAI_NODE_DOWNLOAD/SHASUMS256.txt")"
printf '%s  %s\n' "$GENAI_NODE_SHA" "$GENAI_NODE_DOWNLOAD/node.tar.xz" | sha256sum --check
mkdir -p "$GENAI_NODE_DIR"
tar -xJf "$GENAI_NODE_DOWNLOAD/node.tar.xz" --strip-components=1 -C "$GENAI_NODE_DIR"
export PATH="$GENAI_NODE_DIR/bin:$PATH"
```

Install the package manager and Harness once, outside the repository:

```bash
node --version
export GENAI_NODE_TOOLS="$HOME/.local/share/genainet/node-tools"
export GENAI_DSH_RUNTIME="$HOME/.local/share/genainet/dsh-runtime"
npm install --prefix "$GENAI_NODE_TOOLS" --save-exact pnpm@11.7.0
export PATH="$GENAI_NODE_TOOLS/node_modules/.bin:$PATH"
mkdir -p "$GENAI_DSH_RUNTIME"
# For a fresh runtime only; preserve an existing installation's files.
for GENAI_RUNTIME_FILE in package.json pnpm-lock.yaml pnpm-workspace.yaml; do
  if test ! -e "$GENAI_DSH_RUNTIME/$GENAI_RUNTIME_FILE"; then
    cp "tools/harness-runtime/$GENAI_RUNTIME_FILE" "$GENAI_DSH_RUNTIME/$GENAI_RUNTIME_FILE"
  fi
done
pnpm --dir "$GENAI_DSH_RUNTIME" install --frozen-lockfile

export GENAI_DSH_CLI="$GENAI_DSH_RUNTIME/node_modules/@deepseek-ai/dsh/lib/bin.js"
node "$GENAI_DSH_CLI" --help
node -e 'console.log(JSON.parse(require("fs").readFileSync(process.env.GENAI_DSH_RUNTIME + "/node_modules/@deepseek-ai/dsh/package.json")).version)'

# Explicitly select this installation, including when another dsh is on PATH.
export RL4CRN_DSH_COMMAND="$(python - <<'PY'
import os, shlex, shutil
print(shlex.join([shutil.which('node'), os.environ['GENAI_DSH_CLI']]))
PY
)"
```

Keep `package.json`, `pnpm-lock.yaml` and `pnpm-workspace.yaml` in the runtime
directory. Reuse the lockfile with
`pnpm --dir "$GENAI_DSH_RUNTIME" install --frozen-lockfile` when reproducing
that installation. Avoid `npx`/`pnpm dlx` launches per request.

## 3. Initialize a separate Harness home

Use a dedicated configuration directory for these experiments. The fresh profile
avoids importing unrelated settings from an existing interactive Harness setup.

```bash
export GENAI_DSH_HOME="$HOME/.config/genainet-dsh"
mkdir -p "$GENAI_DSH_HOME"
chmod 700 "$GENAI_DSH_HOME"
export DSH_HOME="$GENAI_DSH_HOME"
export CHOKIDAR_USEPOLLING=1
export CHOKIDAR_INTERVAL=1000

# Initializes the profile and prints configuration without starting a model request.
node "$GENAI_DSH_CLI" --profile headless --dump-config > /tmp/genainet-dsh-base-config.yml
test -f "$GENAI_DSH_HOME/profiles/headless/package.json"
```

Our client generates the additional `harness.patch.yml` for every request. It
loads the repository's `tools/dsh_headless_continuing_runner.mjs`, pins the
provider/model, applies the role instructions and selects the Bubblewrap wrapper.
Do not copy those files to a machine-specific installation path: keep them in the
repository with the Python client.

## 4A. Connect hosted DeepSeek

Skip this section if you only want a local model. Enter your own DeepSeek API key
at the hidden prompt below. This creates the file-backed credential store read by
Harness. It refuses to overwrite an existing credentials file.

```bash
python - <<'PY'
import getpass, json, os
from pathlib import Path
path = Path(os.environ['GENAI_DSH_HOME']) / '.credentials.yaml'
if path.exists():
    raise SystemExit(f'{path} already exists; retain or edit it rather than overwriting it.')
key = getpass.getpass('DeepSeek API key: ').strip()
if not key:
    raise SystemExit('No key supplied.')
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as f:
    f.write('DEEPSEEK_API_KEY: ' + json.dumps(key) + '\n')
print('Credential stored; its value was not printed.')
PY
```

Do not put the key in the repository or in a generated request patch. Exporting
`DEEPSEEK_API_KEY` alone is insufficient for our Python client: its child process
environment uses an allowlist. The dedicated Harness credential file works
without changing that allowlist.

Make the Flash settings explicit. This also refuses to overwrite existing user
configuration; inspect an existing file before changing it.

```bash
python - <<'PY'
import os
from pathlib import Path
path = Path(os.environ['GENAI_DSH_HOME']) / 'cordis.patch.yml'
with path.open('x') as f:
    f.write('''- id: llm-deepseek
  config:
    apiKeyEnv: DEEPSEEK_API_KEY
    baseURL: https://api.deepseek.com
    thinking: enabled
    reasoningEffort: high
    maxTokens: 256000
''')
print('Hosted-provider settings written.')
PY
```

Run the **offline** integration check:

```bash
python tools/check_harness_setup.py --dsh-home "$GENAI_DSH_HOME" --backend deepseek
```

It checks the Python/evaluator setup and writes a fresh temporary output directory
containing `setup.json` and `harness.patch.yml`. Run the printed `--dump-config`
command if you also want to check that Harness can compose the generated patch.
Neither step verifies the API key or makes a model request.

Then explicitly run one **live** Decider/Writer batch:

```bash
python tools/check_harness_setup.py \
  --dsh-home "$GENAI_DSH_HOME" --backend deepseek \
  --model deepseek-v4-flash --candidates 1 --role-timeout 900 --live
```

This sends the task, reaction library and role instructions to DeepSeek. One role
invocation may contain several model/tool turns, so one candidate is not necessarily
one API call. Use `--candidates 10` for the campaign's batch size after setup works.

## 4B. Connect a local model server

The Harness runs on the search machine. The inference server can run there or on
a GPU machine reached through an SSH tunnel. It must provide an OpenAI-compatible
chat-completions endpoint and support the model's tool-call format. A successful
plain-text completion alone does not establish compatibility with the agent tools.

Start/configure your inference server separately; see the
[vLLM serving documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/).
Use the exact served model ID shown by its `/v1/models` endpoint.

For a server already listening on the GPU machine's loopback port 8000, keep this
command running in a **second terminal** (replace the SSH destination):

```bash
ssh -N -L 18081:127.0.0.1:8000 YOUR_USER@YOUR_GPU_HOST
```

Back in the setup terminal:

```bash
export GENAI_LLM_URL=http://127.0.0.1:18081/v1
curl --fail --silent --show-error "$GENAI_LLM_URL/models" | python -m json.tool
export GENAI_LLM_MODEL='REPLACE_WITH_THE_SERVED_MODEL_ID'
```

If the server runs on the same machine, use its local port directly. The current
client accepts only `localhost`, `127.0.0.1` or `::1` endpoints. It sends a dummy
API-key-shaped value (`local-loopback-no-auth`), suitable for an unauthenticated
loopback server. For an authenticated endpoint, the Python client needs the actual
key passed as `extra_environment={'DSH_LOCAL_LLM_API_KEY': ...}`; do not embed that
key in the README, command line or request patch. The smoke-test CLI below assumes
the unauthenticated loopback/tunnel setup.

Use limits supported by your server. The following smaller values are for a
connection test, **not** the paper's Qwen deployment configuration:

```bash
python tools/check_harness_setup.py \
  --dsh-home "$GENAI_DSH_HOME" --backend local \
  --endpoint "$GENAI_LLM_URL" --model "$GENAI_LLM_MODEL" \
  --context-window 32768 --max-output-tokens 4096 --reasoning high

python tools/check_harness_setup.py \
  --dsh-home "$GENAI_DSH_HOME" --backend local \
  --endpoint "$GENAI_LLM_URL" --model "$GENAI_LLM_MODEL" \
  --context-window 32768 --max-output-tokens 4096 --reasoning high \
  --candidates 1 --role-timeout 3600 --live
```

The local adapter uses the Qwen chat-template thinking format. Other model
families may require changes to the provider's compatibility settings. A 4,096-token
cap may be too small for a reasoning model to finish a role: increase it within
the server's capacity if the logs show token exhaustion. The input transcript and
reserved output must fit together inside the context window.

The paper's Qwen endpoint used a 262,144-token context, a 131,072-token output cap,
and high reasoning; the exact model/GPU/vLLM configuration is documented in the
manuscript. In the production client the generic local path defaults to low
reasoning. This smoke helper explicitly applies the same high-reasoning override
as the matched Qwen campaign when `--reasoning high` is selected.

## 5. Check the outcome

Every invocation prints its output directory. `--output /path/to/new-directory`
can select a persistent destination; existing directories are rejected.

- `setup.json`: selected executable, provider/model and connection settings.
- `harness.patch.yml`: generated configuration for offline inspection.
- `summary.json`: live test completion, role count, candidate counts and losses.
- `evaluations.jsonl`: canonical candidate evaluations.
- `workspaces/`: task/role inputs, returned designs, per-call logs, model-route
  records and validation results. The printed `workspace` identifies the batch.
- `failure.json`: failure type and workspace location if the live batch raises.

A working batch normally has two role invocations, a returned candidate and a
valid numerical evaluation. A high loss is not a connection failure. Invalid
proposals can also occur despite a working connection; inspect validation output
rather than assuming that every batch must yield a qualifying design.

## 6. Connect this setup to numerical search

Keep `RL4CRN_DSH_COMMAND` and the Node/pnpm directories on `PATH`, and pass
`--dsh-home "$GENAI_DSH_HOME"` to a runner that accepts that option. The hosted RPA
runner is `comparisons/search_exploration/run_harness_streams.py`; its help lists
the task, search and Julia configuration arguments:

```bash
python -m comparisons.search_exploration.run_harness_streams --help
```

That runner is configured for hosted Flash; changing the smoke test's backend
does **not** change it to Qwen. For a local deployment, configure
`HarnessJuliaStreamTrainer.attach_harness()` with the selected provider, model,
`openai_compatible_base_url`, context/output limits and reasoning override.
The paper's `local_qwen_matched_campaign.py` records how this was done, but its
campaign paths are machine-specific and it is not a portable one-command launcher.
Julia/SynthEvo and numerical campaign setup are separate from this Harness guide.

## Troubleshooting

| Symptom | Check |
|---|---|
| `node` or `pnpm` missing | Restore the paths from Section 2 in this shell. |
| Wrong Harness version or plugin/API mismatch | Inspect the installed version and lockfile; ensure `RL4CRN_DSH_COMMAND` selects the pinned installation. |
| Missing API credential / HTTP 401 | Check the chosen Harness home and credential-file permissions (`chmod 600`); do not print the key. |
| HTTP 402 / insufficient balance | The hosted account needs funds; setup alone does not validate credit. |
| `bwrap: ... Operation not permitted` | Run Section 1's sandbox check; namespace/container restrictions need host-level resolution. |
| `ENOSPC` from file watchers | This can mean exhausted inotify watches, not a full disk. The smoke helper passes `CHOKIDAR_USEPOLLING=1` and `CHOKIDAR_INTERVAL=1000` directly to the child. For other runners, pass these through `extra_environment`; parent exports alone are filtered by the client. |
| Local connection refused | Check the server, SSH tunnel, port, and `/v1/models`. |
| Context-length error | Reduce the output reservation or transcript size, or serve a larger supported context. A longer timeout does not repair it. |
| Role deadline / token exhaustion | Inspect call logs; distinguish a slow response from exhausted output tokens. Increase the relevant limit deliberately. |
| Invalid JSON / rejected candidates | Inspect Writer and validation artifacts; connectivity and scientific proposal validity are separate checks. |

In a new terminal, reactivate `.venv` and source the installer's `env.sh`.
For a manual installation, restore Node/pnpm `PATH` and re-export
`GENAI_DSH_RUNTIME`, `GENAI_DSH_CLI`, `RL4CRN_DSH_COMMAND`, `GENAI_DSH_HOME` and
`DSH_HOME`. The credential file persists; you do not need to re-enter the key.

## Validation and upstream references

The installer was checked on Linux x86_64 with a fresh Node download and a
locked Harness installation in a temporary directory. GUI startup and its HTTP
response, profile initialization, paths containing spaces, argument validation
and preservation of existing settings/credentials were checked. The helper's
offline RPA preparation and generated-patch composition were also checked.
No live model requests were made; those remain for the target-machine smoke test.
The examples do not constitute a complete frozen reproduction environment.

- [DeepSeek Harness upstream](https://github.com/deepseek-ai/deepseek-harness)
- [Node downloads](https://nodejs.org/en/download)
- [Existing integration overview](../comparisons/llm_crn_generation/HARNESS.md)
- [Recorded Flash readiness check](../comparisons/search_exploration/RPA_FLASH_HARNESS_READINESS.md)
- [Python Harness client](../RL4CRN/llm/harness_client.py)
- [Decider/Writer integration](../RL4CRN/llm/harness_runner.py)
