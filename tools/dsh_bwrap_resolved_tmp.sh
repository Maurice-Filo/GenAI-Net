#!/usr/bin/env bash
set -euo pipefail

# DSH rc.8 mounts a private tmpfs at /tmp. On this host /tmp is a symlink, so
# Bubblewrap needs the resolved mount point while sandboxed processes can keep
# using /tmp through that symlink.
resolved_tmp="$(readlink -f /tmp)"
args=("$@")

for ((index = 0; index + 1 < ${#args[@]}; index++)); do
    if [[ "${args[index]}" == "--tmpfs" && "${args[index + 1]}" == "/tmp" ]]; then
        args[index + 1]="$resolved_tmp"
    fi
done

exec bwrap "${args[@]}"
