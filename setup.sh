#!/usr/bin/env bash
# Old name of install.sh.
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/install.sh" "$@"
