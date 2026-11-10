#!/usr/bin/env bash
# Load .gp5 files with TuxGuitar's own reader (from the Flatpak install):
#   eval/tuxcheck.sh "Tabs/Song/Song.gp5"
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tg="$(flatpak info --show-location ar.com.tuxguitar.TuxGuitar)/files"
cp="$tg/lib/tuxguitar-lib.jar:$tg/share/plugins/tuxguitar-gtp.jar"
build="${TMPDIR:-/tmp}/tuxcheck-$UID"
mkdir -p "$build"
javac --release 17 -nowarn -cp "$cp" -d "$build" "$here/tuxcheck/TuxCheck.java"
exec "$tg/jre/bin/java" -cp "$cp:$build" TuxCheck "$@"
