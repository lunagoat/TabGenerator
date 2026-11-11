#!/usr/bin/env bash
# Guitar Pro files -> JSON lines, using TuxGuitar's readers (Flatpak install):
#   eval/tabdump.sh out.jsonl file.gp file.gpx ...
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tg="$(flatpak info --show-location ar.com.tuxguitar.TuxGuitar)/files"
cp="$tg/lib/tuxguitar-lib.jar:$tg/lib/commons-compress.jar:$tg/lib/commons-io.jar:$tg/share/plugins/tuxguitar-gtp.jar:$tg/share/plugins/tuxguitar-gpx.jar"
build="${TMPDIR:-/tmp}/tabdump-$UID"
mkdir -p "$build"
javac --release 17 -nowarn -cp "$cp" -d "$build" "$here/tabdump/TabDump.java"
exec "$tg/jre/bin/java" -Xmx4g -cp "$cp:$build" TabDump "$@"
