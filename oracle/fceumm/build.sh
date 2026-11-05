#!/usr/bin/env bash
# Build the FCEUmm oracle (target/oracle/libtforacle.so) from the pinned upstream commit.
# The user's clone is never modified: the pinned tree is exported with `git archive`.
set -euo pipefail

PIN=e0f3086a8104f8fbbe49a2615b10401504a9533f
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
FCEUMM_DIR="${FCEUMM_DIR:-$ROOT/../libretro-fceumm}"
OUT="$ROOT/target/oracle"
SRC="$OUT/src"

head="$(git -C "$FCEUMM_DIR" rev-parse HEAD)"
if [[ "$head" != "$PIN" ]]; then
    echo "oracle source is not e0f3086 (found $head in $FCEUMM_DIR)" >&2
    exit 1
fi

stamp="$OUT/.stamp"
want="$PIN $(sha1sum "$HERE"/patches/*.patch "$HERE/tforacle.c" "$HERE/build.sh" | sha1sum | cut -c1-40)"
if [[ -f "$stamp" && "$(cat "$stamp")" == "$want" && -f "$OUT/libtforacle.so" ]]; then
    echo "$OUT/libtforacle.so is up to date"
    exit 0
fi

rm -rf "$SRC"
mkdir -p "$SRC"
git -C "$FCEUMM_DIR" archive "$PIN" | tar -x -C "$SRC"
for p in "$HERE"/patches/*.patch; do
    patch -d "$SRC" -p1 --quiet < "$p"
done

make -C "$SRC" -f Makefile.libretro platform=unix -j"$(nproc)" \
    SHARED="-shared -Wl,-no-undefined" >"$OUT/make.log" 2>&1 || {
    tail -40 "$OUT/make.log" >&2
    exit 1
}
cp "$SRC/fceumm_libretro.so" "$OUT/fceumm_libretro.so"

cc -O2 -fPIC -shared -Wall -Wextra -Wno-unused-parameter \
    -I"$SRC/src/drivers/libretro/libretro-common/include" -I"$SRC/src" \
    -o "$OUT/libtforacle.so" "$HERE/tforacle.c" \
    -L"$OUT" -l:fceumm_libretro.so -Wl,-rpath,'$ORIGIN'

echo "$want" >"$stamp"
echo "built $OUT/libtforacle.so"
