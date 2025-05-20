#!/bin/bash

# ================= Configuration =================
GCC="riscv64-unknown-elf-gcc"
ARCH="-march=rv32gcv"
ABI="-mabi=ilp32d"
LINKER_SCRIPT="veer/link.ld"
WHISPER_CFG="veer/whisper.json"
BUILD_DIR="build"
# =================================================

# === Check Input ===
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <file.s>"
    exit 1
fi

INPUT="$1"
BASE=$(basename "$INPUT" .s)

OBJ="$BUILD_DIR/${BASE}.o"
EXE="$BUILD_DIR/${BASE}.exe"
HEX="$BUILD_DIR/${BASE}.hex"
LOG="$BUILD_DIR/log.txt"

# === Clean & Prep ===
echo "[*] Cleaning and preparing build directory..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# === Compile/Assemble & Link in one shot ===
echo "[*] Assembling & Linking $INPUT → $EXE..."
"$GCC" $ARCH $ABI \
    -c "$INPUT" -o "$OBJ"        \
    && "$GCC" $ARCH $ABI         \
       -nostdlib -T"$LINKER_SCRIPT" \
       "$OBJ" -o "$EXE"

# Bail if link failed
if [[ ! -f "$EXE" ]]; then
  echo "Error: build failed, no executable produced."
  exit 1
fi

# === Hex dump ===
echo "[*] Converting $EXE → $HEX..."
riscv64-unknown-elf-objcopy -O verilog "$EXE" "$HEX"

# === Run in Whisper ===
echo "[*] Running simulation with Whisper..."
whisper -x "$HEX" \
        -s 0x80000000 \
        --tohost 0xd0580000 \
        -f "$LOG" \
        --configfile "$WHISPER_CFG"

# === Done ===
echo "[+] Build complete. Artifacts in $BUILD_DIR:"
echo "    $OBJ"
echo "    $EXE"
echo "    $HEX"
echo "    $LOG"
