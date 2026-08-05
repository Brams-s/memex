#!/usr/bin/env bash
# Shared helpers for the memex herdr plugin. Sourced by plugin.sh and memex-pane.sh; not
# executable on its own. Callers set `mode` before sourcing so refuse() knows whether it has a
# caller to inform.
#
# shellcheck shell=bash

# herdr runs plugin commands server-side with a minimal PATH: jq, curl, cargo, and a Homebrew or
# ~/.local/bin memex are otherwise invisible.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

# The plugin root. Runtime env is authoritative; the fallback keeps this file usable from a plain
# shell (and from any context where herdr did not inject its env).
MEMEX_PLUGIN_ROOT="${HERDR_PLUGIN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# The herdr CLI to drive. herdr passes its own absolute path so the plugin talks to the running
# binary rather than whatever is first on PATH.
H="${HERDR_BIN_PATH:-herdr}"

PLUGIN_ID="${HERDR_PLUGIN_ID:-nicosuave.memex}"

# Actions refuse loudly: exit 1 plus exactly one stderr line, both surfaced by
# `herdr plugin log list`. Startup and event modes have no caller to inform and must never fail a
# session start, so they refuse silently.
refuse() {
  case "${mode:-}" in
  startup | event | event-* | auto-*) exit 0 ;;
  esac
  printf 'memex: %s\n' "$1" >&2
  exit 1
}

# The memex binary, in the order that keeps every install shape working:
#   bin/memex             installed by herdr/install.sh on `herdr plugin install`
#   target/release/memex  a dev checkout linked with `herdr plugin link .`
#   $(command -v memex)   a brew / setup.sh install already on PATH
resolve_memex() {
  local candidate
  for candidate in "$MEMEX_PLUGIN_ROOT/bin/memex" "$MEMEX_PLUGIN_ROOT/target/release/memex"; do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  candidate=$(command -v memex 2>/dev/null) || return 1
  [ -n "$candidate" ] || return 1
  printf '%s\n' "$candidate"
}

# Plugin config lives at $HERDR_PLUGIN_CONFIG_DIR/config.toml and is re-read on every invocation,
# so an edit takes effect without restarting herdr. Parsed with sed (no TOML dependency at
# runtime); an unparsable or unknown value simply leaves the caller's default in place.
MEMEX_CONFIG_FILE="${HERDR_PLUGIN_CONFIG_DIR:-}/config.toml"

# A quoted string value: key = "value"
config_str() {
  [ -n "${HERDR_PLUGIN_CONFIG_DIR:-}" ] && [ -f "$MEMEX_CONFIG_FILE" ] || return 0
  sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*[\"']\([^\"']*\)[\"'].*/\1/p" \
    "$MEMEX_CONFIG_FILE" 2>/dev/null | tail -n1
}

# A bare boolean value: key = true (quotes tolerated, trailing comments ignored)
config_bool() {
  [ -n "${HERDR_PLUGIN_CONFIG_DIR:-}" ] && [ -f "$MEMEX_CONFIG_FILE" ] || return 0
  sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*[\"']*\([a-z]*\)[\"']*.*/\1/p" \
    "$MEMEX_CONFIG_FILE" 2>/dev/null | tail -n1
}

# Where background output goes. herdr supplies the state dir; the fallback keeps logging working
# if it is ever absent instead of writing into the user's repo.
state_dir() {
  local dir="${HERDR_PLUGIN_STATE_DIR:-${TMPDIR:-/tmp}/memex-herdr}"
  mkdir -p "$dir" 2>/dev/null || return 1
  printf '%s\n' "$dir"
}
