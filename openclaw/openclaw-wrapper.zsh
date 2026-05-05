# Hardened OpenClaw wrapper for local model profile + gateway work.
# This is a zsh autoload function, so it intentionally has no shebang.

local real_openclaw="/opt/homebrew/bin/openclaw"
local log_dir="$HOME/.openclaw/logs"
local gateway_log="$log_dir/gateway-autostart.log"
local gateway_err="$log_dir/gateway-autostart.err.log"
local model_helper="$HOME/.openclaw/bin/openclaw-model-profile"
local prefix_warmer="$HOME/.openclaw/bin/openclaw-prefix-warmer"
local runtime_deps_guard="$HOME/.openclaw/bin/openclaw-runtime-deps-guard"
local speed_research="$HOME/.openclaw/bin/openclaw-speed-research"
local speed_research_autopilot="$HOME/.openclaw/bin/openclaw-speed-research-autopilot"
local model_env_override="$HOME/.openclaw/runtime/model-env.override.json"
local model_label="local.openclaw-model-server"
local model_plist="$HOME/Library/LaunchAgents/${model_label}.plist"
local gateway_port="${OPENCLAW_GATEWAY_PORT:-18789}"
local openclaw_cleanup_model=0
local openclaw_cleanup_gateway=0
local openclaw_mode="${1:-}"
local openclaw_custom_command=""
local -a openclaw_args
openclaw_args=("$@")

_openclaw_memory_snapshot() {
  local pressure_free_pct
  pressure_free_pct="$(/usr/bin/memory_pressure 2>/dev/null | /usr/bin/awk -F': ' '/System-wide memory free percentage:/ { gsub(/%/, "", $2); print int($2); exit }')"
  /usr/bin/vm_stat 2>/dev/null | /usr/bin/awk -v pressure_free_pct="${pressure_free_pct:-}" '
    /page size of/ { gsub(/[^0-9]/, "", $8); page=$8 }
    /^Pages free:/ { gsub(/\./, "", $3); free=$3 }
    /^Pages speculative:/ { gsub(/\./, "", $3); speculative=$3 }
    /^Pages occupied by compressor:/ { gsub(/\./, "", $5); compressor=$5 }
    END {
      if (!page) page=16384;
      printf "free_mb=%d compressor_mb=%d memory_pressure_free_pct=%s\n", ((free+speculative)*page/1048576), (compressor*page/1048576), pressure_free_pct
    }'
}

_openclaw_memory_pressure_message() {
  local snapshot free_mb compressor_mb pressure_free_pct swap_used
  snapshot="$(_openclaw_memory_snapshot)"
  free_mb="${${(s: :)snapshot}[1]#free_mb=}"
  compressor_mb="${${(s: :)snapshot}[2]#compressor_mb=}"
  pressure_free_pct="${${(s: :)snapshot}[3]#memory_pressure_free_pct=}"
  swap_used="$(/usr/sbin/sysctl vm.swapusage 2>/dev/null | /usr/bin/sed -n 's/.*used = \([0-9.]*\)M.*/\1/p' | /usr/bin/awk '{ printf "%d", $1 }')"
  if [[ -n "$compressor_mb" && "$compressor_mb" -ge 8192 ]]; then
    echo "active memory pressure: compressor=${compressor_mb}MB, free=${free_mb:-unknown}MB, swapUsed=${swap_used:-unknown}MB"
    return 0
  fi
  if [[ -n "$pressure_free_pct" && "$pressure_free_pct" != "memory_pressure_free_pct=" && "$pressure_free_pct" -ge 20 ]]; then
    return 1
  fi
  if [[ -n "$free_mb" && "$free_mb" -lt 1024 ]]; then
    echo "active memory pressure: free=${free_mb}MB, compressor=${compressor_mb:-unknown}MB, swapUsed=${swap_used:-unknown}MB"
    return 0
  fi
  if [[ -n "$swap_used" && "$swap_used" -ge 32768 && -n "$free_mb" && "$free_mb" -lt 4096 ]]; then
    echo "active memory pressure: swapUsed=${swap_used}MB with free=${free_mb}MB"
    return 0
  fi
  return 1
}

_openclaw_model_field() {
  "$model_helper" current --field "$1" 2>/dev/null
}

_openclaw_model_ready() {
  local health_url
  health_url="$(_openclaw_model_field healthUrl)" || return 1
  [[ -n "$health_url" ]] || return 1
  /usr/bin/curl -fsS --max-time 2 "$health_url" >/dev/null 2>&1
}

_openclaw_gateway_ready() {
  /usr/bin/curl -fsS --max-time 2 "http://127.0.0.1:${gateway_port}/health" >/dev/null 2>&1
}

_openclaw_export_local_runtime_env() {
  export OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN:-local-dev-token}"
  export OPENCLAW_AGENT_RUNTIME="${OPENCLAW_AGENT_RUNTIME:-pi}"
  export OPENCLAW_DISABLE_MLX_PROVIDER_PLUGIN_HOOKS="${OPENCLAW_DISABLE_MLX_PROVIDER_PLUGIN_HOOKS:-1}"
}

_openclaw_export_safe_tui_env() {
  export OPENCLAW_TUI_SAFE_TERMINAL="${OPENCLAW_TUI_SAFE_TERMINAL:-1}"
  export PI_TUI_SAFE_TERMINAL="${PI_TUI_SAFE_TERMINAL:-1}"
  export OPENCLAW_TUI_STREAMING_WATCHDOG_MS="${OPENCLAW_TUI_STREAMING_WATCHDOG_MS:-180000}"
}

_openclaw_has_arg_prefix() {
  local needle="$1"
  shift
  local arg
  for arg in "$@"; do
    [[ "$arg" == "$needle" || "$arg" == "$needle="* ]] && return 0
  done
  return 1
}

_openclaw_append_default_thinking() {
  local default_thinking="${OPENCLAW_DEFAULT_THINKING:-off}"
  [[ -n "$default_thinking" ]] || return 0
  _openclaw_has_arg_prefix "--thinking" "${openclaw_args[@]}" && return 0
  openclaw_args+=("--thinking" "$default_thinking")
}

_openclaw_guard_runtime_deps() {
  [[ -x "$runtime_deps_guard" ]] || return 0
  OPENCLAW_REAL_BIN="$real_openclaw" "$runtime_deps_guard" || {
    echo "OpenClaw plugin runtime dependency repair failed; refusing to start a broken agent loop."
    return 1
  }
}

_openclaw_clear_model_env_override() {
  /bin/rm -f "$model_env_override" 2>/dev/null || true
}

_openclaw_write_speed_research_model_env() {
  /bin/mkdir -p "${model_env_override:h}" || return $?
  /usr/bin/python3 - "$model_env_override" <<'PY'
import json
import os
import sys
from pathlib import Path

keys = (
    "OPENCLAW_RAPID_SPEED_PROFILE",
    "OPENCLAW_RAPID_CACHE_MEMORY_MB",
    "OPENCLAW_RAPID_GPU_MEMORY_UTILIZATION",
    "OPENCLAW_RAPID_DISABLE_PREFIX_CACHE",
    "OPENCLAW_RAPID_PREFILL_BATCH_SIZE",
    "OPENCLAW_RAPID_PREFILL_STEP_SIZE",
    "OPENCLAW_RAPID_COMPLETION_BATCH_SIZE",
    "OPENCLAW_RAPID_CHUNKED_PREFILL_TOKENS",
    "OPENCLAW_RAPID_PREFIX_CACHE_SIZE",
    "OPENCLAW_RAPID_NO_GC_CONTROL",
    "OPENCLAW_RAPID_MIN_FREE_MB",
    "OPENCLAW_RAPID_MAX_COMPRESSOR_MB",
    "OPENCLAW_RAPID_MAX_SWAP_MB",
    "OPENCLAW_RAPID_MEMORY_CHECK_INTERVAL_SECONDS",
    "OPENCLAW_MODEL_TOOL_MAX_TOKENS",
    "OPENCLAW_MODEL_MAX_STREAM_TOOL_CALLS",
    "OPENCLAW_MODEL_MAX_STREAM_TOOL_CHUNKS",
    "OPENCLAW_MODEL_FINISH_AFTER_FIRST_COMPLETE_TOOL_CALL",
    "OPENCLAW_MODEL_MAX_TOOL_HISTORY_MESSAGES",
    "OPENCLAW_MODEL_MAX_ASSISTANT_HISTORY_MESSAGES",
    "OPENCLAW_MODEL_MAX_STREAM_TOOL_ARG_CHARS",
    "OPENCLAW_MODEL_MAX_TOOL_RESULT_CHARS",
    "OPENCLAW_MODEL_MAX_ASSISTANT_TOOL_ARG_CHARS",
    "OPENCLAW_MODEL_MAX_SAFE_PROMPT_TOKENS",
    "OPENCLAW_MODEL_MAX_SAFE_TOOL_PROMPT_TOKENS",
    "OPENCLAW_MODEL_MIN_SAFE_FREE_MB",
    "OPENCLAW_MODEL_MAX_SAFE_COMPRESSOR_MB",
    "OPENCLAW_MODEL_MAX_SAFE_SWAP_MB",
)
path = Path(sys.argv[1])
path.write_text(json.dumps({key: os.environ[key] for key in keys if os.environ.get(key)}, indent=2) + "\n")
path.chmod(0o600)
PY
}

_openclaw_start_model_once() {
  if [[ ! -x "$model_helper" ]]; then
    echo "Missing executable $model_helper; cannot resolve OpenClaw model profile."
    return 1
  fi
  "$model_helper" sync-config >/dev/null || return $?
  if _openclaw_model_ready; then
    return 0
  fi
  local memory_class display_name blocker
  memory_class="$(_openclaw_model_field memoryClass)"
  display_name="$(_openclaw_model_field displayName)"
  if [[ "$memory_class" == "large-local" ]]; then
    blocker="$(_openclaw_memory_pressure_message)"
    if [[ -n "$blocker" ]]; then
      echo "OpenClaw memory gate: $blocker"
      echo "${display_name:-OpenClaw model} is not ready, but starting another large local model now risks a memory crash."
      echo "Stop duplicate model services or let memory settle, then retry."
      return 1
    fi
  fi
  if [[ ! -f "$model_plist" ]]; then
    "$model_helper" launchd-plist > "$model_plist" || return $?
  fi
  echo "Starting OpenClaw model server: ${display_name:-active profile}"
  if ! /bin/launchctl print "gui/$UID/$model_label" >/dev/null 2>&1; then
    /bin/launchctl bootstrap "gui/$UID" "$model_plist" >/dev/null 2>&1 || {
      echo "LaunchAgent bootstrap failed for $model_label."
      return 1
    }
  fi
  /bin/launchctl kickstart -k "gui/$UID/$model_label" >/dev/null 2>&1 || /bin/launchctl start "$model_label" >/dev/null 2>&1 || {
    echo "LaunchAgent start failed for $model_label."
    return 1
  }
  local i
  local wait_limit="${OPENCLAW_MODEL_START_WAIT_SECONDS:-180}"
  local stdout_log stderr_log upstream_port
  stdout_log="$(_openclaw_model_field server.stdout)"
  stderr_log="$(_openclaw_model_field server.stderr)"
  upstream_port="$(_openclaw_model_field server.upstreamPort 2>/dev/null || true)"
  local i=1
  while [[ "$i" -le "$wait_limit" ]]; do
    if _openclaw_model_ready; then
      echo "OpenClaw model server ready after ${i}s."
      return 0
    fi
    if [[ "$i" -gt 3 ]] && ! /bin/ps -axo command | /usr/bin/grep -E -q '[o]penclaw-model-profile exec-server|[o]penclaw-model-proxy.py'; then
      echo "OpenClaw model server exited before becoming ready."
      [[ -n "$stdout_log" ]] && echo "Logs: $stdout_log"
      [[ -n "$stderr_log" && "$stderr_log" != "$stdout_log" ]] && echo "Errors: $stderr_log"
      return 1
    fi
    if [[ "$i" -eq 5 || $((i % 10)) -eq 0 ]]; then
      local upstream_state proxy_state
      upstream_state="unknown"
      proxy_state="waiting"
      if [[ -n "$upstream_port" ]] && /usr/sbin/lsof -nP -iTCP:"$upstream_port" -sTCP:LISTEN >/dev/null 2>&1; then
        upstream_state="listening:${upstream_port}"
      elif [[ -n "$upstream_port" ]]; then
        upstream_state="starting:${upstream_port}"
      fi
      if /usr/sbin/lsof -nP -iTCP:"$(_openclaw_model_field server.port)" -sTCP:LISTEN >/dev/null 2>&1; then
        proxy_state="listening"
      fi
      echo "OpenClaw model startup still in progress (${i}s/${wait_limit}s): upstream=${upstream_state}, proxy=${proxy_state}"
      [[ -n "$stdout_log" ]] && echo "Logs: $stdout_log"
    fi
    sleep 1
    i=$((i + 1))
  done
  echo "OpenClaw model server did not become ready within ${wait_limit}s."
  [[ -n "$stdout_log" ]] && echo "Logs: $stdout_log"
  [[ -n "$stderr_log" && "$stderr_log" != "$stdout_log" ]] && echo "Errors: $stderr_log"
  return 1
}

_openclaw_warm_model_prefix_once() {
  [[ "${OPENCLAW_MODEL_PREFIX_WARM_ENABLED:-1}" == "1" ]] || return 0
  [[ -x "$prefix_warmer" ]] || return 0
  local warm_log="$log_dir/openclaw-prefix-warmer.log"
  echo "Preparing OpenClaw model prefix cache..."
  "$prefix_warmer" >>"$warm_log" 2>&1 || {
    echo "OpenClaw prefix warmup did not complete; first prompt may be slower. Logs: $warm_log"
    return 0
  }
}

_openclaw_stop_model_owned() {
  local port upstream_port pid candidate_port
  port="$(_openclaw_model_field server.port)"
  upstream_port="$(_openclaw_model_field server.upstreamPort 2>/dev/null || true)"
  /bin/launchctl bootout "gui/$UID" "$model_plist" >/dev/null 2>&1 || true
  for candidate_port in "$port" "$upstream_port"; do
    [[ -n "$candidate_port" ]] || continue
    for pid in "${(@f)$('/usr/sbin/lsof' -nP -tiTCP:"$candidate_port" -sTCP:LISTEN 2>/dev/null)}"; do
      [[ -n "$pid" ]] && /bin/kill -TERM "$pid" 2>/dev/null || true
    done
  done
  sleep 1
  for candidate_port in "$port" "$upstream_port"; do
    [[ -n "$candidate_port" ]] || continue
    for pid in "${(@f)$('/usr/sbin/lsof' -nP -tiTCP:"$candidate_port" -sTCP:LISTEN 2>/dev/null)}"; do
      [[ -n "$pid" ]] && /bin/kill -KILL "$pid" 2>/dev/null || true
    done
  done
  for pid in "${(@f)$('/usr/bin/pgrep' -f "$HOME/.openclaw/bin/openclaw-rapid-launcher" 2>/dev/null)}"; do
    [[ -n "$pid" ]] && /bin/kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${(@f)$('/usr/bin/pgrep' -f "$HOME/.openclaw/bin/openclaw-jang-vlm-launcher" 2>/dev/null)}"; do
    [[ -n "$pid" ]] && /bin/kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${(@f)$('/usr/bin/pgrep' -f "$HOME/.openclaw/bin/openclaw-vmlx-launcher" 2>/dev/null)}"; do
    [[ -n "$pid" ]] && /bin/kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 1
  for pid in "${(@f)$('/usr/bin/pgrep' -f "$HOME/.openclaw/bin/openclaw-rapid-launcher" 2>/dev/null)}"; do
    [[ -n "$pid" ]] && /bin/kill -KILL "$pid" 2>/dev/null || true
  done
  for pid in "${(@f)$('/usr/bin/pgrep' -f "$HOME/.openclaw/bin/openclaw-jang-vlm-launcher" 2>/dev/null)}"; do
    [[ -n "$pid" ]] && /bin/kill -KILL "$pid" 2>/dev/null || true
  done
  for pid in "${(@f)$('/usr/bin/pgrep' -f "$HOME/.openclaw/bin/openclaw-vmlx-launcher" 2>/dev/null)}"; do
    [[ -n "$pid" ]] && /bin/kill -KILL "$pid" 2>/dev/null || true
  done
}

_openclaw_stop_gateway_owned() {
  local pid
  for pid in "${(@f)$('/usr/sbin/lsof' -nP -tiTCP:"$gateway_port" -sTCP:LISTEN 2>/dev/null)}"; do
    [[ -n "$pid" ]] && /bin/kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 1
  for pid in "${(@f)$('/usr/sbin/lsof' -nP -tiTCP:"$gateway_port" -sTCP:LISTEN 2>/dev/null)}"; do
    [[ -n "$pid" ]] && /bin/kill -KILL "$pid" 2>/dev/null || true
  done
}

_openclaw_start_gateway_once() {
  if _openclaw_gateway_ready || /usr/sbin/lsof -nP -iTCP:"$gateway_port" -sTCP:LISTEN >/dev/null 2>&1; then
    return 0
  fi
  _openclaw_guard_runtime_deps || return $?
  echo "Starting OpenClaw gateway on ${gateway_port}..."
  OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN:-local-dev-token}" \
    nohup "$real_openclaw" gateway run --port "$gateway_port" --token "${OPENCLAW_GATEWAY_TOKEN:-local-dev-token}" >"$gateway_log" 2>"$gateway_err" &
  local j
  for j in {1..30}; do
    if _openclaw_gateway_ready || /usr/sbin/lsof -nP -iTCP:"$gateway_port" -sTCP:LISTEN >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "OpenClaw gateway did not become ready within 30s. See $gateway_log and $gateway_err"
  return 1
}

case "${1:-}" in
  model-profile)
    shift
    "$model_helper" "$@"
    return $?
    ;;
  model-start|gemma-start)
    export OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN:-local-dev-token}"
    _openclaw_clear_model_env_override
    mkdir -p "$log_dir"
    _openclaw_start_model_once || return $?
    _openclaw_warm_model_prefix_once
    return 0
    ;;
  model-stop|gemma-stop)
    _openclaw_clear_model_env_override
    _openclaw_stop_model_owned
    return 0
    ;;
  model-status|gemma-status)
    if _openclaw_model_ready; then
      echo "$(_openclaw_model_field displayName) ready at $(_openclaw_model_field healthUrl)"
      return 0
    fi
    local launch_state
    launch_state="$(/bin/launchctl print "gui/$UID/$model_label" 2>/dev/null || true)"
    if [[ "$launch_state" == *"state = running"* ]]; then
      echo "OpenClaw model LaunchAgent running but API not ready"
      return 1
    fi
    echo "OpenClaw model server stopped: $(_openclaw_model_field displayName)"
    return 1
    ;;
  local-health|health-local)
    "$HOME/.openclaw/bin/openclaw-local-health"
    return $?
    ;;
  speed-research-setup|research-speed-setup)
    "$speed_research" setup
    return $?
    ;;
  speed-research-prompt|research-speed-prompt)
    "$speed_research" prompt
    return $?
    ;;
  speed-research-benchmark|research-speed-benchmark)
    shift
    "$speed_research" benchmark "$@"
    return $?
    ;;
  speed-research-add-source|research-speed-add-source)
    shift
    "$speed_research" add-source "$@"
    return $?
    ;;
  speed-research|research-speed|speed-research-auto|research-speed-auto|speed-research-overnight|research-speed-overnight|speed-research-tui|research-speed-tui)
    _openclaw_export_local_runtime_env
    _openclaw_export_safe_tui_env
    export OPENCLAW_RAPID_SPEED_PROFILE="${OPENCLAW_RAPID_SPEED_PROFILE:-balanced}"
    export OPENCLAW_RAPID_CACHE_MEMORY_MB="${OPENCLAW_RAPID_CACHE_MEMORY_MB:-1536}"
    export OPENCLAW_RAPID_GPU_MEMORY_UTILIZATION="${OPENCLAW_RAPID_GPU_MEMORY_UTILIZATION:-0.82}"
    export OPENCLAW_RAPID_DISABLE_PREFIX_CACHE="${OPENCLAW_RAPID_DISABLE_PREFIX_CACHE:-0}"
    export OPENCLAW_RAPID_PREFILL_BATCH_SIZE="${OPENCLAW_RAPID_PREFILL_BATCH_SIZE:-8}"
    export OPENCLAW_RAPID_PREFILL_STEP_SIZE="${OPENCLAW_RAPID_PREFILL_STEP_SIZE:-2048}"
    export OPENCLAW_RAPID_COMPLETION_BATCH_SIZE="${OPENCLAW_RAPID_COMPLETION_BATCH_SIZE:-16}"
    export OPENCLAW_RAPID_CHUNKED_PREFILL_TOKENS="${OPENCLAW_RAPID_CHUNKED_PREFILL_TOKENS:-2048}"
    export OPENCLAW_RAPID_PREFIX_CACHE_SIZE="${OPENCLAW_RAPID_PREFIX_CACHE_SIZE:-64}"
    export OPENCLAW_RAPID_NO_GC_CONTROL="${OPENCLAW_RAPID_NO_GC_CONTROL:-0}"
    export OPENCLAW_MODEL_PREFIX_WARM_ENABLED="${OPENCLAW_MODEL_PREFIX_WARM_ENABLED:-1}"
    export OPENCLAW_MODEL_TOOL_MAX_TOKENS="${OPENCLAW_MODEL_TOOL_MAX_TOKENS:-128}"
    export OPENCLAW_MODEL_MAX_STREAM_TOOL_CALLS="${OPENCLAW_MODEL_MAX_STREAM_TOOL_CALLS:-1}"
    export OPENCLAW_MODEL_MAX_STREAM_TOOL_CHUNKS="${OPENCLAW_MODEL_MAX_STREAM_TOOL_CHUNKS:-3}"
    export OPENCLAW_MODEL_FINISH_AFTER_FIRST_COMPLETE_TOOL_CALL="${OPENCLAW_MODEL_FINISH_AFTER_FIRST_COMPLETE_TOOL_CALL:-1}"
    export OPENCLAW_MODEL_MAX_TOOL_HISTORY_MESSAGES="${OPENCLAW_MODEL_MAX_TOOL_HISTORY_MESSAGES:-2}"
    export OPENCLAW_MODEL_MAX_ASSISTANT_HISTORY_MESSAGES="${OPENCLAW_MODEL_MAX_ASSISTANT_HISTORY_MESSAGES:-4}"
    export OPENCLAW_MODEL_MAX_STREAM_TOOL_ARG_CHARS="${OPENCLAW_MODEL_MAX_STREAM_TOOL_ARG_CHARS:-1200}"
    export OPENCLAW_MODEL_MAX_TOOL_RESULT_CHARS="${OPENCLAW_MODEL_MAX_TOOL_RESULT_CHARS:-700}"
    export OPENCLAW_MODEL_MAX_ASSISTANT_TOOL_ARG_CHARS="${OPENCLAW_MODEL_MAX_ASSISTANT_TOOL_ARG_CHARS:-350}"
    export OPENCLAW_MODEL_MAX_SAFE_PROMPT_TOKENS="${OPENCLAW_MODEL_MAX_SAFE_PROMPT_TOKENS:-12000}"
    export OPENCLAW_MODEL_MAX_SAFE_TOOL_PROMPT_TOKENS="${OPENCLAW_MODEL_MAX_SAFE_TOOL_PROMPT_TOKENS:-12000}"
    export OPENCLAW_RAPID_MIN_FREE_MB="${OPENCLAW_RAPID_MIN_FREE_MB:-4096}"
    export OPENCLAW_RAPID_MAX_COMPRESSOR_MB="${OPENCLAW_RAPID_MAX_COMPRESSOR_MB:-4096}"
    export OPENCLAW_RAPID_MAX_SWAP_MB="${OPENCLAW_RAPID_MAX_SWAP_MB:-2048}"
    export OPENCLAW_RAPID_MEMORY_CHECK_INTERVAL_SECONDS="${OPENCLAW_RAPID_MEMORY_CHECK_INTERVAL_SECONDS:-2}"
    export OPENCLAW_MODEL_MIN_SAFE_FREE_MB="${OPENCLAW_MODEL_MIN_SAFE_FREE_MB:-512}"
    export OPENCLAW_MODEL_MAX_SAFE_COMPRESSOR_MB="${OPENCLAW_MODEL_MAX_SAFE_COMPRESSOR_MB:-4096}"
    export OPENCLAW_MODEL_MAX_SAFE_SWAP_MB="${OPENCLAW_MODEL_MAX_SAFE_SWAP_MB:-2048}"
    export OPENCLAW_MODEL_START_WAIT_SECONDS="${OPENCLAW_MODEL_START_WAIT_SECONDS:-420}"
    _openclaw_append_default_thinking
    mkdir -p "$log_dir"
    "$speed_research" setup >/dev/null || return $?
    local research_prompt
    research_prompt="$("$speed_research" prompt)" || return $?
    local speed_research_session
    speed_research_session="${OPENCLAW_SPEED_RESEARCH_SESSION:-speed-research-$(/bin/date +%Y%m%d-%H%M%S)}"
    _openclaw_stop_model_owned
    _openclaw_write_speed_research_model_env || return $?
    openclaw_cleanup_model=1
    _openclaw_start_model_once || return $?
    _openclaw_warm_model_prefix_once
    _openclaw_gateway_ready || /usr/sbin/lsof -nP -iTCP:"$gateway_port" -sTCP:LISTEN >/dev/null 2>&1 || openclaw_cleanup_gateway=1
    _openclaw_start_gateway_once || return $?
    if [[ "$openclaw_mode" == "speed-research-tui" || "$openclaw_mode" == "research-speed-tui" ]]; then
      openclaw_args=("tui" "--session" "$speed_research_session" "--history-limit" "${OPENCLAW_SPEED_RESEARCH_HISTORY_LIMIT:-1}" "--timeout-ms" "${OPENCLAW_SPEED_RESEARCH_TIMEOUT_MS:-900000}" "--message" "$research_prompt")
    else
      shift
      openclaw_custom_command="speed-research-autopilot"
      openclaw_args=("$@")
    fi
    ;;
  tui|terminal)
    _openclaw_export_local_runtime_env
    _openclaw_export_safe_tui_env
    _openclaw_append_default_thinking
    _openclaw_clear_model_env_override
    mkdir -p "$log_dir"
    if [[ "${OPENCLAW_START_MODEL_ON_TUI:-${OPENCLAW_START_GEMMA_ON_TUI:-1}}" == "1" ]]; then
      _openclaw_model_ready || openclaw_cleanup_model=1
      _openclaw_start_model_once || return $?
      _openclaw_warm_model_prefix_once
    fi
    _openclaw_gateway_ready || /usr/sbin/lsof -nP -iTCP:"$gateway_port" -sTCP:LISTEN >/dev/null 2>&1 || openclaw_cleanup_gateway=1
    _openclaw_start_gateway_once || return $?
    ;;
  chat|agent)
    _openclaw_export_local_runtime_env
    _openclaw_append_default_thinking
    _openclaw_clear_model_env_override
    mkdir -p "$log_dir"
    _openclaw_model_ready || openclaw_cleanup_model=1
    _openclaw_start_model_once || return $?
    _openclaw_warm_model_prefix_once
    _openclaw_gateway_ready || /usr/sbin/lsof -nP -iTCP:"$gateway_port" -sTCP:LISTEN >/dev/null 2>&1 || openclaw_cleanup_gateway=1
    _openclaw_start_gateway_once || return $?
    ;;
  gateway)
    _openclaw_export_local_runtime_env
    mkdir -p "$log_dir"
    ;;
esac

local openclaw_status=0
{
  if [[ "$openclaw_custom_command" == "speed-research-autopilot" ]]; then
    OPENCLAW_REAL_BIN="$real_openclaw" OPENCLAW_SPEED_RESEARCH_SESSION="${speed_research_session:-${OPENCLAW_SPEED_RESEARCH_SESSION:-speed-research-auto}}" "$speed_research_autopilot" "${openclaw_args[@]}" || openclaw_status=$?
  else
    command "$real_openclaw" "${openclaw_args[@]}" || openclaw_status=$?
  fi
} always {
  if [[ "$openclaw_cleanup_model" == "1" ]]; then
    _openclaw_stop_model_owned
  fi
  if [[ "$openclaw_cleanup_gateway" == "1" ]]; then
    if [[ "$openclaw_mode" == "agent" || "$openclaw_mode" == "chat" ]] && [[ "${OPENCLAW_KEEP_GATEWAY_WARM_AFTER_AGENT:-0}" == "1" ]]; then
      :
    else
      _openclaw_stop_gateway_owned
    fi
  fi
}
return "$openclaw_status"
