#!/usr/bin/env python3
"""Hidden-test evaluator for EPLB rebalancing policies.

Every ``outputs/eplb/<method>.py`` on disk is a submitted policy exposing the
paper's interface::

    rebalance_experts(weight[L, E], num_replicas, num_groups, num_nodes, num_gpus)
        -> (phy2log[L, R], log2phy[L, E, X], logcnt[L, E])

The hidden test holds 20 rebalance pairs of the held-out MoE layers, taken
from time blocks of the trace that no visible layer covers.  Each policy gets
a fork server (harness.py, which imports torch once and never runs policy
code).  For each pair, the evaluator has it fork a fresh sandboxed policy
process, which imports the policy, runs one warm-up rebalance on a synthetic
window, re-imports the policy, and is then sent the pair's input window only.
The placement is scored on the following window, which never enters any
policy process.  One scored rebalance per process, and every process the
policy started is killed before the next pair: nothing a policy keeps in
memory can carry over between pairs.

Rebalance work is measured here, outside the policy process: a hardware
counter of retired user-space instructions (Linux perf_event_open) is
attached to the policy process and inherited by every thread or process it
creates.  The work of a rebalance is the counter's increase from the moment
the input window is sent until the complete placement has been received.
Every operation is thus charged by the instructions it actually executes,
whatever the implementation style, and nothing the policy does can switch the
counter off or rewrite it.

From the raw returned maps and the raw counter values, this evaluator
recomputes every reported number:

* validity of each placement (exact replica budget, every expert deployed,
  consistent maps, expert groups kept whole on equally loaded nodes),
* GPU load balance on the next window (sum over layers of mean GPU load /
  sum over layers of max GPU load, the paper's imbalance factor),
* speedup: mean instructions of the frozen frontier-reference policy in this
  directory, run on the same pairs, divided by the policy's mean instructions.

A policy that fails anything (load error, crash, invalid output or placement,
instruction cap, stall, or balance more than BALANCE_TOLERANCE below the
frontier reference) is scored as failed: worst value in every scored metric.

Usage: python eval/test/evaluate.py [--outputs-dir outputs/eplb] [--scores scoring/scores_test.json]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import select
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
from array import array
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = "eplb_moe_placement"
SLICE = "test"
WINDOWS_PATH = HERE / "test_windows.json"
HARNESS_PATH = HERE / "harness.py"
ANCHOR_PATH = HERE / "anchor_frontier_reference.py"
ANCHOR_NAME = "frontier_reference_zigzag"

BALANCE_TOLERANCE = 0.005
MAX_POLICY_BYTES = 64 * 1024
# Instruction caps (deterministic; no wall-clock limit applies to a policy that computes).
STARTUP_CAP = 1.0e11     # interpreter start, torch import, sandbox
LOAD_CAP = 1.0e11        # executing the policy file (import-time work)
CALL_CAP = 1.0e11        # one rebalance (warm-up or scored), roughly 30 s of CPU
STALL_S = 600.0          # fail a process whose counter has not moved for this long
POLL_S = 0.25
MAX_REPLY_BYTES = 64 << 20

SCORED_WORST = {"speedup_vs_frontier_reference": 0.0, "balancedness": 0.0}


class HarnessError(RuntimeError):
    pass


class MeasurementUnavailable(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Hardware instruction counter (perf_event_open), read from this process only
# ---------------------------------------------------------------------------
class _PerfEventAttr(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("size", ctypes.c_uint32), ("config", ctypes.c_uint64),
                ("sample_period", ctypes.c_uint64), ("sample_type", ctypes.c_uint64),
                ("read_format", ctypes.c_uint64), ("flags", ctypes.c_uint64),
                ("wakeup_events", ctypes.c_uint32), ("bp_type", ctypes.c_uint32),
                ("config1", ctypes.c_uint64), ("config2", ctypes.c_uint64),
                ("branch_sample_type", ctypes.c_uint64), ("sample_regs_user", ctypes.c_uint64),
                ("sample_stack_user", ctypes.c_uint32), ("clockid", ctypes.c_int32),
                ("sample_regs_intr", ctypes.c_uint64), ("aux_watermark", ctypes.c_uint32),
                ("sample_max_stack", ctypes.c_uint16), ("reserved", ctypes.c_uint16)]


_SYS_PERF_EVENT_OPEN = {"x86_64": 298, "amd64": 298, "aarch64": 241, "arm64": 241}.get(platform.machine().lower())
_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.syscall.restype = ctypes.c_long
PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS = 0, 1
PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK = 1, 1
_FLAGS = (1 << 1) | (1 << 5) | (1 << 6)  # inherit to new threads/processes, user space only
_READ_FORMAT = 1 | 2  # time enabled, time running


def _perf_open(pid: int, type_: int, config: int) -> int:
    if _SYS_PERF_EVENT_OPEN is None:
        raise MeasurementUnavailable(f"no perf_event_open syscall number for {platform.machine()}")
    attr = _PerfEventAttr()
    attr.type, attr.size, attr.config = type_, ctypes.sizeof(_PerfEventAttr), config
    attr.flags, attr.read_format = _FLAGS, _READ_FORMAT
    fd = _LIBC.syscall(ctypes.c_long(_SYS_PERF_EVENT_OPEN), ctypes.byref(attr), ctypes.c_int(pid),
                       ctypes.c_int(-1), ctypes.c_int(-1), ctypes.c_ulong(8))  # FD_CLOEXEC
    if fd < 0:
        err = ctypes.get_errno()
        try:
            paranoid = Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip()
        except OSError:
            paranoid = "?"
        raise MeasurementUnavailable(
            f"perf_event_open(type={type_}, config={config}) failed: {os.strerror(err)} "
            f"(perf_event_paranoid={paranoid})")
    return int(fd)


class Counters:
    """Retired user-space instructions and CPU time of one process tree."""

    def __init__(self, pid: int):
        self.instr_fd = _perf_open(pid, PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS)
        try:
            self.clock_fd = _perf_open(pid, PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK)
        except MeasurementUnavailable:
            self.clock_fd = None
        self.multiplexed = False

    @staticmethod
    def _read(fd: int) -> tuple[float, bool]:
        value, enabled, running = struct.unpack("QQQ", os.read(fd, 24))
        if running and running < enabled:
            return value * enabled / running, True
        return float(value), False

    def instructions(self) -> float:
        value, scaled = self._read(self.instr_fd)
        self.multiplexed |= scaled
        return value

    def cpu_ns(self) -> float:
        return self._read(self.clock_fd)[0] if self.clock_fd is not None else 0.0

    def close(self) -> None:
        for fd in (self.instr_fd, self.clock_fd):
            if fd is not None:
                os.close(fd)


def _no_aslr() -> None:
    """Disable address-space randomisation in the policy process (stable counts)."""
    try:
        libc = ctypes.CDLL(None)
        persona = libc.personality(0xFFFFFFFF)
        if persona != -1:
            libc.personality(persona | 0x0040000)  # ADDR_NO_RANDOMIZE
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Process tree hygiene: the evaluator adopts orphans and kills them per policy
# ---------------------------------------------------------------------------
PR_SET_CHILD_SUBREAPER = 36


def _descendants(root: int) -> list[int]:
    parent_of = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            stat = Path(f"/proc/{entry}/stat").read_bytes()
        except OSError:
            continue
        parent_of[int(entry)] = int(stat[stat.rfind(b")") + 2:].split()[1])
    found, frontier = [], {root}
    while frontier:
        nxt = {pid for pid, ppid in parent_of.items() if ppid in frontier and pid != root}
        found.extend(nxt)
        frontier = nxt
    return found


def kill_descendants() -> None:
    for _ in range(20):
        pids = _descendants(os.getpid())
        if not pids:
            return
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        while True:
            try:
                if os.waitpid(-1, os.WNOHANG)[0] == 0:
                    break
            except ChildProcessError:
                break
        time.sleep(0.05)


def _process_gone(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_bytes()
    except OSError:
        return True
    return stat[stat.rfind(b")") + 2:stat.rfind(b")") + 3] in (b"Z", b"X")


# ---------------------------------------------------------------------------
# Policy processes: one fork server per policy, one forked child per pair
# ---------------------------------------------------------------------------
class Zygote:
    """Fork server for one policy file (imports torch once, never runs policy code)."""

    START_TIMEOUT_S = 1800.0  # torch import only; no policy code runs in the zygote
    CONTROL_TIMEOUT_S = 300.0

    def __init__(self, policy_path: Path, log, workdir: Path):
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "ATEN_", "OMP_", "MKL_"))}
        env.update({
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "ATEN_CPU_CAPABILITY": "default",  # same kernels on every x86 host
            "HOME": str(workdir),
        })
        self.proc = subprocess.Popen(
            [sys.executable, "-s", "-B", "-P", str(HARNESS_PATH), "--policy", str(policy_path),
             "--control-fd", str(child.fileno()), "--workspace", str(ROOT)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, pass_fds=(child.fileno(),), env=env,
            cwd=str(workdir), preexec_fn=_no_aslr,
        )
        child.close()
        self.sock = parent
        reply = self._recv(self.START_TIMEOUT_S, "policy server start")
        if not reply.get("ok"):
            raise HarnessError(f"policy server failed to start: {reply.get('error')}")

    def _recv(self, timeout: float, what: str) -> dict:
        ready, _, _ = select.select([self.sock], [], [], timeout)
        if not ready:
            raise HarnessError(f"policy server did not answer during {what}")
        data = self.sock.recv(1 << 16)
        if not data:
            raise HarnessError(f"policy server exited during {what}")
        try:
            reply = json.loads(data)
        except ValueError as exc:
            raise HarnessError(f"malformed policy server reply during {what}") from exc
        if not isinstance(reply, dict):
            raise HarnessError(f"malformed policy server reply during {what}")
        return reply

    def fork(self) -> "Session":
        req_r, req_w = os.pipe()
        rep_r, rep_w = os.pipe()
        try:
            socket.send_fds(self.sock, [b'{"cmd":"fork"}'], [req_r, rep_w])
        except OSError as exc:
            for fd in (req_r, req_w, rep_r, rep_w):
                os.close(fd)
            raise HarnessError(f"policy server died ({exc})") from exc
        os.close(req_r)
        os.close(rep_w)
        try:
            reply = self._recv(self.CONTROL_TIMEOUT_S, "fork")
            pid = reply.get("pid")
            if not reply.get("ok") or not isinstance(pid, int):
                raise HarnessError(f"policy server could not fork: {reply.get('error')}")
        except HarnessError:
            os.close(req_w)
            os.close(rep_r)
            raise
        return Session(pid, req_w, rep_r)

    def cleanup(self) -> None:
        """Kill everything the last child left behind."""
        try:
            self.sock.send(b'{"cmd":"cleanup"}')
            self._recv(self.CONTROL_TIMEOUT_S, "cleanup")
        except (OSError, HarnessError):
            pass

    def close(self) -> None:
        try:
            self.sock.send(b'{"cmd":"exit"}')
            self.proc.wait(timeout=60)
        except Exception:  # noqa: BLE001
            pass
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self.sock.close()


class Session:
    """One forked policy process serving one rebalance pair; its work is counted."""

    def __init__(self, pid: int, req_fd: int, rep_fd: int):
        self.pid = pid
        self.req_fd = req_fd
        self.fd = rep_fd
        self.buffer = bytearray()
        self.counters = None
        try:
            self.counters = Counters(pid)
        except MeasurementUnavailable:
            self.close()
            raise
        self._write(b"start\n")
        self.ready, _ = self.receive(STARTUP_CAP, "startup")
        if not self.ready.get("ok"):
            raise HarnessError(f"policy process failed to start: {self.ready.get('error')}")

    def _write(self, data: bytes) -> None:
        view = memoryview(data)
        try:
            while view:
                view = view[os.write(self.req_fd, view):]
        except OSError as exc:
            raise HarnessError(f"policy process died ({exc})") from exc

    def send(self, header: dict, payload: bytes = b"") -> None:
        header = dict(header, nbytes=len(payload))
        self._write(json.dumps(header, separators=(",", ":")).encode() + b"\n" + payload)

    def kill(self) -> None:
        try:
            os.kill(self.pid, signal.SIGKILL)
        except OSError:
            pass

    def receive(self, cap: float, what: str) -> tuple[dict, bytes]:
        start = self.counters.instructions()
        last_value, last_change = start, time.monotonic()
        header, need = None, 0
        while True:
            if header is None:
                newline = self.buffer.find(b"\n")
                if newline >= 0:
                    try:
                        header = json.loads(bytes(self.buffer[:newline]))
                        need = int(header.get("nbytes", 0))
                    except (ValueError, TypeError, AttributeError) as exc:
                        self.kill()
                        raise HarnessError(f"malformed reply during {what}") from exc
                    if not isinstance(header, dict) or not 0 <= need <= MAX_REPLY_BYTES:
                        self.kill()
                        raise HarnessError(f"malformed reply during {what}")
                    del self.buffer[:newline + 1]
            if header is not None and len(self.buffer) >= need:
                payload = bytes(self.buffer[:need])
                del self.buffer[:need]
                return header, payload
            ready, _, _ = select.select([self.fd], [], [], POLL_S)
            if ready:
                chunk = os.read(self.fd, 1 << 22)
                if not chunk:
                    self.kill()
                    raise HarnessError(f"policy process exited unexpectedly during {what}")
                self.buffer += chunk
            elif _process_gone(self.pid):
                raise HarnessError(f"policy process exited unexpectedly during {what}")
            value = self.counters.instructions()
            if value - start > cap:
                self.kill()
                raise HarnessError(f"{what} exceeded the instruction cap ({cap:.0e})")
            now = time.monotonic()
            if value != last_value:
                last_value, last_change = value, now
            elif now - last_change > STALL_S:
                self.kill()
                raise HarnessError(f"policy process stalled during {what} (no instructions for {STALL_S:.0f}s)")

    def load(self) -> None:
        self.send({"cmd": "load"})
        reply, _ = self.receive(LOAD_CAP, "load")
        if not reply.get("ok"):
            raise HarnessError(str(reply.get("error", "load failed")))

    def close(self) -> None:
        try:
            self._write(b'{"cmd":"exit"}\n')
        except HarnessError:
            pass
        deadline = time.monotonic() + 30.0
        while not _process_gone(self.pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.kill()
        for fd in (self.req_fd, self.fd):
            try:
                os.close(fd)
            except OSError:
                pass
        if self.counters is not None:
            self.counters.close()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def int64_bytes(rows) -> bytes:
    flat = array("q", (v for row in rows for v in row))
    if sys.byteorder != "little":
        flat.byteswap()
    return flat.tobytes()


def warmup_rows(num_layers: int, num_experts: int) -> list[list[int]]:
    """Synthetic heavy-tailed load used only to warm up torch in each process."""
    return [[100_000 // (1 + (e * 97 + layer * 31) % num_experts) + 1 for e in range(num_experts)]
            for layer in range(num_layers)]


def decode(reply: dict, payload: bytes, cfg: dict, num_layers: int, num_experts: int):
    replicas = cfg["num_replicas"]
    shapes = reply.get("shapes")
    if not isinstance(shapes, dict):
        raise HarnessError("reply without output shapes")
    width_shape = shapes.get("log2phy")
    if (shapes.get("phy2log") != [num_layers, replicas] or shapes.get("logcnt") != [num_layers, num_experts]
            or not isinstance(width_shape, list) or len(width_shape) != 3
            or width_shape[:2] != [num_layers, num_experts] or not isinstance(width_shape[2], int)
            or not 1 <= width_shape[2] <= replicas):
        raise HarnessError(f"unexpected output shapes {shapes}")
    width = width_shape[2]
    values = array("q")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    n1, n2, n3 = num_layers * replicas, num_layers * num_experts * width, num_layers * num_experts
    if len(values) != n1 + n2 + n3:
        raise HarnessError("reply payload has the wrong size")
    return values[:n1], values[n1:n1 + n2], width, values[n1 + n2:]


def check_placement(phy2log, log2phy, width, logcnt, cfg, num_layers, num_experts) -> str | None:
    """Return None if the returned maps are a valid EPLB placement, else a reason."""
    replicas = cfg["num_replicas"]
    num_groups = cfg["num_groups"]
    num_nodes = cfg["num_nodes"]
    hierarchical = num_groups % num_nodes == 0
    group_size = num_experts // num_groups
    slots_per_node = replicas // num_nodes
    groups_per_node = num_groups // num_nodes if hierarchical else 0
    for layer in range(num_layers):
        row = phy2log[layer * replicas:(layer + 1) * replicas]
        cnt_row = logcnt[layer * num_experts:(layer + 1) * num_experts]
        positions: list[list[int]] = [[] for _ in range(num_experts)]
        for physical, expert in enumerate(row):
            if not 0 <= expert < num_experts:
                return f"layer {layer}: invalid logical expert id {expert}"
            positions[expert].append(physical)
        for expert in range(num_experts):
            count = len(positions[expert])
            if count == 0:
                return f"layer {layer}: logical expert {expert} has no replica"
            if cnt_row[expert] != count:
                return f"layer {layer}: logcnt[{expert}]={cnt_row[expert]} but phy2log holds {count}"
            if count > width:
                return f"layer {layer}: log2phy is narrower than the replica count"
            base = (layer * num_experts + expert) * width
            entries = log2phy[base:base + width]
            if sorted(entries[:count]) != positions[expert]:
                return f"layer {layer}: log2phy disagrees with phy2log for expert {expert}"
            if entries[count:].count(-1) != width - count:
                return "log2phy entries beyond each expert's replica count must be -1"
        if hierarchical:
            node_of_group = [-1] * num_groups
            for physical, expert in enumerate(row):
                node = physical // slots_per_node
                group = expert // group_size
                if node_of_group[group] == -1:
                    node_of_group[group] = node
                elif node_of_group[group] != node:
                    return f"layer {layer}: expert group {group} is split across nodes"
            per_node = [0] * num_nodes
            for node in node_of_group:
                per_node[node] += 1
            if any(count != groups_per_node for count in per_node):
                return f"layer {layer}: each node must host exactly {groups_per_node} expert groups"
    return None


def balancedness(phy2log, logcnt, next_load, cfg, num_layers, num_experts) -> float:
    """Paper's imbalance factor: sum_l mean GPU load / sum_l max GPU load."""
    replicas, num_gpus = cfg["num_replicas"], cfg["num_gpus"]
    per_gpu = replicas // num_gpus
    total_mean = 0.0
    total_max = 0.0
    for layer in range(num_layers):
        gpu_load = [0.0] * num_gpus
        load_row = next_load[layer]
        cnt_row = logcnt[layer * num_experts:(layer + 1) * num_experts]
        for physical, expert in enumerate(phy2log[layer * replicas:(layer + 1) * replicas]):
            gpu_load[physical // per_gpu] += load_row[expert] / cnt_row[expert]
        total_mean += sum(gpu_load) / num_gpus
        total_max += max(gpu_load)
    return total_mean / total_max if total_max > 0 else 1.0


def run_policy(policy_path: Path, workload: dict, log_dir: Path) -> dict:
    """Run one policy on every rebalance pair; return raw per-call records."""
    cfg = workload["config"]
    num_layers = workload["num_layers"]
    num_experts = workload["num_logical_experts"]
    pairs = workload["pairs"]
    shape = [num_layers, num_experts]
    warm = int64_bytes(warmup_rows(num_layers, num_experts))
    record = {"calls": [], "failure": None, "total_calls": len(pairs), "valid": 0,
              "sandbox": None, "multiplexed": False}

    try:
        size = policy_path.stat().st_size
    except OSError as exc:
        record["failure"] = f"cannot read policy: {exc}"
        return record
    if size > MAX_POLICY_BYTES:
        record["failure"] = f"policy file is {size} bytes (limit {MAX_POLICY_BYTES})"
        return record

    policy_path = policy_path.resolve()
    workdir = Path(tempfile.mkdtemp(prefix="eplb_policy_"))
    with (log_dir.resolve() / f"{policy_path.stem}.log").open("wb") as log:
        zygote = None
        try:
            zygote = Zygote(policy_path, log, workdir)
            for index, pair in enumerate(pairs):
                session = None
                try:
                    session = zygote.fork()
                    if record["sandbox"] is None:
                        record["sandbox"] = session.ready.get("sandbox")
                    session.load()
                    session.send({"cmd": "call", "shape": shape, **cfg}, warm)
                    session.receive(CALL_CAP, "warm-up rebalance")  # result unscored
                    session.load()  # fresh module for the scored call
                    request = int64_bytes(pair["load"])
                    before_instr = session.counters.instructions()
                    before_cpu = session.counters.cpu_ns()
                    session.send({"cmd": "call", "shape": shape, **cfg}, request)
                    reply, payload = session.receive(CALL_CAP, f"rebalance of pair {index}")
                    if not reply.get("ok"):
                        raise HarnessError(
                            f"pair {index} (start {pair['start']}): {reply.get('error', 'call failed')}")
                    phy2log, log2phy, width, logcnt = decode(reply, payload, cfg, num_layers, num_experts)
                    reason = check_placement(phy2log, log2phy, width, logcnt, cfg, num_layers, num_experts)
                    if reason is not None:
                        raise HarnessError(f"pair {index} (start {pair['start']}): invalid placement: {reason}")
                    # The policy process is idle again; read the counters once it is.
                    instructions = session.counters.instructions() - before_instr
                    cpu_ns = session.counters.cpu_ns() - before_cpu
                    if instructions <= 0 or instructions > CALL_CAP:
                        raise HarnessError(f"pair {index}: impossible instruction count {instructions}")
                    record["multiplexed"] |= session.counters.multiplexed
                    record["valid"] += 1
                    record["calls"].append({
                        "pair": index, "start": pair["start"], "instructions": instructions,
                        "cpu_ns": cpu_ns,
                        "balancedness": balancedness(phy2log, logcnt, pair["next_load"], cfg,
                                                     num_layers, num_experts),
                    })
                finally:
                    if session is not None:
                        session.close()
                    zygote.cleanup()
        except HarnessError as exc:
            record["failure"] = str(exc)
        finally:
            if zygote is not None:
                zygote.close()
            kill_descendants()
    return record


def summarize(record: dict, anchor: dict | None) -> tuple[dict, str | None]:
    calls = record["calls"]
    total = record["total_calls"]
    failure = record["failure"]
    instr = [c["instructions"] for c in calls]
    bals = [c["balancedness"] for c in calls]
    cpu = [c["cpu_ns"] for c in calls]
    mean_instr = sum(instr) / len(instr) if instr else CALL_CAP
    mean_bal = sum(bals) / len(bals) if bals else 0.0
    metrics = {
        "mean_rebalance_instructions": mean_instr,
        "worst_window_balancedness": min(bals) if bals else 0.0,
        "valid_fraction": record["valid"] / total if total else 0.0,
        "diagnostic_cpu_ms_per_rebalance": (sum(cpu) / len(cpu) / 1e6) if cpu else 0.0,
        "num_rebalances": float(total),
    }
    if anchor is not None:
        metrics["balance_change_vs_frontier_reference"] = mean_bal - anchor["balancedness"]
        if failure is None and mean_bal < anchor["balancedness"] - BALANCE_TOLERANCE:
            failure = (
                f"balance constraint: balancedness {mean_bal:.4f} is more than "
                f"{BALANCE_TOLERANCE} below the frontier reference ({anchor['balancedness']:.4f})"
            )
    if failure is None and len(calls) != total:
        failure = "not every rebalance pair was scored"
    if failure is None:
        metrics["balancedness"] = mean_bal
        metrics["speedup_vs_frontier_reference"] = (
            anchor["mean_instructions"] / mean_instr if anchor is not None else 1.0
        )
        metrics["run_failed"] = 0.0
    else:
        metrics.update(SCORED_WORST)
        metrics["worst_window_balancedness"] = 0.0
        metrics["run_failed"] = 1.0
    return metrics, failure


def failed_metrics(total: int) -> dict:
    return {**SCORED_WORST, "mean_rebalance_instructions": CALL_CAP, "worst_window_balancedness": 0.0,
            "valid_fraction": 0.0, "diagnostic_cpu_ms_per_rebalance": 0.0, "num_rebalances": float(total),
            "run_failed": 1.0}


def discover_policies(outputs_dir: Path) -> dict[str, Path]:
    if not outputs_dir.is_dir():
        return {}
    found = {}
    for path in sorted(outputs_dir.glob("*.py")):
        if path.name.startswith(("_", ".")) or not path.is_file():
            continue
        found[path.stem] = path
    return found


def write_scores(out: Path, payload: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default=str(ROOT / "outputs" / "eplb"))
    parser.add_argument("--scores", default=str(ROOT / "scoring" / "scores_test.json"))
    parser.add_argument("--details", default=None, help="optional per-call record dump (JSON)")
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()
    _LIBC.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)  # adopt anything a policy leaves behind

    workload = json.loads(WINDOWS_PATH.read_text())
    total = len(workload["pairs"])
    log_dir = Path(args.log_dir) if args.log_dir else Path(tempfile.mkdtemp(prefix="eplb_logs_"))
    log_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.scores)
    policies = discover_policies(Path(args.outputs_dir))

    # Until a policy has been scored it stands as failed, so an interrupted
    # evaluation never leaves stale or missing results behind.
    results = {name: failed_metrics(total) for name in policies}
    failures = {name: "evaluation did not complete" for name in policies}
    payload = {
        "_slice": {
            "name": SLICE,
            "windows": "held-out MoE layers in held-out time blocks of the paper's trace",
            "num_rebalances_per_method": total,
        },
        "_measurement": {
            "work": "retired user-space instructions of the policy process tree (hardware counter)",
            "instruction_cap_per_rebalance": CALL_CAP,
            "balance_tolerance_vs_anchor": BALANCE_TOLERANCE,
        },
        "_failures": failures,
        "experiments": {EXPERIMENT: {"results": results}},
    }
    write_scores(out, payload)

    print(f"[{SLICE}] running frozen anchor {ANCHOR_NAME}", flush=True)
    try:
        anchor_record = run_policy(ANCHOR_PATH, workload, log_dir)
    except MeasurementUnavailable as exc:
        for name in policies:
            failures[name] = f"evaluation impossible: {exc}"
        write_scores(out, payload)
        raise SystemExit(f"[{SLICE}] cannot measure rebalance work on this host: {exc}")
    if anchor_record["failure"] is not None or not anchor_record["calls"]:
        for name in policies:
            failures[name] = f"anchor policy failed: {anchor_record['failure']}"
        write_scores(out, payload)
        raise SystemExit(f"anchor policy failed: {anchor_record['failure']} (logs in {log_dir})")
    calls = anchor_record["calls"]
    anchor = {
        "mean_instructions": sum(c["instructions"] for c in calls) / len(calls),
        "balancedness": sum(c["balancedness"] for c in calls) / len(calls),
    }
    payload["_anchor"] = {"name": ANCHOR_NAME, **anchor}
    payload["_measurement"]["sandbox"] = anchor_record["sandbox"]
    payload["_measurement"]["counter_multiplexed"] = anchor_record["multiplexed"]

    details: dict[str, dict] = {"_anchor": anchor_record}
    for name, path in policies.items():
        started = time.monotonic()
        record = run_policy(path, workload, log_dir)
        metrics, failure = summarize(record, anchor)
        results[name] = metrics
        details[name] = record
        if failure is None:
            failures.pop(name, None)
        else:
            failures[name] = failure
        payload["_measurement"]["counter_multiplexed"] |= record["multiplexed"]
        write_scores(out, payload)
        status = "FAILED: " + failure if failure else "ok"
        print(
            f"[{SLICE}] {name}: speedup={metrics['speedup_vs_frontier_reference']:.4f} "
            f"balancedness={metrics['balancedness']:.4f} "
            f"instr={metrics['mean_rebalance_instructions']:.4g} "
            f"({time.monotonic() - started:.0f}s) {status}",
            flush=True,
        )

    write_scores(out, payload)
    if args.details:
        Path(args.details).write_text(json.dumps(details, separators=(",", ":")) + "\n")
    print(f"[{SLICE}] wrote {out} ({len(results)} methods, {len(failures)} failed)", flush=True)


if __name__ == "__main__":
    main()
