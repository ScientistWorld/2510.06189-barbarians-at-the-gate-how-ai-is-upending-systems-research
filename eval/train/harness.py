#!/usr/bin/env python3
"""Sandboxed runner for one EPLB policy file (started by evaluate.py).

This file runs as a fork server ("zygote") for one policy file.  The zygote
imports torch once and reads the policy file's bytes, but never executes
policy code.  For every rebalance pair the evaluator asks it to fork a fresh
child; the child serves exactly that pair and exits.

Nothing in these processes measures anything.  The evaluator measures the
work of every rebalance from outside, with the CPU's hardware counter of
retired user-space instructions attached to the child (and inherited by every
thread or process the child creates).  No state in the child takes part in
the measurement, so nothing the policy does can switch it off or forge it.

A child:
1. waits until the evaluator has attached its counters (``start`` line);
2. seals itself into an OS-level sandbox (Linux Landlock): read-only access
   to the Python and torch installation and system libraries, no access at
   all to the workspace (and so to any data or evaluator file), no file
   writes, no exec, no TCP, and no signals or abstract-socket connections to
   processes outside its own sandbox.  If the kernel has no Landlock, an
   in-process audit hook blocks the same file and process operations;
3. serves requests: ``load`` (execute the policy file in a fresh module),
   ``call`` (one rebalance on the window it is sent), ``exit``.

The zygote is a child subreaper: after each child has finished, it kills
every process that child left behind, so no helper process outlives the pair
it was started for.

Protocol (child): requests on the request pipe, replies on the reply pipe.
Every message is one JSON header line, followed by a raw little-endian int64
payload of ``nbytes`` bytes.  Control (zygote): JSON messages on a
SOCK_SEQPACKET socket; ``fork`` carries the child's two pipe ends.  Anything
the policy prints goes to stderr.
"""

import argparse
import ctypes
import gc
import json
import os
import platform
import random
import signal
import socket
import struct
import sys
import types

MAX_POLICY_BYTES = 64 * 1024
PR_SET_CHILD_SUBREAPER = 36


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--control-fd", type=int, required=True)
    parser.add_argument("--workspace", required=True)
    return parser.parse_args()


ARGS = parse_args()
os.dup2(2, 1)  # policy prints never reach a protocol channel
sys.stdout = sys.stderr
POLICY_PATH = os.path.abspath(ARGS.policy)
with open(POLICY_PATH, "rb") as _f:
    POLICY_BYTES = _f.read(MAX_POLICY_BYTES + 1)
WORKSPACE = os.path.realpath(ARGS.workspace)
LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.syscall.restype = ctypes.c_long

import torch  # noqa: E402

try:  # loaded before forking so tensor -> bytes conversion is available
    import numpy  # noqa: E402,F401
    HAVE_NUMPY = True
except Exception:  # noqa: BLE001
    HAVE_NUMPY = False

torch.set_num_threads(1)
INT_DTYPES = (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8)


# ---------------------------------------------------------------------------
# OS-level sandbox (applied in each child)
# ---------------------------------------------------------------------------
SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446
LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1
PR_SET_NO_NEW_PRIVS = 38
ACCESS_FS_READ_FILE = 1 << 2
ACCESS_FS_READ_DIR = 1 << 3
FS_RIGHTS_BY_ABI = {1: 13, 2: 14, 3: 15, 4: 15}  # number of handled fs right bits


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64),
                ("handled_access_net", ctypes.c_uint64),
                ("scoped", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def _readable_paths() -> list[str]:
    candidates = {sys.prefix, sys.base_prefix, sys.exec_prefix,
                  os.path.dirname(os.path.realpath(torch.__file__))}
    if HAVE_NUMPY:
        candidates.add(os.path.dirname(os.path.realpath(numpy.__file__)))
    candidates.update(p for p in sys.path if p)
    candidates.update(("/usr/lib", "/usr/lib64", "/usr/local/lib", "/lib", "/lib64", "/etc",
                       "/usr/share/zoneinfo", "/sys/devices/system/cpu", "/proc/cpuinfo",
                       "/dev/null", "/dev/urandom", "/dev/zero"))
    keep = []
    for path in sorted(candidates):
        real = os.path.realpath(path)
        if not os.path.exists(real):
            continue
        # never grant the workspace or anything that contains it
        if real == WORKSPACE or WORKSPACE.startswith(real.rstrip("/") + "/") or real.startswith(WORKSPACE + "/"):
            continue
        keep.append(real)
    return sorted(set(keep))


READABLE = _readable_paths()


def _landlock(paths: list[str]) -> dict:
    syscall = LIBC.syscall
    abi = syscall(ctypes.c_long(SYS_LANDLOCK_CREATE_RULESET), None, ctypes.c_size_t(0),
                  ctypes.c_uint32(LANDLOCK_CREATE_RULESET_VERSION))
    if abi < 1:
        return {"mode": "unavailable", "errno": ctypes.get_errno()}
    fs_bits = FS_RIGHTS_BY_ABI.get(abi, 16)
    attr = _RulesetAttr((1 << fs_bits) - 1, 0, 0)
    size = 8
    if abi >= 4:
        attr.handled_access_net = 0b11  # bind and connect TCP
        size = 16
    if abi >= 6:
        attr.scoped = 0b11  # abstract unix sockets and signals stay inside the sandbox
        size = 24
    ruleset = syscall(ctypes.c_long(SYS_LANDLOCK_CREATE_RULESET), ctypes.byref(attr),
                      ctypes.c_size_t(size), ctypes.c_uint32(0))
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")
    for path in paths:
        try:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        except OSError:
            continue
        rights = ACCESS_FS_READ_FILE | (ACCESS_FS_READ_DIR if os.path.isdir(path) else 0)
        rule = _PathBeneathAttr(rights, fd)
        syscall(ctypes.c_long(SYS_LANDLOCK_ADD_RULE), ctypes.c_int(ruleset),
                ctypes.c_int(LANDLOCK_RULE_PATH_BENEATH), ctypes.byref(rule), ctypes.c_uint32(0))
        os.close(fd)
    if LIBC.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")
    if syscall(ctypes.c_long(SYS_LANDLOCK_RESTRICT_SELF), ctypes.c_int(ruleset), ctypes.c_uint32(0)) != 0:
        raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    os.close(ruleset)
    return {"mode": "landlock", "abi": abi}


_FALLBACK_BLOCKED = frozenset({
    "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork",
    "os.forkpty", "pty.spawn", "os.kill", "os.killpg", "ctypes.dlopen", "ctypes.dlsym",
    "ctypes.dlsym/handle", "socket.connect", "socket.bind", "os.remove", "os.rename",
    "os.truncate", "os.chmod", "shutil.copyfile", "shutil.rmtree", "mmap.__new__",
})


def _fallback_hook(paths: list[str]) -> dict:
    allowed = tuple(p.rstrip("/") + "/" for p in paths)

    def hook(event, args):
        if event == "open":
            path = args[0]
            if not isinstance(path, (str, bytes, os.PathLike)):
                raise PermissionError("sandbox: raw file descriptors are not allowed")
            real = os.path.realpath(os.fsdecode(path))
            mode = args[1] if len(args) > 1 else "r"
            if isinstance(mode, str) and any(c in mode for c in "wax+") and real != "/dev/null":
                raise PermissionError(f"sandbox: writing {real} is not allowed")
            if not (real in paths or real.startswith(allowed)):
                raise PermissionError(f"sandbox: reading {real} is not allowed")
        elif event in _FALLBACK_BLOCKED:
            raise PermissionError(f"sandbox: {event} is not allowed")

    sys.addaudithook(hook)
    return {"mode": "audit-hook (no Landlock in this kernel)"}


def apply_sandbox() -> dict:
    info = None
    if platform.system() == "Linux":
        try:
            info = _landlock(READABLE)
        except OSError as exc:
            info = {"mode": "unavailable", "error": str(exc)}
    if info is None or info.get("mode") != "landlock":
        info = _fallback_hook(READABLE)
    try:
        os.listdir(os.path.join(WORKSPACE, "eval"))
        info["workspace_readable"] = True
    except OSError:
        info["workspace_readable"] = False
    return info


# ---------------------------------------------------------------------------
# Child: one rebalance pair
# ---------------------------------------------------------------------------
class Channel:
    def __init__(self, req_fd: int, rep_fd: int):
        self.reader = os.fdopen(req_fd, "rb")
        self.rep_fd = rep_fd

    def send(self, header: dict, payload: bytes = b"") -> None:
        header["nbytes"] = len(payload)
        view = memoryview(json.dumps(header, separators=(",", ":")).encode() + b"\n" + payload)
        while view:
            view = view[os.write(self.rep_fd, view):]

    def receive(self):
        line = self.reader.readline()
        if not line:
            return None, b""
        header = json.loads(line)
        n = int(header.get("nbytes", 0))
        payload = self.reader.read(n) if n else b""
        if len(payload) != n:
            return None, b""
        return header, payload


def load_policy():
    if len(POLICY_BYTES) > MAX_POLICY_BYTES:
        raise ValueError(f"policy file exceeds {MAX_POLICY_BYTES} bytes")
    source = POLICY_BYTES.decode("utf-8")
    random.seed(0)
    torch.manual_seed(0)
    module = types.ModuleType("eplb_policy")
    module.__dict__["__file__"] = POLICY_PATH
    sys.modules["eplb_policy"] = module
    exec(compile(source, POLICY_PATH, "exec"), module.__dict__)
    func = module.__dict__.get("rebalance_experts")
    if not callable(func):
        raise ValueError("policy does not define a callable rebalance_experts")
    return func


def tensor_bytes(value, name: str, shape_ok) -> tuple[list[int], bytes]:
    if type(value) is not torch.Tensor:
        raise TypeError(f"{name} must be a plain torch.Tensor, got {type(value).__name__}")
    if value.dtype not in INT_DTYPES:
        raise TypeError(f"{name} must have an integer dtype, got {value.dtype}")
    shape = list(value.shape)
    reason = shape_ok(shape)
    if reason:
        raise ValueError(f"{name} has shape {tuple(shape)}: {reason}")
    flat = value.detach().to(device="cpu", dtype=torch.int64).contiguous().reshape(-1)
    if HAVE_NUMPY:
        return shape, flat.numpy().astype("<i8", copy=False).tobytes()
    values = flat.tolist()
    return shape, struct.pack(f"<{len(values)}q", *values)


def run_call(func, header: dict, payload: bytes) -> tuple[dict, bytes]:
    layers, experts = int(header["shape"][0]), int(header["shape"][1])
    replicas = int(header["num_replicas"])
    weight = torch.frombuffer(bytearray(payload), dtype=torch.int64).view(layers, experts)
    try:
        result = func(weight, replicas, int(header["num_groups"]), int(header["num_nodes"]),
                      int(header["num_gpus"]))
    except BaseException as exc:  # noqa: BLE001 - every policy failure is reported
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, b""
    try:
        if not isinstance(result, tuple) or len(result) != 3:
            raise TypeError("rebalance_experts must return (phy2log, log2phy, logcnt)")
        checks = (
            ("phy2log", lambda s: None if s == [layers, replicas] else f"expected {(layers, replicas)}"),
            ("log2phy", lambda s: None if len(s) == 3 and s[:2] == [layers, experts] and 1 <= s[2] <= replicas
             else f"expected ({layers}, {experts}, X) with 1 <= X <= {replicas}"),
            ("logcnt", lambda s: None if s == [layers, experts] else f"expected {(layers, experts)}"),
        )
        shapes, chunks = {}, []
        for (name, ok), value in zip(checks, result):
            shape, data = tensor_bytes(value, name, ok)
            shapes[name] = shape
            chunks.append(data)
    except BaseException as exc:  # noqa: BLE001
        return {"ok": False, "error": f"invalid output: {exc}"}, b""
    return {"ok": True, "shapes": shapes}, b"".join(chunks)


def serve(channel: Channel) -> None:
    line = channel.reader.readline()  # the evaluator has attached its counters
    if line != b"start\n":
        return
    sandbox = apply_sandbox()
    channel.send({"ok": True, "ready": True, "sandbox": sandbox, "numpy": HAVE_NUMPY,
                  "hash_randomization": sys.flags.hash_randomization})
    func = None
    while True:
        header, payload = channel.receive()
        if header is None or header.get("cmd") == "exit":
            return
        cmd = header.get("cmd")
        if cmd == "load":
            try:
                func = load_policy()
                reply = {"ok": True}
            except BaseException as exc:  # noqa: BLE001
                func = None
                reply = {"ok": False, "error": f"load failed: {type(exc).__name__}: {exc}"}
            gc.collect()  # start every measured call from a collected heap
            channel.send(reply)
        elif cmd == "call":
            if func is None:
                channel.send({"ok": False, "error": "policy not loaded"})
            else:
                reply, data = run_call(func, header, payload)
                channel.send(reply, data)
        else:
            channel.send({"ok": False, "error": f"unknown command {cmd!r}"})


# ---------------------------------------------------------------------------
# Zygote
# ---------------------------------------------------------------------------
def _descendants() -> list[int]:
    me = os.getpid()
    parent_of = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as f:
                stat = f.read()
        except OSError:
            continue
        fields = stat[stat.rfind(b")") + 2:].split()
        parent_of[int(entry)] = int(fields[1])
    found, frontier = [], [me]
    while frontier:
        nxt = [pid for pid, ppid in parent_of.items() if ppid in frontier and pid != me]
        found.extend(nxt)
        frontier = nxt
    return found


def kill_leftovers() -> int:
    """Kill every process below the zygote (it is a subreaper, so orphans land here)."""
    killed = 0
    for _ in range(20):
        pids = _descendants()
        if not pids:
            break
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
                killed += 1
            except OSError:
                pass
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if pid == 0:
                break
    while True:  # reap anything already dead
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                break
        except ChildProcessError:
            break
    return killed


def zygote() -> None:
    LIBC.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    control = socket.socket(fileno=ARGS.control_fd)
    control.send(json.dumps({"ok": True, "ready": True}).encode())
    while True:
        try:
            msg, fds, _, _ = socket.recv_fds(control, 1 << 16, 4)
        except OSError:
            break
        if not msg:
            break
        try:
            cmd = json.loads(msg).get("cmd")
        except ValueError:
            cmd = None
        if cmd == "fork" and len(fds) == 2:
            pid = os.fork()
            if pid == 0:
                code = 0
                try:
                    control.close()
                    serve(Channel(fds[0], fds[1]))
                except BaseException:  # noqa: BLE001
                    code = 1
                finally:
                    os._exit(code)
            for fd in fds:
                os.close(fd)
            control.send(json.dumps({"ok": True, "pid": pid}).encode())
        elif cmd == "cleanup":
            for fd in fds:
                os.close(fd)
            control.send(json.dumps({"ok": True, "killed": kill_leftovers()}).encode())
        else:
            for fd in fds:
                os.close(fd)
            if cmd == "exit":
                break
            control.send(json.dumps({"ok": False, "error": f"unknown command {cmd!r}"}).encode())
    kill_leftovers()


if __name__ == "__main__":
    zygote()
