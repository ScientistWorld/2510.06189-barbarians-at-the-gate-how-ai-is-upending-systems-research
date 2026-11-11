#!/usr/bin/env python3
"""Sandboxed runner for one EPLB policy file (started by evaluate.py).

The evaluator process keeps the workload; this process only ever receives the
window it must rebalance now, so a policy cannot see the window it is scored
on.  For every call the runner measures the rebalance work itself:

* every Python bytecode instruction executed (any thread),
* every call from Python into a C function, plus the size of its first
  argument (Python containers; tensor->Python conversions such as ``tolist``),
* every PyTorch kernel dispatched, plus the tensor elements it reads/writes
  (view ops are free; sort-like kernels pay an extra log factor).

The raw counters go back to the evaluator, which turns them into work units.

Protocol: one JSON request per line on stdin, one JSON reply per line on the
file descriptor given by ``--reply-fd``.  Anything the policy prints goes to
stderr.
"""

from __future__ import annotations

import argparse
import builtins
import gc
import json
import math
import os
import random
import sys
import threading
import time
import types

_DUMPS = json.dumps
_LOADS = json.loads
_LOG2 = math.log2
_PROCESS_TIME_NS = time.process_time_ns
_RANDOM_SEED = getattr(random, "seed")  # seeds the policy RNG before each load

HARNESS_FILE = os.path.abspath(__file__)
MAX_POLICY_BYTES = 64 * 1024
TOOL_ID = 4

# Work weights only matter for the cost cap here; the evaluator recomputes the
# reported cost from the raw counters with the same constants.
W_INSTRUCTION = 1.0
W_C_CALL = 1.0
W_PY_ELEMENT = 2.0
W_KERNEL = 20.0
W_TENSOR_ELEMENT = 0.06
COST_CAP = 5.0e8


class CostCapExceeded(BaseException):
    """Raised inside the policy when one rebalance exceeds the work cap."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--reply-fd", type=int, required=True)
    return parser.parse_args()


sys.dont_write_bytecode = True
ARGS = parse_args()
REPLY = os.fdopen(ARGS.reply_fd, "w", buffering=1)
os.dup2(2, 1)  # policy prints never reach the protocol channel
sys.stdout = sys.stderr

with open(ARGS.policy, "rb") as _f:
    POLICY_BYTES = _f.read(MAX_POLICY_BYTES + 1)
POLICY_PATH = os.path.abspath(ARGS.policy)

import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

torch.set_num_threads(1)

_TB = getattr(torch._C, "TensorBase", None) or torch._C._TensorBase
_TOLIST = _TB.tolist
_NUMEL = _TB.numel
_DIM = _TB.dim
_SIZE = _TB.size
_DTYPE = _TB.dtype.__get__
_DEVICE = _TB.device.__get__
_TO_CPU = _TB.cpu
_TORCH_TENSOR = torch.tensor
_MANUAL_SEED = torch.manual_seed
_INT64 = torch.int64
_INT_DTYPES = (torch.int64, torch.int32, torch.int16, torch.uint8, torch.int8)
_DISPATCH_STACK_LEN = torch._C._len_torch_dispatch_stack
_FUNCTION_STACK_LEN = getattr(torch._C, "_len_torch_function_stack", None)
_THREAD_ACTIVE_COUNT = threading.active_count
import _thread  # noqa: E402

_THREAD_COUNT = _thread._count

TORCH_DIR = os.path.dirname(os.path.abspath(torch.__file__))


def _code_objects(namespace, seen_ids, out):
    """Collect code objects reachable from a module/class namespace."""
    stack = list(namespace.values()) if isinstance(namespace, dict) else list(namespace)
    while stack:
        obj = stack.pop()
        if id(obj) in seen_ids:
            continue
        seen_ids.add(id(obj))
        if isinstance(obj, types.CodeType):
            out.append(obj)
            stack.extend(c for c in obj.co_consts if isinstance(c, types.CodeType))
        elif isinstance(obj, types.FunctionType):
            if getattr(obj, "__module__", None) in _EXCLUDED_MODULE_NAMES:
                stack.append(obj.__code__)
        elif isinstance(obj, types.MethodType):
            stack.append(obj.__func__)
        elif isinstance(obj, (staticmethod, classmethod)):
            stack.append(obj.__func__)
        elif isinstance(obj, property):
            stack.extend(f for f in (obj.fget, obj.fset, obj.fdel) if f is not None)
        elif isinstance(obj, type) and obj.__module__ in _EXCLUDED_MODULE_NAMES:
            stack.extend(vars(obj).values())


_EXCLUDED_MODULE_NAMES = {
    __name__,
    "torch._ops",
    "torch.utils._python_dispatch",
    "torch.utils._pytree",
    "torch.utils._mode_utils",
}
EXCLUDED_CODES: list = []  # keeps the code objects alive so their ids stay unique
EXCLUDED_CODE_IDS: set = set()

# ---------------------------------------------------------------------------
# Counters (raw measurement)
# ---------------------------------------------------------------------------
COUNTS = [0, 0, 0, 0, 0.0]  # instructions, c_calls, py_elements, kernels, tensor_elements
STATE = {"metering": False}


def current_cost() -> float:
    return (
        W_INSTRUCTION * COUNTS[0]
        + W_C_CALL * COUNTS[1]
        + W_PY_ELEMENT * COUNTS[2]
        + W_KERNEL * COUNTS[3]
        + W_TENSOR_ELEMENT * COUNTS[4]
    )


MON = sys.monitoring
_MON = {
    name: getattr(MON, name)
    for name in (
        "use_tool_id",
        "free_tool_id",
        "get_tool",
        "register_callback",
        "set_events",
        "get_events",
        "set_local_events",
        "get_local_events",
        "restart_events",
    )
}
EV = MON.events
DISABLE = MON.DISABLE
METER_EVENTS = EV.INSTRUCTION | EV.CALL

_PY_FUNCTION_TYPES = (types.FunctionType, types.MethodType)
_METHOD_TYPES = (
    types.MethodDescriptorType,
    types.WrapperDescriptorType,
    types.ClassMethodDescriptorType,
)
_SIZED_TYPES = (dict, set, frozenset, str, bytes, bytearray, range,
                type({}.keys()), type({}.values()), type({}.items()))
_ITER_TYPES = tuple(
    type(x)
    for x in (iter([]), iter(()), iter(range(0)), iter(""), reversed([]), iter({}), iter(set()))
)
# Calls whose work grows with the size of the container they are given.
_SCAN_FUNCTIONS = frozenset({
    "sorted", "sum", "min", "max", "any", "all", "fsum", "prod", "heapify",
    "nlargest", "nsmallest", "tensor", "as_tensor", "LongTensor", "IntTensor",
    "FloatTensor", "DoubleTensor", "copy", "deepcopy", "Counter", "deque",
    "list", "tuple", "set", "frozenset", "dict", "bytes", "bytearray", "array",
    "OrderedDict", "fromkeys", "accumulate_all",
})
_SCAN_METHODS = frozenset({
    "sort", "index", "count", "copy", "reverse", "insert", "remove", "union",
    "intersection", "difference", "symmetric_difference", "issubset",
    "issuperset", "isdisjoint", "tolist", "numpy", "__array__", "tobytes",
    "__copy__", "__deepcopy__",
})
_LOG_N = frozenset({"heappush", "heappop", "heapreplace", "heappushpop", "bisect",
                    "bisect_left", "bisect_right", "insort", "insort_left", "insort_right"})
_N_LOG_N = frozenset({"sorted", "sort", "nlargest", "nsmallest"})
TENSOR_TYPE = torch.Tensor


def _container_size(obj) -> int:
    t = type(obj)
    if t is list or t is tuple:
        n = len(obj)
        if n:
            first = obj[0]
            ft = type(first)
            if ft is list or ft is tuple:
                inner = len(first)
                if inner and type(first[0]) in (list, tuple):
                    inner *= len(first[0])
                n *= max(1, inner)
        return n
    if t in _SIZED_TYPES:
        return len(obj)
    if t in _ITER_TYPES:
        return obj.__length_hint__()
    return 0


def _on_instruction(code, offset):
    if id(code) in EXCLUDED_CODE_IDS:
        return DISABLE
    COUNTS[0] += 1
    if not (COUNTS[0] & 0x3FF) and current_cost() > COST_CAP:
        raise CostCapExceeded("rebalance work cap exceeded")
    return None


def _size(obj) -> int:
    if type(obj) is TENSOR_TYPE:
        return _NUMEL(obj)
    return _container_size(obj)


def _on_call(code, offset, func, arg0):
    if id(code) in EXCLUDED_CODE_IDS:
        return DISABLE
    ft = type(func)
    if ft in _PY_FUNCTION_TYPES:
        return None
    COUNTS[1] += 1
    n = 0
    name = ""
    if ft is types.BuiltinFunctionType:
        name = func.__name__
        owner = func.__self__
        if type(owner) is TENSOR_TYPE or type(owner) in (list, tuple) or type(owner) in _SIZED_TYPES:
            if name in _SCAN_METHODS:
                n = _size(owner)
        elif name in _SCAN_FUNCTIONS or name in _LOG_N:
            n = _container_size(arg0) if type(arg0) is not TENSOR_TYPE else 0
    elif ft in _METHOD_TYPES:
        name = func.__name__
        if name in _SCAN_METHODS:
            n = _size(arg0)
    elif ft is type:
        name = func.__name__
        if name in _SCAN_FUNCTIONS:
            n = _container_size(arg0)
    if n:
        if name in _N_LOG_N:
            n = n * max(1.0, _LOG2(n))
        elif name in _LOG_N:
            n = max(1.0, _LOG2(n + 1))
        COUNTS[2] += n
    return None


_SORT_OPS = frozenset({"sort", "argsort", "topk", "kthvalue", "median", "msort",
                       "_unique2", "unique_dim", "unique_consecutive", "nanmedian", "mode"})
_OP_INFO: dict = {}


def _op_info(func):
    info = _OP_INFO.get(id(func))
    if info is None:
        is_view = False
        sort_like = False
        try:
            schema = func._schema
            for arg in schema.arguments:
                alias = arg.alias_info
                if alias is not None and not alias.is_write:
                    is_view = True
                    break
            sort_like = schema.name.split("::")[-1] in _SORT_OPS
        except Exception:  # noqa: BLE001 - unknown schema: charge as a normal kernel
            pass
        info = (is_view, sort_like, func)
        _OP_INFO[id(func)] = info
    return info


def _numel_all(values) -> int:
    n = 0
    for v in values:
        if isinstance(v, TENSOR_TYPE):
            n += _NUMEL(v)
        elif type(v) in (list, tuple):
            for w in v:
                if isinstance(w, TENSOR_TYPE):
                    n += _NUMEL(w)
    return n


class KernelMeter(TorchDispatchMode):
    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        COUNTS[3] += 1
        is_view, sort_like, _ = _op_info(func)
        if not is_view:
            n = _numel_all(args) + _numel_all(kwargs.values())
            n += _numel_all(out if type(out) in (list, tuple) else (out,))
            if sort_like:
                last = 2
                for a in args:
                    if isinstance(a, TENSOR_TYPE) and _DIM(a) > 0:
                        last = max(2, _SIZE(a)[-1])
                        break
                n = n * _LOG2(last)
            COUNTS[4] += n
        if current_cost() > COST_CAP:
            raise CostCapExceeded("rebalance work cap exceeded")
        return out


_MON["use_tool_id"](TOOL_ID, "eplb-work-meter")
_MON["register_callback"](TOOL_ID, EV.INSTRUCTION, _on_instruction)
_MON["register_callback"](TOOL_ID, EV.CALL, _on_call)


# ---------------------------------------------------------------------------
# Guards: no hidden work outside the metered interpreter thread
# ---------------------------------------------------------------------------
GUARD = {"active": False, "violation": None, "in_hook": False}
_ALLOWED_READ = tuple(
    sorted(
        {
            os.path.realpath(p) + os.sep
            for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, TORCH_DIR)
            + tuple(p for p in sys.path if p and "site-packages" in p)
        }
    )
) + ("/proc/", "/sys/", "/etc/", "/usr/lib/", "/usr/local/lib/", "/usr/share/")
_BLOCKED_EVENTS = frozenset(
    {
        "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
        "os.fork", "os.forkpty", "pty.spawn", "os.kill", "os.killpg",
        "ctypes.dlopen", "ctypes.dlsym", "ctypes.dlsym/handle", "ctypes.call_function",
        "ctypes.cdata", "ctypes.addressof", "ctypes.string_at", "ctypes.wstring_at",
        "socket.__new__", "socket.connect", "socket.bind", "socket.getaddrinfo",
        "mmap.__new__", "gc.get_objects", "gc.get_referrers", "gc.get_referents",
        "os.remove", "os.rename", "os.mkdir", "os.rmdir", "os.truncate", "os.chmod",
        "os.symlink", "os.link", "shutil.copyfile", "shutil.rmtree", "sys.setprofile",
        "sys.settrace", "sys._current_frames", "sys.setrecursionlimit",
    }
)


def _violation(message: str) -> None:
    if GUARD["violation"] is None:
        GUARD["violation"] = message
    raise PermissionError(message)


def _audit(event, args):
    if not GUARD["active"] or GUARD["in_hook"]:
        return
    GUARD["in_hook"] = True
    try:
        if event == "open":
            path, mode = args[0], args[1]
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC)
            )
            if not isinstance(path, (str, bytes, os.PathLike)):
                _violation(f"policy opened a raw file descriptor ({path!r})")
            real = os.path.realpath(os.fsdecode(path))
            if writing and real != "/dev/null":
                _violation(f"policy may not write files ({real})")
            if not real.startswith(_ALLOWED_READ) and real not in ("/dev/null", "/dev/urandom"):
                _violation(f"policy may not read files outside the Python installation ({real})")
        elif event in _BLOCKED_EVENTS:
            _violation(f"policy used a blocked operation: {event}")
        elif event == "sys._getframe":
            frame = args[0]
            if frame is not None and frame.f_code.co_filename == HARNESS_FILE:
                _violation("policy inspected the evaluator's frames")
    finally:
        GUARD["in_hook"] = False


sys.addaudithook(_audit)


def _blocked(name):
    def blocked(*args, **kwargs):
        if GUARD["active"]:
            _violation(f"policy called blocked API {name}")
        raise RuntimeError(f"{name} is unavailable here")
    return blocked


_ORIG_THREAD_START = threading.Thread.start


def _thread_start(self, *args, **kwargs):
    if GUARD["active"]:
        _violation("policy may not start threads")
    return _ORIG_THREAD_START(self, *args, **kwargs)


threading.Thread.start = _thread_start
for _name in ("set_events", "set_local_events", "register_callback", "use_tool_id",
              "free_tool_id", "restart_events"):
    setattr(MON, _name, _blocked(f"sys.monitoring.{_name}"))
torch.Tensor.numpy = _blocked("Tensor.numpy")
torch.Tensor.__array__ = _blocked("Tensor.__array__")
torch.jit.script = _blocked("torch.jit.script")
torch.jit.trace = _blocked("torch.jit.trace")
torch.jit.trace_module = _blocked("torch.jit.trace_module")
torch.compile = _blocked("torch.compile")
torch.ops.load_library = _blocked("torch.ops.load_library")

_GETFRAME = sys._getframe


def _guard_bypass(owner, attr: str) -> None:
    """Wrap a torch API that can switch kernel metering off.

    Torch's own Python code may still use it (e.g. tensor printing); a call
    from policy code while metering is a violation.
    """
    original = getattr(owner, attr, None)
    if original is None:
        return

    def guarded(*args, **kwargs):
        if GUARD["active"]:
            caller = _GETFRAME(1).f_code.co_filename
            if not caller.startswith(TORCH_DIR + os.sep):
                _violation(f"policy called {attr}, which bypasses the work meter")
        return original(*args, **kwargs)

    try:
        setattr(owner, attr, guarded)
    except (AttributeError, TypeError):
        pass


import torch._ops  # noqa: E402
import torch.utils._mode_utils  # noqa: E402
import torch.utils._python_dispatch as _tpd  # noqa: E402

for _owner, _attr in (
    (torch._C, "_DisableTorchDispatch"),
    (torch._C, "_pop_torch_dispatch_stack"),
    (torch._C, "_set_dispatch_mode"),
    (torch._C, "_unset_dispatch_mode"),
    (torch._C, "_ExcludeDispatchKeyGuard"),
    (torch._C, "_IncludeDispatchKeyGuard"),
    (torch._C, "_ForceDispatchKeyGuard"),
    (torch._C, "_PreserveDispatchKeyGuard"),
    (torch._C, "_dispatch_tls_set_dispatch_key_excluded"),
    (torch._C, "_dispatch_tls_set_dispatch_key_included"),
    (torch._C, "_RestorePythonTLSSnapshot"),
    (_tpd, "_pop_mode"),
    (_tpd, "_pop_mode_temporarily"),
    (_tpd, "_disable_current_modes"),
    (torch.utils._mode_utils, "no_dispatch"),
    (torch._ops.OpOverload, "redispatch"),
):
    _guard_bypass(_owner, _attr)

_BLOCKED_IMPORTS = frozenset(
    {
        "ctypes", "_ctypes", "subprocess", "_posixsubprocess", "multiprocessing",
        "_multiprocessing", "threading", "_thread", "concurrent", "asyncio", "socket",
        "_socket", "ssl", "select", "selectors", "signal", "mmap", "gc", "importlib",
        "imp", "zipimport", "pkgutil", "runpy", "sys", "os", "posix", "nt", "inspect",
        "tracemalloc", "faulthandler", "resource", "pty", "tty", "termios", "fcntl",
        "shutil", "tempfile", "pickle", "_pickle", "marshal", "code", "codeop", "pdb",
        "bdb", "trace", "cProfile", "profile", "_lsprof", "sysconfig", "site", "io",
        "_io", "pathlib", "glob", "fileinput", "linecache", "sqlite3", "_sqlite3",
        "urllib", "http", "ftplib", "smtplib", "xmlrpc", "webbrowser", "builtins",
        "_xxsubinterpreters", "_interpreters", "_xxinterpchannels", "atexit",
        "json", "_json", "time", "argparse", "weakref", "_weakref", "ast", "dis",
    }
)
_ALLOWED_TOP = (frozenset(sys.stdlib_module_names) - _BLOCKED_IMPORTS) | {"torch"}
_REAL_IMPORT = builtins.__import__


def _policy_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level:
        raise ImportError("a policy file must be self-contained (no relative imports)")
    top = name.partition(".")[0]
    if top not in _ALLOWED_TOP:
        if GUARD["active"]:
            GUARD["violation"] = GUARD["violation"] or f"policy imported a disallowed module: {name}"
        raise ImportError(f"policy may only import torch and computational stdlib modules, not {name!r}")
    return _REAL_IMPORT(name, globals, locals, fromlist, level)


_PY_BUILTINS_SRC = '''
def map(function, iterable, *iterables):
    if iterables:
        for items in zip(iterable, *iterables):
            yield function(*items)
    else:
        for item in iterable:
            yield function(item)


def filter(function, iterable):
    if function is None:
        for item in iterable:
            if item:
                yield item
    else:
        for item in iterable:
            if function(item):
                yield item
'''
_py_builtins: dict = {}
exec(compile(_PY_BUILTINS_SRC, "<eplb-policy-builtins>", "exec"), _py_builtins)
POLICY_BUILTINS = dict(builtins.__dict__)
POLICY_BUILTINS.update(
    {
        "__import__": _policy_import,
        "map": _py_builtins["map"],
        "filter": _py_builtins["filter"],
        "open": _blocked("open"),
        "input": _blocked("input"),
        "breakpoint": _blocked("breakpoint"),
        "exit": _blocked("exit"),
        "quit": _blocked("quit"),
    }
)


# ---------------------------------------------------------------------------
# Policy lifecycle
# ---------------------------------------------------------------------------
def load_policy():
    if len(POLICY_BYTES) > MAX_POLICY_BYTES:
        raise ValueError(f"policy file exceeds {MAX_POLICY_BYTES} bytes")
    source = POLICY_BYTES.decode("utf-8")
    _RANDOM_SEED(0)
    _MANUAL_SEED(0)
    module = types.ModuleType("eplb_policy")
    module.__dict__["__builtins__"] = POLICY_BUILTINS
    module.__dict__["__file__"] = POLICY_PATH
    sys.modules["eplb_policy"] = module
    GUARD["active"] = True
    try:
        exec(compile(source, POLICY_PATH, "exec"), module.__dict__)
    finally:
        GUARD["active"] = False
    func = module.__dict__.get("rebalance_experts")
    if not callable(func):
        raise ValueError("policy does not define a callable rebalance_experts")
    return func


def threads_clean() -> bool:
    return _THREAD_ACTIVE_COUNT() == 1 and _THREAD_COUNT() == 0


def modes_clean() -> bool:
    if _DISPATCH_STACK_LEN() != 0:
        return False
    return _FUNCTION_STACK_LEN is None or _FUNCTION_STACK_LEN() == 0


def run_call(func, request: dict, metered: bool) -> dict:
    weight = _TORCH_TENSOR(request["load"], dtype=_INT64)
    args = (
        weight,
        int(request["num_replicas"]),
        int(request["num_groups"]),
        int(request["num_nodes"]),
        int(request["num_gpus"]),
    )
    num_layers, num_experts = len(request["load"]), len(request["load"][0])
    for i in range(len(COUNTS)):
        COUNTS[i] = 0
    GUARD["violation"] = None
    error = None
    result = None
    cpu_ns = 0
    gc.collect()
    gc.disable()  # no collector-triggered finalizers inside the measured call
    if metered:
        meter = KernelMeter()
        meter.__enter__()
        GUARD["active"] = True
        _MON["set_events"](TOOL_ID, METER_EVENTS)
        try:
            result = func(*args)
        except CostCapExceeded as exc:
            error = f"cost cap: {exc}"
        except BaseException as exc:  # noqa: BLE001 - report every policy failure
            error = f"{type(exc).__name__}: {exc}"
        finally:
            _MON["set_events"](TOOL_ID, 0)
            GUARD["active"] = False
            meter.__exit__(None, None, None)
    else:
        GUARD["active"] = True
        start = _PROCESS_TIME_NS()
        try:
            result = func(*args)
        except BaseException as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        finally:
            cpu_ns = _PROCESS_TIME_NS() - start
            GUARD["active"] = False

    gc.enable()
    reply = {
        "ok": False,
        "counts": list(COUNTS),
        "cpu_ns": cpu_ns,
    }
    if GUARD["violation"]:
        reply["error"] = f"guard: {GUARD['violation']}"
        return reply
    if _MON["get_tool"](TOOL_ID) != "eplb-work-meter":
        reply["error"] = "guard: work meter was tampered with"
        return reply
    if not threads_clean():
        reply["error"] = "guard: policy left extra threads running"
        return reply
    if not modes_clean():
        reply["error"] = "guard: policy left a torch dispatch/function mode active"
        return reply
    if error is not None:
        reply["error"] = error
        return reply
    try:
        reply.update(serialize(result, (num_layers, num_experts)))
        reply["ok"] = True
    except Exception as exc:  # noqa: BLE001
        reply["error"] = f"invalid output: {exc}"
    return reply


def _as_int_tensor(value, name: str, ndim: int):
    if type(value) is not TENSOR_TYPE:
        raise TypeError(f"{name} must be a plain torch.Tensor, got {type(value).__name__}")
    if _DTYPE(value) not in _INT_DTYPES:
        raise TypeError(f"{name} must have an integer dtype")
    if _DIM(value) != ndim:
        raise ValueError(f"{name} must have {ndim} dims, got shape {tuple(_SIZE(value))}")
    if _DEVICE(value).type != "cpu":
        value = _TO_CPU(value)
    return value


def serialize(result, weight_shape) -> dict:
    if not isinstance(result, tuple) or len(result) != 3:
        raise TypeError("rebalance_experts must return (phy2log, log2phy, logcnt)")
    phy2log = _as_int_tensor(result[0], "phy2log", 2)
    log2phy = _as_int_tensor(result[1], "log2phy", 3)
    logcnt = _as_int_tensor(result[2], "logcnt", 2)
    num_layers, num_experts = int(weight_shape[0]), int(weight_shape[1])
    if tuple(_SIZE(logcnt)) != (num_layers, num_experts):
        raise ValueError(f"logcnt shape {tuple(_SIZE(logcnt))} != {(num_layers, num_experts)}")
    l2p_shape = tuple(_SIZE(log2phy))
    if l2p_shape[0] != num_layers or l2p_shape[1] != num_experts:
        raise ValueError(f"log2phy shape {l2p_shape} does not start with {(num_layers, num_experts)}")
    if _NUMEL(phy2log) > 1_000_000 or _NUMEL(log2phy) > 20_000_000:
        raise ValueError("output tensors are implausibly large")
    cnt = _TOLIST(logcnt)
    width = int(l2p_shape[2])
    rows = _TOLIST(log2phy)
    slots = []
    padding_ok = True
    for layer in range(num_layers):
        layer_slots = []
        for expert in range(num_experts):
            c = cnt[layer][expert]
            entries = rows[layer][expert]
            if c < 0 or c > width:
                padding_ok = False
                layer_slots.append([])
                continue
            layer_slots.append(entries[:c])
            if any(v != -1 for v in entries[c:]):
                padding_ok = False
        slots.append(layer_slots)
    return {
        "phy2log": _TOLIST(phy2log),
        "logcnt": cnt,
        "log2phy_slots": slots,
        "log2phy_padding_ok": padding_ok,
        "log2phy_width": width,
    }


def _collect_excluded_codes() -> None:
    import torch._ops
    import torch.utils._mode_utils
    import torch.utils._python_dispatch
    import torch.utils._pytree

    seen: set = set()
    for module in (
        sys.modules[__name__],
        torch._ops,
        torch.utils._python_dispatch,
        torch.utils._pytree,
        torch.utils._mode_utils,
    ):
        _code_objects(vars(module), seen, EXCLUDED_CODES)
    EXCLUDED_CODE_IDS.update(id(c) for c in EXCLUDED_CODES)


def main() -> None:
    _collect_excluded_codes()
    func = None
    for line in sys.stdin.buffer:
        request = _LOADS(line)
        cmd = request.get("cmd")
        if cmd == "load":
            try:
                GUARD["violation"] = None
                func = load_policy()
                if GUARD["violation"]:
                    raise PermissionError(GUARD["violation"])
                if not threads_clean():
                    raise PermissionError("policy started threads at import")
                reply = {"ok": True}
            except BaseException as exc:  # noqa: BLE001
                func = None
                reply = {"ok": False, "error": f"load failed: {type(exc).__name__}: {exc}"}
        elif cmd == "call":
            if func is None:
                reply = {"ok": False, "error": "policy not loaded"}
            else:
                reply = run_call(func, request, bool(request.get("metered", True)))
        elif cmd == "exit":
            break
        else:
            reply = {"ok": False, "error": f"unknown command {cmd!r}"}
        REPLY.write(_DUMPS(reply, separators=(",", ":")) + "\n")
        REPLY.flush()


if __name__ == "__main__":
    main()
