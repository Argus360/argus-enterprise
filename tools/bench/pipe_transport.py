"""How much CPU does one 2592x1520 raw frame cost to move from ffmpeg into
Python, per transport? Run on the OS under test (CHI is Windows)."""
import socket
import subprocess
import sys
import time

import numpy as np

FF = sys.argv[1]
W, H = 2592, 1520
N = W * H * 3
FRAMES = 12
SRC = ["-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate=4", "-frames:v", str(FRAMES),
       "-an", "-f", "rawvideo", "-pix_fmt", "bgr24"]


def measure(label, read_frame, start, stop):
    ctx = start()
    c0 = time.thread_time()
    t0 = time.perf_counter()
    n = 0
    while n < FRAMES:
        if read_frame(ctx) is None:
            break
        n += 1
    cpu = time.thread_time() - c0
    wall = time.perf_counter() - t0
    stop(ctx)
    print(f"{label:34} frames={n} cpu/frame={1000 * cpu / max(n, 1):8.1f} ms "
          f"wall/frame={1000 * wall / max(n, 1):8.1f} ms", flush=True)


def a_start():
    return subprocess.Popen([FF, "-hide_banner", "-loglevel", "error", *SRC, "-"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)


def a_read(p):
    buf = bytearray()
    while len(buf) < N:
        ch = p.stdout.read(N - len(buf))
        if not ch:
            return None
        buf += ch
    return np.frombuffer(bytes(buf), np.uint8).reshape(H, W, 3).copy()


def a_stop(p):
    p.kill()
    p.wait()


c_stop = a_stop


def c_start():
    return subprocess.Popen([FF, "-hide_banner", "-loglevel", "error", *SRC, "-"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1 << 22)


def c_read(p):
    buf = bytearray(N)
    mv = memoryview(buf)
    got = 0
    while got < N:
        k = p.stdout.readinto(mv[got:])
        if not k:
            return None
        got += k
    return np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()


def b_start():
    ls = socket.socket()
    ls.bind(("127.0.0.1", 0))
    ls.listen(1)
    port = ls.getsockname()[1]
    p = subprocess.Popen([FF, "-hide_banner", "-loglevel", "error", *SRC, f"tcp://127.0.0.1:{port}"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ls.settimeout(15)
    s, _ = ls.accept()
    ls.close()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
    return (p, s, bytearray(N))


def b_read(ctx):
    p, s, buf = ctx
    mv = memoryview(buf)
    got = 0
    while got < N:
        k = s.recv_into(mv[got:], N - got)
        if k <= 0:
            return None
        got += k
    return np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()


def b_stop(ctx):
    p, s, _ = ctx
    s.close()
    p.kill()
    p.wait()


print(sys.platform, sys.version.split()[0], flush=True)
for _ in range(2):
    measure("A pipe bufsize=0 read() (engine)", a_read, a_start, a_stop)
    measure("C pipe bufsize=4MB readinto", c_read, c_start, c_stop)
    measure("B tcp loopback recv_into", b_read, b_start, b_stop)
