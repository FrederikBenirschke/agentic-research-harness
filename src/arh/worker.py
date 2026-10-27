"""The model worker. Runs in its own sandboxed interpreter (arh.score starts
it) and imports nothing from arh, so the harness's data never shares a process
with model code.

Protocol on stdin/stdout, frames of 1 type byte + 8-byte length + Arrow IPC:
  F <train>   fit; reply K
  P <today>   today's rows without y; reply Y <yhat>
  R <today>   today's rows with y, appended to history; no reply
  N <any>     reply N <json>: the [name, claim] of each piece of the model's MODEL, or null if it has none
Any exception replies E <traceback>.
"""

import importlib.util
import io
import json
import os
import struct
import sys
import traceback

import numpy as np
import polars as pl


def read(f):
    head = f.read(9)
    if len(head) < 9:
        return None, None
    return head[:1], f.read(struct.unpack("<Q", head[1:])[0])


def write(f, kind: bytes, payload: bytes = b"") -> None:
    f.write(kind + struct.pack("<Q", len(payload)) + payload)
    f.flush()


def main(model_path: str) -> None:
    # Private copies of the protocol pipes; the model's prints go to stderr.
    inp, out = os.fdopen(os.dup(0), "rb"), os.fdopen(os.dup(1), "wb")
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    os.dup2(2, 1)
    try:
        spec = importlib.util.spec_from_file_location("model", model_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        state, history = None, None
        while True:
            kind, payload = read(inp)
            if kind is None:
                return
            df = pl.read_ipc(io.BytesIO(payload))
            if kind == b"N":
                names = getattr(getattr(mod, "MODEL", None), "names", None)
                write(out, b"N", json.dumps(names() if callable(names) else None).encode())
            elif kind == b"F":
                history, state = df, mod.fit(df)
                write(out, b"K")
            elif kind == b"P":
                yhat = np.asarray(mod.predict(state, history, df), dtype=float).ravel()
                if yhat.shape != (df.height,) or not np.isfinite(yhat).all():
                    raise ValueError(f"predict must return {df.height} finite numbers, got shape {yhat.shape}")
                buf = io.BytesIO()
                pl.DataFrame({"yhat": yhat}).write_ipc(buf)
                write(out, b"Y", buf.getvalue())
            elif kind == b"R":
                history = pl.concat([history, df], how="vertical_relaxed")
    except BaseException:
        write(out, b"E", traceback.format_exc()[-3000:].encode())


if __name__ == "__main__":
    main(sys.argv[1])
