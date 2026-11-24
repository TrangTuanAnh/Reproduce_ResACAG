"""Fetch one member of the DistriNet improved CSE-CIC-IDS2018 zip by HTTP Range,
stream-inflate it (CRC-checked), and keep only the flows that can match an
E-ResGAT target row: same (Dst Port, Protocol) and start second within +-3 s of
a target (both 12-hour readings of the AWS timestamp are allowed).

Usage: python -I fetch_distrinet.py MEMBER_NAME DAY_KEY CENTRAL_DIR_JSON MATCHED_RAW_PARQUET OUT_DIR
"""
import io
import json
import os
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

CHUNK = 8 * 1024 * 1024
KEEP = ["id", "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp", "Flow Duration",
        "Total Fwd Packet", "Total Bwd packets", "Total Length of Fwd Packet", "Total Length of Bwd Packet",
        "Fwd Packet Length Max", "Bwd Packet Length Max", "FWD Init Win Bytes", "Bwd Init Win Bytes",
        "Fwd Seg Size Min", "Fwd Act Data Pkts", "Label", "Attempted Category"]


def aws_to_utc_candidates(ts):
    """AWS CSVs: dd/MM/yyyy hh:mm:ss, 12-hour clock without AM/PM, AST (UTC-4)."""
    t = pd.to_datetime(ts, format="%d/%m/%Y %H:%M:%S")
    am = t + pd.Timedelta(hours=4)
    pm = t + pd.Timedelta(hours=16)
    return am, pm


def key(sec, dport, proto):
    return (sec.astype(np.int64) * 65536 + dport.astype(np.int64)) * 256 + proto.astype(np.int64)


def download(url, a, b, path, conns=16):
    total = b - a + 1
    done = path + ".done"
    have = set()
    if os.path.exists(done):
        have = set(int(x) for x in open(done).read().split())
    if not os.path.exists(path) or os.path.getsize(path) != total:
        with open(path, "wb") as f:
            f.truncate(total)
        have = set()
    todo = [i for i in range((total + CHUNK - 1) // CHUNK) if i not in have]

    def one(i):
        s = requests.Session()
        lo, hi = a + i * CHUNK, min(b, a + (i + 1) * CHUNK - 1)
        for attempt in range(10):
            try:
                r = s.get(url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=180)
                if r.status_code != 206 or len(r.content) != hi - lo + 1:
                    raise IOError(f"bad response {r.status_code} {len(r.content)}")
                with open(path, "r+b") as f:
                    f.seek(lo - a)
                    f.write(r.content)
                return i
            except Exception:
                time.sleep(3 + 5 * attempt)
        raise IOError(f"chunk {i} failed")

    t0 = time.time()
    with ThreadPoolExecutor(conns) as pool, open(done, "a") as log:
        for i in pool.map(one, todo):
            log.write(f"{i}\n")
            log.flush()
    print(f"downloaded {total/1e6:.0f} MB in {time.time()-t0:.0f}s", flush=True)


def main(member, day, cd_json, matched_parquet, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cd = json.load(open(cd_json))
    m = next(x for x in cd["members"] if x["name"] == member)
    targets = pd.read_parquet(matched_parquet, columns=["day", "Dst Port", "Protocol", "Timestamp"])
    targets = targets[targets["day"] == day]
    am, pm = aws_to_utc_candidates(targets["Timestamp"])
    dport = pd.to_numeric(targets["Dst Port"]).to_numpy()
    proto = pd.to_numeric(targets["Protocol"]).to_numpy()
    keys = []
    for base in (am, pm):
        sec = (base.astype("int64") // 10**9).to_numpy()
        for delta in range(-3, 4):
            keys.append(key(sec + delta, dport, proto))
    target_keys = np.unique(np.concatenate(keys))

    path = os.path.join(out_dir, member + ".deflate")
    download(cd["url"], m["data_start"], m["data_end"], path)
    inflater = zlib.decompressobj(-15)
    crc, produced, rest, header, kept, rows, labels = 0, 0, b"", None, [], 0, {}
    t0 = time.time()

    def parse(block):
        nonlocal header, rows
        if header is None:
            first, block = block.split(b"\n", 1)
            header = first.decode().rstrip("\r").split(",")
        if not block:
            return
        frame = pd.read_csv(io.BytesIO(block), header=None, names=header, usecols=KEEP, dtype=str,
                            low_memory=False)
        rows += len(frame)
        for lab, n in frame["Label"].value_counts().items():
            labels[lab] = labels.get(lab, 0) + int(n)
        ts = pd.to_datetime(frame["Timestamp"], format="ISO8601")
        sec = (ts.astype("int64") // 10**9).to_numpy()
        k = key(sec, pd.to_numeric(frame["Dst Port"]).to_numpy(), pd.to_numeric(frame["Protocol"]).to_numpy())
        keep = np.isin(k, target_keys)
        if keep.any():
            kept.append(frame[keep])

    with open(path, "rb") as f:
        while True:
            data = f.read(CHUNK)
            if not data:
                break
            out = inflater.decompress(data)
            crc = zlib.crc32(out, crc)
            produced += len(out)
            buf = rest + out
            cut = buf.rfind(b"\n")
            if cut < 0:
                rest = buf
                continue
            parse(buf[:cut + 1])
            rest = buf[cut + 1:]
        out = inflater.flush()
        crc = zlib.crc32(out, crc)
        produced += len(out)
        tail = rest + out
        if tail.strip():
            parse(tail if tail.endswith(b"\n") else tail + b"\n")
    if crc != m["crc32"] or produced != m["uncompressed"]:
        raise ValueError(f"CRC/size mismatch {crc} {m['crc32']} {produced} {m['uncompressed']}")
    result = pd.concat(kept, ignore_index=True)
    result.to_parquet(os.path.join(out_dir, f"{day}.parquet"), index=False)
    stats = {"member": member, "day": day, "rows_total": rows, "rows_kept": len(result), "labels": labels,
             "crc_ok": True, "uncompressed": produced, "seconds_parse": time.time() - t0}
    json.dump(stats, open(os.path.join(out_dir, f"{day}.stats.json"), "w"), indent=1)
    os.remove(path)
    os.remove(path + ".done")
    print(json.dumps({k: v for k, v in stats.items() if k != "labels"}), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:6])
