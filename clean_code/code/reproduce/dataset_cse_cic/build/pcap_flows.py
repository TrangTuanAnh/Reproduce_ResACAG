"""Recover observed endpoints for one day by re-splitting the original pcaps.

Each AWS "TrafficForML" CSV is the concatenated CICFlowMeter-V3 output of a few capture
hosts (victim server first, then capDESKTOP hosts in zip order, cut at 1,048,575 rows).
For every needed host member of the day's pcap.zip this script
  1. downloads the member with parallel HTTP ranges (Hugging Face mirror, S3 fallback),
  2. inflates it as a stream and checks CRC32,
  3. splits packets into flows with the CICFlowMeter-V3 (2018) rules: canonical flow id,
     forward = first packet, first FIN in either direction ends the flow, 120 s activity
     timeout measured from flow start (the continuation keeps the old orientation), flows
     emitted on timeout/EOF need more than one packet,
  4. joins flows to the AWS rows on (Dst Port, Protocol, UTC start second, Tot Fwd/Bwd Pkts,
     TotLen Fwd/Bwd Pkts as IP-length payload) and Flow Duration (exact, then +-20 us).
Candidate pairs for the E-ResGAT target rows are written; one-to-one assignment happens later.

Usage: python -I pcap_flows.py DAY_KEY ZIP_DIR_NAME CD_JSON AWS_CSV ROW_MAP_CSV OUT_DIR [victim members ...]
"""
import io
import json
import os
import re
import struct
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

HF = "https://huggingface.co/datasets/bencorn/CICIDS2018/resolve/main/Original_Network_Traffic_and_Log_data/{}/pcap.zip"
S3 = "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Original%20Network%20Traffic%20and%20Log%20data/{}/pcap.zip"
CHUNK = 8 * 1024 * 1024
TIMEOUT_US = 120_000_000
MAX_DAY_BYTES = 18e9


class Source:
    def __init__(self, day_dir):
        self.hf, self.s3 = HF.format(day_dir), S3.format(day_dir)
        self.signed, self.signed_at = None, 0

    def url(self):
        if self.signed is None or time.time() - self.signed_at > 600:
            try:
                r = requests.head(self.hf, allow_redirects=False, timeout=60)
                self.signed = r.headers["location"] if r.status_code in (301, 302, 307) else self.s3
            except Exception:
                self.signed = self.s3
            self.signed_at = time.time()
        return self.signed

    def fetch(self, lo, hi, session):
        for attempt in range(12):
            url = self.url() if attempt < 6 else self.s3
            try:
                r = session.get(url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=180)
                if r.status_code == 206 and len(r.content) == hi - lo + 1:
                    return r.content
                if r.status_code in (401, 403):
                    self.signed = None
                raise IOError(f"status {r.status_code} len {len(r.content)}")
            except Exception:
                time.sleep(2 + 4 * attempt)
        raise IOError(f"range {lo}-{hi} failed")


def download_member(src, member, path, conns):
    s = requests.Session()
    head = src.fetch(member["local_header_offset"], member["local_header_offset"] + 29, s)
    nlen, xlen = struct.unpack("<HH", head[26:30])
    start = member["local_header_offset"] + 30 + nlen + xlen
    total = member["csize"]
    with open(path, "wb") as f:
        f.truncate(total)
    n = (total + CHUNK - 1) // CHUNK

    def one(i):
        sess = requests.Session()
        lo = start + i * CHUNK
        hi = min(start + total - 1, lo + CHUNK - 1)
        data = src.fetch(lo, hi, sess)
        with open(path, "r+b") as f:
            f.seek(i * CHUNK)
            f.write(data)

    with ThreadPoolExecutor(conns) as ex:
        list(ex.map(one, range(n)))


def inflate_chunks(path, method):
    if method == 0:
        with open(path, "rb") as f:
            while True:
                b = f.read(CHUNK)
                if not b:
                    return
                yield b
    z = zlib.decompressobj(-15)
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            out = z.decompress(b)
            if out:
                yield out
    out = z.flush()
    if out:
        yield out


TS_LO, TS_HI, MAX_INCL = 1517443200, 1522540800, 262144  # captures lie in Feb-Mar 2018 UTC


def plausible(rec, buf, q):
    ts, tu, incl, orig = rec.unpack_from(buf, q)
    return TS_LO <= ts < TS_HI and tu < 1_000_000 and incl <= MAX_INCL and incl <= orig


def resync(rec, buf, start, blen):
    """Next offset holding two consecutive plausible record headers, or -1."""
    i = start
    while True:
        j = buf.find(b"\x5a", i + 3)  # high byte of a little-endian 2018 epoch second
        if j < 0 or j - 3 + 16 > blen:
            return -1
        q = j - 3
        if plausible(rec, buf, q):
            nxt = q + 16 + rec.unpack_from(buf, q)[2]
            if nxt + 16 <= blen and plausible(rec, buf, nxt):
                return q
            if nxt + 16 > blen:
                return -1
        i = q + 1


STATS = {}


def split_flows(chunks):
    """CICFlowMeter-V3 flow splitting over a libpcap byte stream. Returns (flows, crc, size, packets).

    Some members are several pcap files appended; embedded global headers are skipped and a
    corrupt record triggers a resync on the next two consecutive plausible record headers."""
    cur, flows = {}, []
    STATS.clear()
    STATS.update(embedded_headers=0, resyncs=0, skipped_bytes=0)
    buf, pos, crc, size, npk = b"", 0, 0, 0, 0
    header_done, endian = False, "<"
    rec = None

    def emit(fl):
        flows.append(fl[:13])

    for chunk in chunks:
        crc = zlib.crc32(chunk, crc)
        size += len(chunk)
        buf = buf[pos:] + chunk
        pos = 0
        if not header_done:
            if len(buf) < 24:
                continue
            magic = buf[:4]
            if magic == b"\xd4\xc3\xb2\xa1":
                endian = "<"
            elif magic == b"\xa1\xb2\xc3\xd4":
                endian = ">"
            else:
                raise ValueError(f"unsupported capture format {magic.hex()}")
            rec = struct.Struct(endian + "IIII")
            link = struct.unpack(endian + "I", buf[20:24])[0]
            if link != 1:
                raise ValueError(f"link type {link}")
            pos, header_done = 24, True
        blen = len(buf)
        while pos + 16 <= blen:
            if not plausible(rec, buf, pos):
                if buf[pos:pos + 4] in (b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4"):
                    if pos + 24 > blen:
                        break
                    pos += 24
                    STATS["embedded_headers"] += 1
                    continue
                q = resync(rec, buf, pos + 1, blen)
                if q < 0:
                    keep = max(pos, blen - MAX_INCL - 64)
                    STATS["skipped_bytes"] += keep - pos
                    pos = keep
                    break
                STATS["resyncs"] += 1
                STATS["skipped_bytes"] += q - pos
                pos = q
                continue
            ts, tu, incl, _ = rec.unpack_from(buf, pos)
            if pos + 16 + incl > blen:
                break
            p = pos + 16
            pos = p + incl
            npk += 1
            if incl < 34:
                continue
            et = (buf[p + 12] << 8) | buf[p + 13]
            off = p + 14
            if et == 0x8100:
                et = (buf[p + 16] << 8) | buf[p + 17]
                off = p + 18
            if et != 0x0800 or buf[off] >> 4 != 4:
                continue
            ihl = (buf[off] & 15) * 4
            proto = buf[off + 9]
            if proto != 6 and proto != 17:
                continue
            if ((buf[off + 6] & 0x1f) << 8) | buf[off + 7]:
                continue
            totlen = (buf[off + 2] << 8) | buf[off + 3]
            l4 = off + ihl
            end = p + incl
            if end < l4 + 8:
                continue
            sport = (buf[l4] << 8) | buf[l4 + 1]
            dport = (buf[l4 + 2] << 8) | buf[l4 + 3]
            fin = False
            if proto == 6:
                if end < l4 + 14:
                    continue
                hl = (buf[l4 + 12] >> 4) * 4
                fin = bool(buf[l4 + 13] & 1)
            else:
                hl = 8
            pay = (end - l4 - hl) if totlen == 0 else (totlen - ihl - hl)
            if pay < 0:
                pay = 0
            t = ts * 1_000_000 + tu
            s = bytes(buf[off + 12:off + 16])
            d = bytes(buf[off + 16:off + 20])
            # CICFlowMeter-V3 canonical id: compare IP bytes as signed Java bytes
            fwd_canon = True
            for a, b in zip(s, d):
                if a != b:
                    if (a - 256 if a > 127 else a) > (b - 256 if b > 127 else b):
                        fwd_canon = False
                    break
            key = (s, d, sport, dport, proto) if fwd_canon else (d, s, dport, sport, proto)
            fl = cur.get(key)
            if fl is None:
                cur[key] = [s, sport, d, dport, proto, t, t, 1, 0, pay, 0, 0, 0]
                continue
            isfwd = s == fl[0] and sport == fl[1]
            if t - fl[5] > TIMEOUT_US:
                if fl[7] + fl[8] > 1:
                    emit(fl)
                nf = [fl[0], fl[1], fl[2], fl[3], proto, t, t, 0, 0, 0, 0, 0, 0]
                if isfwd:
                    nf[7], nf[9] = 1, pay
                else:
                    nf[8], nf[10] = 1, pay
                cur[key] = nf
                continue
            if isfwd:
                fl[7] += 1
                fl[9] += pay
            else:
                fl[8] += 1
                fl[10] += pay
            fl[6] = t
            if fin:
                emit(fl)
                del cur[key]
    for fl in cur.values():
        if fl[7] + fl[8] > 1:
            emit(fl)
    return flows, crc, size, npk


def aws_keys(aws_csv):
    cols = ["Dst Port", "Protocol", "Timestamp", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
            "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Label"]
    a = pd.read_csv(aws_csv, usecols=cols, dtype=str, low_memory=False)
    a["file_row"] = np.arange(len(a))
    a = a[a["Label"] != "Label"]
    t = pd.to_datetime(a["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    # 12-hour clock without AM/PM: hours 1-7 are afternoon; AST = UTC-4
    hour = t.dt.hour
    t = t + pd.to_timedelta(np.where((hour >= 1) & (hour <= 7), 12, 0), unit="h") + pd.Timedelta(hours=4)
    out = pd.DataFrame({"file_row": a["file_row"].to_numpy(),
                        "sec": (t.astype("int64") // 10**9).to_numpy()})
    for name, c in [("dport", "Dst Port"), ("proto", "Protocol"), ("dur", "Flow Duration"), ("fwd", "Tot Fwd Pkts"),
                    ("bwd", "Tot Bwd Pkts"), ("lfwd", "TotLen Fwd Pkts"), ("lbwd", "TotLen Bwd Pkts")]:
        out[name] = pd.to_numeric(a[c], errors="coerce").to_numpy()
    out = out.dropna()
    return out.astype({k: np.int64 for k in out.columns})


KEYS = ["dport", "proto", "sec", "fwd", "bwd", "lfwd", "lbwd"]


def join(flows, aws, aws_t, member):
    """Candidate flow pairs for the target rows, plus host statistics from semi-joins.

    Statistics use unique key tuples, so floods of identical flows (02-21 HOIC) do not
    blow up into a full rows x flows product."""
    if not flows:
        return pd.DataFrame(), 0, pd.Series(dtype=np.int64)
    f = pd.DataFrame(flows, columns=["src", "sport", "dst", "dport", "proto", "start", "last", "fwd", "bwd",
                                     "lfwd", "lbwd", "_a", "_b"]).drop(columns=["_a", "_b"])
    f["sec"] = f["start"] // 1_000_000
    f["fdur"] = f["last"] - f["start"]
    flows_matched = int(len(f.merge(aws[KEYS].drop_duplicates(), on=KEYS)))
    rows_hit = aws.merge(f[KEYS].drop_duplicates(), on=KEYS)["file_row"]
    exact = aws_t.merge(f, left_on=KEYS + ["dur"], right_on=KEYS + ["fdur"])
    exact["ddur"] = 0
    rest = aws_t[~aws_t["file_row"].isin(exact["file_row"])]
    loose = rest.merge(f, on=KEYS)
    loose["ddur"] = (loose["dur"] - loose["fdur"]).abs()
    loose = loose[loose["ddur"] <= 20]
    # CICFlowMeter-V3 sometimes books one zero-payload packet in the other direction
    # (02-21 server backscatter: AWS 5/2 vs re-split 4/3). Same total packets, same byte
    # totals per direction and identical duration still identify the flow.
    rest = rest[~rest["file_row"].isin(loose["file_row"])]
    swap_keys = ["dport", "proto", "sec", "lfwd", "lbwd"]
    swap = rest.assign(tot=rest["fwd"] + rest["bwd"]).merge(
        f.assign(tot=f["fwd"] + f["bwd"]).drop(columns=["fwd", "bwd"]),
        left_on=swap_keys + ["tot", "dur"], right_on=swap_keys + ["tot", "fdur"])
    swap["ddur"] = 0
    m = pd.concat([exact, loose, swap], ignore_index=True)
    m["src"] = [".".join(map(str, b)) for b in m["src"]]
    m["dst"] = [".".join(map(str, b)) for b in m["dst"]]
    m["member"] = member
    return (m[["file_row", "src", "sport", "dst", "dport", "proto", "start", "fdur", "ddur", "member"]],
            flows_matched, rows_hit)


def block_bounds(rows_hit, width=1000, dense=0.3):
    """Row range of the host's contiguous CSV block: 1000-row bins with >=30 % of rows hit."""
    if not len(rows_hit):
        return -1, -1
    bins = np.bincount(np.asarray(rows_hit, dtype=np.int64) // width)
    full = np.flatnonzero(bins >= dense * width)
    if not len(full):
        return -1, -1
    return int(full.min() * width), int(min(rows_hit.max(), (full.max() + 1) * width - 1))


def main(day, zip_dir, cd_json, aws_csv, row_map_csv, out_dir, *victims):
    os.makedirs(out_dir, exist_ok=True)
    cd = json.load(open(cd_json))
    members = [m for m in cd["members"] if m["csize"] > 0]
    by_name = {m["name"].split("/")[-1]: m for m in members}
    first = [v for v in victims if v != "--more"]
    order = [by_name[v] for v in first]
    rank = lambda m: (0 if re.search(r"172\.31\.69\.", m["name"]) and day in ("02-28", "03-02") else
                      1 if "capDESKTOP" in m["name"] else 2 if "capEC2AMAZ" in m["name"] else 3)
    # priority members (named victims, or the 172.31.69.x subnet on 02-28/03-02) are always all
    # processed: their CSV block order is not guaranteed to follow zip order
    priority = {m["name"] for m in order}
    if not first or "--more" in victims:
        rest = [m for m in members if m not in order]
        order += sorted(rest, key=lambda m: (rank(m), members.index(m)))
        priority |= {m["name"] for m in rest if rank(m) == 0}
    aws = aws_keys(aws_csv)
    rm = pd.read_csv(row_map_csv)
    targets = set(rm.loc[(rm["day"] == day) & (rm["file_row"] >= 0), "file_row"].astype(int))
    aws_t = aws[aws["file_row"].isin(targets)]
    src = Source(zip_dir)
    log = open(os.path.join(out_dir, f"{day}.hosts.jsonl"), "a")
    done_members = set()
    if os.path.exists(os.path.join(out_dir, f"{day}.hosts.jsonl")):
        for line in open(os.path.join(out_dir, f"{day}.hosts.jsonl")):
            done_members.add(json.loads(line)["member"])
    pairs_path = os.path.join(out_dir, f"{day}.pairs.csv")
    matched_targets, low_streak, spent, max_row = set(), 0, 0, -1
    if os.path.exists(pairs_path):
        prev = pd.read_csv(pairs_path)
        matched_targets = set(prev["file_row"])
    for line in (open(os.path.join(out_dir, f"{day}.hosts.jsonl")).read().splitlines()):
        info = json.loads(line)
        if info["match_rate"] >= 0.5:
            max_row = max(max_row, info.get("block_hi", -1))
        spent += info["csize"]
    resume_until = max([i for i, m in enumerate(order) if m["name"] in done_members], default=-1)
    for m in order:
        name = m["name"].split("/")[-1]
        if m["name"] in done_members:
            continue
        if spent > MAX_DAY_BYTES:
            print(day, "byte cap reached", flush=True)
            break
        t0 = time.time()
        tmp = os.path.join(out_dir, f"{day}.member.tmp")
        download_member(src, m, tmp, conns=6)
        t1 = time.time()
        try:
            flows, crc, size, npk = split_flows(inflate_chunks(tmp, m["method"]))
            ok = (f"{crc:08x}" == m["crc32"].lower()) and size == m["usize"]
            err = None
        except Exception as e:
            flows, crc, size, npk, ok, err = [], 0, 0, 0, False, str(e)
        os.remove(tmp)
        spent += m["csize"]
        hit, nmatch, rows_hit = join(flows, aws, aws_t, m["name"])
        if len(hit):
            hit.to_csv(pairs_path, mode="a", header=not os.path.exists(pairs_path), index=False)
            matched_targets |= set(hit["file_row"])
        rate = nmatch / max(1, len(flows))
        info = {"member": m["name"], "csize": m["csize"], "crc_ok": ok, "error": err, "packets": npk, **STATS,
                "flows": len(flows), "flows_matched": nmatch, "match_rate": rate,
                "aws_rows_hit": int(rows_hit.nunique()), "min_row_hit": int(rows_hit.min()) if len(rows_hit) else -1,
                "max_row_hit": int(rows_hit.max()) if len(rows_hit) else -1,
                # robust block bounds: coincidental far-away hits must not end the scan
                "block_lo": block_bounds(rows_hit)[0],
                "block_hi": block_bounds(rows_hit)[1],
                "targets_hit_total": len(matched_targets), "targets": len(targets),
                "download_s": round(t1 - t0, 1), "process_s": round(time.time() - t1, 1)}
        log.write(json.dumps(info) + "\n")
        log.flush()
        print(day, name, {k: info[k] for k in ("crc_ok", "packets", "embedded_headers", "resyncs", "flows", "match_rate", "aws_rows_hit", "min_row_hit",
                                               "max_row_hit", "targets_hit_total", "download_s", "process_s")},
              flush=True)
        if rate >= 0.5:
            max_row = max(max_row, info["block_hi"])
        position = order.index(m)
        if m["name"] in priority and position + 1 < len(order) and order[position + 1]["name"] in priority:
            continue
        if position < resume_until:  # earlier members are still being re-run after a resume
            continue
        low_streak = low_streak + 1 if rate < 0.01 else 0
        beyond = sum(1 for r in targets - matched_targets if r > max_row)
        # tolerate a handful of unmatched tail rows (block bins are 1000 rows wide), not 0.1 % of targets
        if max_row >= 0 and beyond <= max(5, len(targets) // 5000):
            print(day, "processed blocks reach row", max_row, "; unmatched targets beyond:", beyond, flush=True)
            break
        if low_streak >= 4 and max_row >= 0:
            print(day, "four consecutive non-matching hosts; stopping", flush=True)
            break
    print(day, "targets matched", len(matched_targets), "of", len(targets), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
