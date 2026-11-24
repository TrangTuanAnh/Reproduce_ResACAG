"""List members of a remote (ZIP64) archive via HTTP Range; write JSON with data offsets.

Usage: python -I zip_remote.py URL [OUT_JSON]   (prints to stdout without OUT_JSON)
"""
import json, struct, sys
import requests

URL = sys.argv[1]


def get(a, b):
    r = requests.get(URL, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
    assert r.status_code == 206, r.status_code
    return r.content


size = int(requests.head(URL, timeout=60).headers["Content-Length"])
tail = get(size - 65536, size - 1)
eocd = tail.rfind(b"PK\x05\x06")
loc = tail.rfind(b"PK\x06\x07")
z64_off = struct.unpack("<Q", tail[loc + 8:loc + 16])[0]
z64 = get(z64_off, z64_off + 55)
cd_count, cd_size, cd_off = struct.unpack("<QQQ", z64[32:56])
cd = get(cd_off, cd_off + cd_size - 1)
out, p = [], 0
for _ in range(cd_count):
    (sig, _, _, flag, method, _, _, crc, csize, usize, nlen, xlen, clen, _, _, _, lho) = struct.unpack(
        "<IHHHHHHIIIHHHHHII", cd[p:p + 46])
    name = cd[p + 46:p + 46 + nlen].decode()
    extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
    q = 0
    while q < len(extra):
        hid, hlen = struct.unpack("<HH", extra[q:q + 4])
        if hid == 1:
            vals, r = [], q + 4
            for field in ("usize", "csize", "lho"):
                cur = {"usize": usize, "csize": csize, "lho": lho}[field]
                if cur == 0xFFFFFFFF:
                    vals.append((field, struct.unpack("<Q", extra[r:r + 8])[0]))
                    r += 8
            for field, v in vals:
                if field == "usize": usize = v
                if field == "csize": csize = v
                if field == "lho": lho = v
        q += 4 + hlen
    lh = get(lho, lho + 29)
    lnl, lxl = struct.unpack("<HH", lh[26:30])
    data = lho + 30 + lnl + lxl
    out.append({"name": name, "method": method, "flag": flag, "crc32": crc, "compressed": csize,
                "uncompressed": usize, "local_header": lho, "data_start": data, "data_end": data + csize - 1})
    p += 46 + nlen + xlen + clen
listing = json.dumps({"url": URL, "size": size, "members": out}, indent=1)
if len(sys.argv) > 2:
    with open(sys.argv[2], "w", encoding="utf-8") as stream:
        stream.write(listing)
else:
    print(listing)
