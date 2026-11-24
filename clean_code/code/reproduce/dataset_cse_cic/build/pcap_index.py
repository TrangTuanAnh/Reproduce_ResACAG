"""Read remote ZIP central directory via HTTP Range requests (no full download).
Usage: python -I pcap_index.py <pcap.zip url> <out_json>
"""
import sys, struct, json, urllib.request, urllib.parse

def rng(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()

def head(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"]), dict(r.headers)

def main(url, out):
    size, hdrs = head(url)
    tail_len = min(size, 1 << 20)
    tail = rng(url, size - tail_len, size - 1)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise SystemExit("no EOCD")
    (sig, disk, cd_disk, n_disk, n_total, cd_size, cd_off, clen) = struct.unpack("<IHHHHIIH", tail[i:i+22])
    zip64 = False
    j = tail.rfind(b"PK\x06\x07", 0, i)
    if j >= 0:
        sig, disk64, eocd64_off, ndisks = struct.unpack("<IIQI", tail[j:j+20])
        # read zip64 EOCD record
        rec = rng(url, eocd64_off, eocd64_off + 56 - 1)
        (sig, recsz, vm, vn, dnum, dcd, n_disk, n_total, cd_size, cd_off) = struct.unpack("<IQHHIIQQQQ", rec[:56])
        zip64 = True
    cd = rng(url, cd_off, cd_off + cd_size - 1)
    p = 0
    members = []
    while p < len(cd) and cd[p:p+4] == b"PK\x01\x02":
        (sig, vmade, vneed, flag, meth, mtime, mdate, crc, csize, usize, nlen, xlen, cmlen, dstart, iattr, eattr, loff) = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p+46])
        name = cd[p+46:p+46+nlen].decode("utf-8", "replace")
        extra = cd[p+46+nlen:p+46+nlen+xlen]
        # parse zip64 extra
        q = 0
        while q + 4 <= len(extra):
            hid, hsz = struct.unpack("<HH", extra[q:q+4])
            data = extra[q+4:q+4+hsz]
            if hid == 0x0001:
                k = 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack("<Q", data[k:k+8])[0]; k += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack("<Q", data[k:k+8])[0]; k += 8
                if loff == 0xFFFFFFFF:
                    loff = struct.unpack("<Q", data[k:k+8])[0]; k += 8
            q += 4 + hsz
        yr = ((mdate >> 9) & 0x7f) + 1980; mo = (mdate >> 5) & 0xf; dy = mdate & 0x1f
        hh = (mtime >> 11) & 0x1f; mi = (mtime >> 5) & 0x3f; ss = (mtime & 0x1f) * 2
        members.append({"name": name, "csize": csize, "usize": usize, "method": meth,
                        "local_header_offset": loff, "crc32": f"{crc:08x}",
                        "dos_mtime": f"{yr:04d}-{mo:02d}-{dy:02d} {hh:02d}:{mi:02d}:{ss:02d}"})
        p += 46 + nlen + xlen + cmlen
    res = {"url": url, "zip_size": size, "last_modified": hdrs.get("Last-Modified"), "zip64": zip64,
           "n_entries_eocd": n_total, "cd_size": cd_size, "cd_offset": cd_off,
           "n_members_parsed": len(members), "members": members}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(url, "size", size, "entries", n_total, "parsed", len(members), "cd_size", cd_size)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
