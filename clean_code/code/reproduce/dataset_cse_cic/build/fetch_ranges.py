"""Parallel HTTP range downloader with resume. Usage: fetch_ranges.py URL OUT [conns]"""
import os, sys, threading, queue, time
import requests

CHUNK = 8 * 1024 * 1024


def size_of(url):
    r = requests.head(url, timeout=60, allow_redirects=True)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def main(url, out, conns=8):
    total = size_of(url)
    part = out + ".part"
    done_dir = out + ".chunks"
    os.makedirs(done_dir, exist_ok=True)
    if not os.path.exists(part) or os.path.getsize(part) != total:
        with open(part, "wb") as f:
            f.truncate(total)
    q = queue.Queue()
    n = (total + CHUNK - 1) // CHUNK
    for i in range(n):
        if not os.path.exists(os.path.join(done_dir, str(i))):
            q.put(i)
    lock = threading.Lock()
    errors = []

    def worker():
        s = requests.Session()
        while True:
            try:
                i = q.get_nowait()
            except queue.Empty:
                return
            a, b = i * CHUNK, min(total, (i + 1) * CHUNK) - 1
            for attempt in range(8):
                try:
                    r = s.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
                    r.raise_for_status()
                    data = r.content
                    if len(data) != b - a + 1:
                        raise IOError("short read")
                    with lock:
                        with open(part, "r+b") as f:
                            f.seek(a)
                            f.write(data)
                    open(os.path.join(done_dir, str(i)), "w").close()
                    break
                except Exception as e:
                    time.sleep(2 + 3 * attempt)
            else:
                errors.append(i)

    t0 = time.time()
    threads = [threading.Thread(target=worker) for _ in range(conns)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        print("FAILED chunks", errors, flush=True)
        sys.exit(1)
    os.replace(part, out)
    for name in os.listdir(done_dir):
        os.remove(os.path.join(done_dir, name))
    os.rmdir(done_dir)
    print(f"done {out} {total/1e6:.1f}MB in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8)
