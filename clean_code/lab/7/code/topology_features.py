"""Đặc trưng topology cho lab 7 (xem ../PROTOCOL.md). Không dùng nhãn ở bất kỳ đâu trong file này."""
from collections import Counter, defaultdict, deque

import numpy as np

TOPO_NAMES = ["n_src", "n_src_dst_ip", "n_src_dst_port", "n_dst", "n_dst_src_ip",
              "n_pair", "n_dst_dst_port", "n_src_dport"]


class _Counts:
    """Bộ đếm có thể thêm/bớt flow (s, d, p) với s,d là id IP, p là dst port."""

    def __init__(self):
        self.src = Counter(); self.dst = Counter(); self.pair = Counter(); self.src_dport = Counter()
        self.src_dst = defaultdict(Counter); self.src_port = defaultdict(Counter)
        self.dst_src = defaultdict(Counter); self.dst_port = defaultdict(Counter)

    def _upd(self, s, d, p, k):
        self.src[s] += k; self.dst[d] += k; self.pair[(s, d)] += k; self.src_dport[(s, p)] += k
        for table, key, sub in ((self.src_dst, s, d), (self.src_port, s, p),
                                (self.dst_src, d, s), (self.dst_port, d, p)):
            c = table[key]
            c[sub] += k
            if c[sub] <= 0:
                del c[sub]
                if not c:
                    del table[key]

    def add(self, s, d, p):
        self._upd(s, d, p, 1)

    def remove(self, s, d, p):
        self._upd(s, d, p, -1)

    def feats(self, s, d, p):
        return (self.src[s], len(self.src_dst.get(s, ())), len(self.src_port.get(s, ())),
                self.dst[d], len(self.dst_src.get(d, ())), self.pair[(s, d)],
                len(self.dst_port.get(d, ())), self.src_dport[(s, p)])


def topo_causal(src, dst, dport, t_us, window_us=300_000_000):
    """Với flow i: chỉ dùng flow j có t_j < t_i và t_j >= t_i - W (quá khứ thật sự, ties không tính).
    Không dùng nhãn. Trả về (n, 8) float64. src/dst/dport: mảng int; t_us: float (µs), không NaN."""
    n = len(src)
    order = np.argsort(t_us, kind="stable")
    out = np.zeros((n, len(TOPO_NAMES)))
    cnt = _Counts()
    win = deque()
    pos = 0
    while pos < n:
        t = t_us[order[pos]]
        end = pos
        while end < n and t_us[order[end]] == t:
            end += 1
        while win and t_us[win[0]] < t - window_us:
            j = win.popleft()
            cnt.remove(src[j], dst[j], dport[j])
        for k in range(pos, end):  # tính cho cả lô trước khi thêm lô vào cửa sổ
            i = order[k]
            out[i] = cnt.feats(src[i], dst[i], dport[i])
        for k in range(pos, end):
            i = order[k]
            cnt.add(src[i], dst[i], dport[i]); win.append(i)
        pos = end
    return out


def topo_transductive(src, dst, dport):
    """Cùng 8 đặc trưng nhưng đếm trên MỌI flow khác (cả tương lai), không giới hạn thời gian."""
    n = len(src)
    cnt = _Counts()
    for i in range(n):
        cnt.add(src[i], dst[i], dport[i])
    out = np.zeros((n, len(TOPO_NAMES)))
    for i in range(n):
        cnt.remove(src[i], dst[i], dport[i])
        out[i] = cnt.feats(src[i], dst[i], dport[i])
        cnt.add(src[i], dst[i], dport[i])
    return out


def rewire(src, dst, seed):
    """Hoán vị độc lập cột nguồn và cột đích (giữ phân phối bậc, phá cấu trúc)."""
    rng = np.random.default_rng(seed + 777)
    return rng.permutation(src), rng.permutation(dst)


def ip_octets(ip_series):
    """IP dạng chuỗi 'a.b.c.d' -> (n,4) float. Chuỗi không hợp lệ -> NaN."""
    import pandas as pd
    parts = ip_series.astype(str).str.split(".", expand=True).reindex(columns=range(4))
    return parts.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
