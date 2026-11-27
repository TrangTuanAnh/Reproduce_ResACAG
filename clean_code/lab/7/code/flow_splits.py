"""Chia dữ liệu lab 7 (xem ../PROTOCOL.md): RAND, GRP (theo IP nguồn), TFC (thời gian trong từng lớp)."""
import numpy as np


def rand_split(y, seed, test_frac=0.3):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    return np.sort(tr), np.sort(te)


def tfc_split(y, t_us, train_frac=0.7):
    """Mỗi lớp: 70% flow đầu (theo start_us) train, 30% cuối test."""
    tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        idx = idx[np.argsort(t_us[idx], kind="stable")]
        k = max(1, min(len(idx) - 1, int(round(train_frac * len(idx))))) if len(idx) > 1 else len(idx)
        tr.append(idx[:k]); te.append(idx[k:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(te))


def grp_split(y, src, seed, test_frac=0.3, initial_sides=None):
    """Chia theo nhóm IP nguồn: một IP chỉ ở một phía cho mọi lớp. Xử lý lớp ít IP nhất trước.
    Lớp có >=2 IP: >=1 IP mỗi phía nếu có thể. Trả về (train_idx, test_idx, report)."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    ips_of = {c: np.unique(src[y == c]) for c in classes}
    side = dict(initial_sides or {})  # ip -> 0 train / 1 test
    for c in sorted(classes, key=lambda c: len(ips_of[c])):
        ips = list(ips_of[c])
        todo = [ip for ip in ips if ip not in side]
        rng.shuffle(todo)
        n = len(ips)
        have_te = sum(side[ip] == 1 for ip in ips if ip in side)
        have_tr = sum(side[ip] == 0 for ip in ips if ip in side)
        want_te = max(1 if n >= 2 else 0, int(round(test_frac * n)))
        want_te = min(want_te, n - 1) if n >= 2 else 0
        for ip in todo:
            if have_te == 0 and have_tr >= 1 and ip == todo[-1] and n >= 2:
                side[ip] = 1; have_te += 1; continue
            if have_tr == 0 and have_te >= 1 and ip == todo[-1] and n >= 2:
                side[ip] = 0; have_tr += 1; continue
            if have_te < want_te:
                side[ip] = 1; have_te += 1
            else:
                side[ip] = 0; have_tr += 1
    s = np.array([side[ip] for ip in src])
    tr, te = np.where(s == 0)[0], np.where(s == 1)[0]
    report = {}
    for c in classes:
        m = y == c
        report[int(c)] = dict(ips=len(ips_of[c]), train_flows=int((m & (s == 0)).sum()),
                              test_flows=int((m & (s == 1)).sum()),
                              supported=bool((m & (s == 0)).any() and (m & (s == 1)).any()))
    return tr, te, report
