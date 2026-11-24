"""Test lab 7: đặc trưng topology (causal/transductive/rewire) và các kiểu chia dữ liệu."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "code"))
import topology_features as F
import flow_splits as S
import numpy as np
import pytest

W = 300_000_000


def _make_flows(n, seed=0, t0=0.0):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, 5, n)
    dst = rng.integers(5, 10, n)
    dport = rng.integers(0, 2, n) * 363 + 80  # {80, 443}
    t = np.sort(rng.uniform(t0, t0 + 10 * n, n))
    return src, dst, dport, t


def _slow_cols(src, dst, dport, t, window=W):
    """Tính chậm cột 0 (n_src), 3 (n_dst), 5 (n_pair) theo định nghĩa causal."""
    n = len(src)
    out = np.zeros((n, 3))
    for i in range(n):
        past = (t < t[i]) & (t >= t[i] - window)
        out[i, 0] = np.sum(past & (src == src[i]))
        out[i, 1] = np.sum(past & (dst == dst[i]))
        out[i, 2] = np.sum(past & (src == src[i]) & (dst == dst[i]))
    return out


def test_topo_causal_no_lookahead():
    src, dst, dport, t = _make_flows(200, seed=42)
    out = F.topo_causal(src, dst, dport, t, window_us=W)
    slow = _slow_cols(src, dst, dport, t)
    np.testing.assert_array_equal(out[:, 0], slow[:, 0])
    np.testing.assert_array_equal(out[:, 3], slow[:, 1])
    np.testing.assert_array_equal(out[:, 5], slow[:, 2])


def test_topo_causal_invariant_to_future():
    src, dst, dport, t = _make_flows(100, seed=1)
    before = F.topo_causal(src, dst, dport, t, window_us=W)
    rng = np.random.default_rng(2)
    src2 = np.concatenate([src, rng.integers(0, 5, 100)])
    dst2 = np.concatenate([dst, rng.integers(5, 10, 100)])
    dport2 = np.concatenate([dport, rng.integers(0, 2, 100) * 363 + 80])
    t2 = np.concatenate([t, rng.uniform(t.max() + 1e9, t.max() + 5e9, 100)])
    after = F.topo_causal(src2, dst2, dport2, t2, window_us=W)
    np.testing.assert_array_equal(before, after[:100])


def test_topo_causal_ties_not_counted():
    src = np.array([7, 7])
    dst = np.array([9, 9])
    dport = np.array([80, 80])
    t = np.array([5.0, 5.0])
    out = F.topo_causal(src, dst, dport, t, window_us=W)
    assert out[0, 0] == 0 and out[1, 0] == 0


def test_topo_transductive_counts_all_others():
    src, dst, dport, t = _make_flows(80, seed=3)
    out = F.topo_transductive(src, dst, dport)
    for i in range(len(src)):
        assert out[i, 0] == np.sum((np.arange(len(src)) != i) & (src == src[i]))


def test_rewire_preserves_multiset_and_deterministic():
    src, dst, dport, t = _make_flows(120, seed=4)
    s1, d1 = F.rewire(src, dst, seed=10)
    s2, d2 = F.rewire(src, dst, seed=10)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(d1, d2)
    np.testing.assert_array_equal(np.sort(s1), np.sort(src))
    np.testing.assert_array_equal(np.sort(d1), np.sort(dst))


def test_rand_split_partitions_and_classes():
    rng = np.random.default_rng(5)
    y = np.repeat(np.array([0, 1, 2]), [25, 30, 25])
    y = y[rng.permutation(len(y))]
    tr, te = S.rand_split(y, seed=6)
    assert len(np.intersect1d(tr, te)) == 0
    assert len(np.union1d(tr, te)) == len(y)
    for c in (0, 1, 2):
        assert (y[tr] == c).sum() > 0
        assert (y[te] == c).sum() > 0


def test_tfc_split_time_order_per_class():
    rng = np.random.default_rng(7)
    n = 60
    y = rng.integers(0, 3, n)
    t = rng.uniform(0, 1e9, n)
    tr, te = S.tfc_split(y, t)
    for c in np.unique(y):
        tr_c = t[tr[y[tr] == c]]
        te_c = t[te[y[te] == c]]
        assert len(tr_c) > 0 and len(te_c) > 0
        assert tr_c.max() <= te_c.min()


def _make_grp_data(seed=8):
    rng = np.random.default_rng(seed)
    ips = np.arange(100, 137)  # 37 IP riêng biệt
    ip_pool = {0: ips[:30], 1: ips[30:35], 2: ips[35:37]}
    counts = {0: 60, 1: 40, 2: 20}
    y = np.concatenate([np.repeat(c, counts[c]) for c in (0, 1, 2)])
    perm = rng.permutation(len(y))
    y = y[perm]
    src = np.array([int(rng.choice(ip_pool[c])) for c in y])
    return y, src


def test_grp_split_disjoint_ips_and_report():
    y, src = _make_grp_data()
    tr, te, report = S.grp_split(y, src, seed=9)
    assert len(np.intersect1d(tr, te)) == 0
    assert len(np.union1d(tr, te)) == len(y)
    ip_tr, ip_te = set(np.unique(src[tr])), set(np.unique(src[te]))
    assert ip_tr.isdisjoint(ip_te)
    for c in (0, 1, 2):
        assert report[int(c)]["supported"] is True
        assert (y[tr] == c).sum() == report[int(c)]["train_flows"]
        assert (y[te] == c).sum() == report[int(c)]["test_flows"]
