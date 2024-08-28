
def test_masking(a, xs, ys):
    a = a.todense()
    for x, y in zip(xs, ys):
        assert a[x, y] == 0, "Test masking failed!"


def test_is_edge(adj, out_nodes, in_nodes, is_edge):
    adj = adj.todense()
    for k, p, j in zip(out_nodes, in_nodes, is_edge):
        if adj[k, p] != j:
            return False, "is is_edge is not calculated correctly"

