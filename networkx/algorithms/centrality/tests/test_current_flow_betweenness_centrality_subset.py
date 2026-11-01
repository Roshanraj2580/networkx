import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

import networkx as nx
from networkx import edge_current_flow_betweenness_centrality as edge_current_flow
from networkx import (
    edge_current_flow_betweenness_centrality_subset as edge_current_flow_subset,
)


class TestFlowBetweennessCentrality:
    def test_K4_normalized(self):
        """Betweenness centrality: K4"""
        G = nx.complete_graph(4)
        b = nx.current_flow_betweenness_centrality_subset(
            G, list(G), list(G), normalized=True
        )
        b_answer = nx.current_flow_betweenness_centrality(G, normalized=True)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)

    def test_K4(self):
        """Betweenness centrality: K4"""
        G = nx.complete_graph(4)
        b = nx.current_flow_betweenness_centrality_subset(
            G, list(G), list(G), normalized=True
        )
        b_answer = nx.current_flow_betweenness_centrality(G, normalized=True)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)
        # test weighted network
        G.add_edge(0, 1, weight=0.5, other=0.3)
        b = nx.current_flow_betweenness_centrality_subset(
            G, list(G), list(G), normalized=True, weight=None
        )
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)
        b = nx.current_flow_betweenness_centrality_subset(
            G, list(G), list(G), normalized=True
        )
        b_answer = nx.current_flow_betweenness_centrality(G, normalized=True)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)
        b = nx.current_flow_betweenness_centrality_subset(
            G, list(G), list(G), normalized=True, weight="other"
        )
        b_answer = nx.current_flow_betweenness_centrality(
            G, normalized=True, weight="other"
        )
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)

    def test_P4_normalized(self):
        """Betweenness centrality: P4 normalized"""
        G = nx.path_graph(4)
        b = nx.current_flow_betweenness_centrality_subset(
            G, list(G), list(G), normalized=True
        )
        b_answer = nx.current_flow_betweenness_centrality(G, normalized=True)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)

    def test_P4(self):
        """Betweenness centrality: P4"""
        G = nx.path_graph(4)
        b = nx.current_flow_betweenness_centrality_subset(
            G, list(G), list(G), normalized=True
        )
        b_answer = nx.current_flow_betweenness_centrality(G, normalized=True)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)

    def test_star(self):
        """Betweenness centrality: star"""
        G = nx.Graph()
        nx.add_star(G, ["a", "b", "c", "d"])
        b = nx.current_flow_betweenness_centrality_subset(
            G, list(G), list(G), normalized=True
        )
        b_answer = nx.current_flow_betweenness_centrality(G, normalized=True)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)

    def test_small_graph(self):
        G1 = nx.complete_graph(1)
        b1_norm = nx.current_flow_betweenness_centrality_subset(
            G1, list(G1), list(G1), normalized=True
        )
        b1_unnorm = nx.current_flow_betweenness_centrality_subset(
            G1, list(G1), list(G1), normalized=False
        )
        assert b1_norm == {0: 0.0}
        assert b1_unnorm == {0: 0.0}

        G2 = nx.complete_graph(2)
        b2_norm = nx.current_flow_betweenness_centrality_subset(
            G2, list(G2), list(G2), normalized=True
        )
        b2_unnorm = nx.current_flow_betweenness_centrality_subset(
            G2, list(G2), list(G2), normalized=False
        )
        assert b2_norm == {0: 0.0, 1: 0.0}
        assert b2_unnorm == {0: 0.0, 1: 0.0}


# class TestWeightedFlowBetweennessCentrality():
#     pass


class TestEdgeFlowBetweennessCentrality:
    def test_K4_normalized(self):
        """Betweenness centrality: K4"""
        G = nx.complete_graph(4)
        b = edge_current_flow_subset(G, list(G), list(G), normalized=True)
        b_answer = edge_current_flow(G, normalized=True)
        for (s, t), v1 in b_answer.items():
            v2 = b.get((s, t), b.get((t, s)))
            assert v1 == pytest.approx(v2, abs=1e-7)

    def test_K4(self):
        """Betweenness centrality: K4"""
        G = nx.complete_graph(4)
        b = edge_current_flow_subset(G, list(G), list(G), normalized=False)
        b_answer = edge_current_flow(G, normalized=False)
        for (s, t), v1 in b_answer.items():
            v2 = b.get((s, t), b.get((t, s)))
            assert v1 == pytest.approx(v2, abs=1e-7)
        # test weighted network
        G.add_edge(0, 1, weight=0.5, other=0.3)
        b = edge_current_flow_subset(G, list(G), list(G), normalized=False, weight=None)
        # weight is None => same as unweighted network
        for (s, t), v1 in b_answer.items():
            v2 = b.get((s, t), b.get((t, s)))
            assert v1 == pytest.approx(v2, abs=1e-7)

        b = edge_current_flow_subset(G, list(G), list(G), normalized=False)
        b_answer = edge_current_flow(G, normalized=False)
        for (s, t), v1 in b_answer.items():
            v2 = b.get((s, t), b.get((t, s)))
            assert v1 == pytest.approx(v2, abs=1e-7)

        b = edge_current_flow_subset(
            G, list(G), list(G), normalized=False, weight="other"
        )
        b_answer = edge_current_flow(G, normalized=False, weight="other")
        for (s, t), v1 in b_answer.items():
            v2 = b.get((s, t), b.get((t, s)))
            assert v1 == pytest.approx(v2, abs=1e-7)

    def test_C4(self):
        """Edge betweenness centrality: C4"""
        G = nx.cycle_graph(4)
        b = edge_current_flow_subset(G, list(G), list(G), normalized=True)
        b_answer = edge_current_flow(G, normalized=True)
        for (s, t), v1 in b_answer.items():
            v2 = b.get((s, t), b.get((t, s)))
            assert v1 == pytest.approx(v2, abs=1e-7)

    def test_P4(self):
        """Edge betweenness centrality: P4"""
        G = nx.path_graph(4)
        b = edge_current_flow_subset(G, list(G), list(G), normalized=True)
        b_answer = edge_current_flow(G, normalized=True)
        for (s, t), v1 in b_answer.items():
            v2 = b.get((s, t), b.get((t, s)))
            assert v1 == pytest.approx(v2, abs=1e-7)

    def test_small_graph(self):
        G1 = nx.complete_graph(1)
        assert edge_current_flow_subset(G1, list(G1), list(G1), normalized=True) == {}
        assert edge_current_flow_subset(G1, list(G1), list(G1), normalized=False) == {}

        G2 = nx.complete_graph(2)
        res_norm = edge_current_flow_subset(G2, list(G2), list(G2), normalized=True)
        res_unnorm = edge_current_flow_subset(G2, list(G2), list(G2), normalized=False)
        assert res_norm.get((0, 1), res_norm.get((1, 0))) == 1.0
        assert res_unnorm.get((0, 1), res_unnorm.get((1, 0))) == 0.5
