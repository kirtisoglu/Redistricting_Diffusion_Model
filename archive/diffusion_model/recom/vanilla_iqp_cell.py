"""Missing 2x2 cell: Vanilla ReCom + IQP (lam=0 disables tempering)."""
import random, sys
from pathlib import Path
import numpy as np
_REPO = Path(__file__).resolve().parent.parent.parent
_DIFF = _REPO / "diffusion_model"
if str(_DIFF) not in sys.path:
    sys.path.insert(0, str(_DIFF))
from recom.problems import setup_problem
from recom.chains import run_chain, make_iqp_hybrid_step
from recom.qp_model import make_partition, kernel_uniform

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
p = setup_problem("iowa", _REPO)
graph, eps, ideal, init = p["graph"], p["eps"], p["ideal"], p["init"]
random.seed(SEED * 1000 + 3); np.random.seed(SEED * 1000 + 3)
res = run_chain("V+IQP", make_partition(graph, init),
                make_iqp_hybrid_step(graph, ideal, eps, 0.0, 10, 10.0, 1.0,
                                     kernel_uniform, False, 5.0),
                500, None, graph=graph, ideal=ideal)
c = res["cuts"]
print(f"V+IQP: best={c.min()} mean={c[100:].mean():.2f} std={c[100:].std():.2f} "
      f"maxdev={res['pop_dev'].max():.4f} wall={res['times'].sum():.1f}s")
np.savez(Path(__file__).parent / "iowa_hybrid_test_out" /
         f"vanilla_plus_iqp_seed{SEED}.npz", cuts=c, lam=0.0, seed=SEED)
