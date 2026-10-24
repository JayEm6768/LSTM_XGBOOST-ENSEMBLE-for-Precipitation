import sys, time, nbformat
from nbclient import NotebookClient

path = sys.argv[1] if len(sys.argv) > 1 else "LSTM_XGB_ACTIVATION_OPTIMIZER_EXP.ipynb"
nb = nbformat.read(path, as_version=4)
client = NotebookClient(
    nb,
    timeout=9000,          # per-cell: LSTM ~25 min, dart search ~40 min
    kernel_name="python3",
    resources={"metadata": {"path": "."}},
    allow_errors=False,
)
t0 = time.time()
try:
    client.execute()
    status = "SUCCESS"
except Exception as e:
    status = "FAILED: %s: %s" % (type(e).__name__, str(e)[:600])
finally:
    nbformat.write(nb, path)

elapsed = time.time() - t0
n_out = sum(len(c.get("outputs", [])) for c in nb.cells if c.cell_type == "code")
n_err = sum(1 for c in nb.cells if c.cell_type == "code"
            for o in c.get("outputs", []) if o.get("output_type") == "error")
print("=" * 70)
print("STATUS   : %s" % status)
print("ELAPSED  : %.1fs (%.1f min)" % (elapsed, elapsed / 60))
print("OUTPUTS  : %d stored, %d error outputs" % (n_out, n_err))
print("SAVED    : %s" % path)
