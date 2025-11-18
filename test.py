# test.py (robust version)
import joblib
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score

CANDIDATE_TEST_FILES = [
    "test.pkl",
    "test_data.pkl",
    "data.pkl",
    "model_data.pkl",
    "test.npz",
    "X_test.npy",
    "y_test.npy",
    "X_test.pkl",
    "y_test.pkl"
]

def try_load_npz(path):
    try:
        arrs = np.load(path, allow_pickle=True)
        # If .npz with named arrays:
        if hasattr(arrs, "files") and "X_test" in arrs.files and "y_test" in arrs.files:
            return arrs["X_test"], arrs["y_test"]
        # if single array maybe contains tuple
        if isinstance(arrs, np.ndarray) and arrs.size == 2:
            return arrs[0], arrs[1]
    except Exception:
        pass
    return None

def find_test_data():
    # 1) look for a combined test file
    for f in CANDIDATE_TEST_FILES:
        if os.path.exists(f):
            # npy / npz
            if f.endswith(".npz") or f.endswith(".npy"):
                loaded = try_load_npz(f)
                if loaded:
                    return loaded
                # if single .npy maybe it's X only; try to pair with y file
                if f == "X_test.npy" and os.path.exists("y_test.npy"):
                    return np.load("X_test.npy", allow_pickle=True), np.load("y_test.npy", allow_pickle=True)
            # try joblib/pickle
            try:
                obj = joblib.load(f)
                if isinstance(obj, dict):
                    if "X_test" in obj and "y_test" in obj:
                        return obj["X_test"], obj["y_test"]
                    # sometimes keys are ('X','y') or ('x_test','y_test')
                    for kx in ("X_test","X","x_test"):
                        for ky in ("y_test","Y","y"):
                            if kx in obj and ky in obj:
                                return obj[kx], obj[ky]
                # if obj is tuple/list of length 2
                if isinstance(obj, (list, tuple)) and len(obj) == 2:
                    return obj[0], obj[1]
            except Exception:
                # not a joblib file or failed to load — continue
                pass
    # 2) try to load X_test.npy and y_test.npy together if both exist
    if os.path.exists("X_test.npy") and os.path.exists("y_test.npy"):
        return np.load("X_test.npy", allow_pickle=True), np.load("y_test.npy", allow_pickle=True)
    return None

def main():
    # load model (model.pkl expected)
    if not os.path.exists("model.pkl"):
        print("ERROR: 'model.pkl' not found in the project root. Run train.py first or provide model file.")
        sys.exit(1)

    obj = joblib.load("model.pkl")

    # case A: model.pkl contains a dict with model and test data
    if isinstance(obj, dict):
        model = obj.get("model") or obj.get("estimator") or None
        X_test = obj.get("X_test") or obj.get("X") or None
        y_test = obj.get("y_test") or obj.get("y") or None
        if model is None:
            # maybe the dict IS the model (unlikely) — try common keys
            for v in obj.values():
                if hasattr(v, "predict"):
                    model = v
                    break
    else:
        # case B: model.pkl is the estimator itself
        model = obj
        # try to find test data in other files
        found = find_test_data()
        if found:
            X_test, y_test = found
        else:
            X_test = y_test = None

    # Final checks
    if model is None or not hasattr(model, "predict"):
        print("ERROR: Could not find a valid model in 'model.pkl'.")
        sys.exit(1)

    if X_test is None or y_test is None:
        print("\nERROR: Could not find test data automatically.\n")
        print("Files in repository:")
        for f in sorted(os.listdir(".")):
            print(" ", f)
        print("\nWhat to do next (pick one):")
        print("1) Re-run train.py and save model + test data together, e.g.:")
        print("   joblib.dump({'model': model, 'X_test': X_test, 'y_test': y_test}, 'model.pkl')")
        print("2) Save test arrays as X_test.npy and y_test.npy or test.pkl (dict with X_test/y_test) in project root.")
        print("3) If you already have test files, move them to the project root or rename them to one of:")
        print("   ", ", ".join(CANDIDATE_TEST_FILES))
        sys.exit(1)

    # compute predictions and accuracy (works for classification/regression where accuracy is appropriate)
    try:
        preds = model.predict(X_test)
    except Exception as e:
        print("ERROR: model.predict failed:", e)
        sys.exit(1)

    # For classification use accuracy_score; if regression, you may want other metrics.
    try:
        acc = accuracy_score(y_test, preds)
        print("TEST ACCURACY:", acc)
    except Exception:
        # fallback: print basic info
        print("Prediction results (first 10):", preds[:10])
        print("True labels (first 10):", y_test[:10])
        print("Could not compute accuracy (shapes/types may differ).")

if __name__ == "__main__":
    main()
