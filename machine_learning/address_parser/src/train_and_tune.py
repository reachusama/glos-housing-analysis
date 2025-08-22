# train_and_tune.py
# !/usr/bin/env python
import os
import argparse
import math
import pickle
import inspect

import numpy as np
import sklearn_crfsuite
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn_crfsuite import metrics as crf_metrics

import machine_learning.address_parser.src.tokens as tok
import machine_learning.address_parser.src.metrics as metric

MODEL_FILE = 'addressCRF.crfsuite'
MODEL_PATH = '../configs/model/training'


# ------------------------- IO & data -------------------------

def ensure_model_dir():
    os.makedirs(MODEL_PATH, exist_ok=True)
    return os.path.join(MODEL_PATH, MODEL_FILE)


def read_data(training_xml: str, holdout_xml: str, verbose: bool = True):
    if verbose:
        print(f"Reading training XML: {training_xml}")
    X_train, y_train = tok.readData(training_xml)
    if verbose:
        print(f"Training sequences: {len(X_train)}")

    if verbose:
        print(f"Reading holdout XML:  {holdout_xml}")
    X_test, y_test = tok.readData(holdout_xml)
    if verbose:
        print(f"Holdout sequences:   {len(X_test)}")

    return X_train, y_train, X_test, y_test


def maybe_subsample(X, Y, n: int | None, seed: int = 42):
    if n is None or n >= len(X): return X, Y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    Xs = [X[i] for i in idx]
    Ys = [Y[i] for i in idx]
    return Xs, Ys


# ------------------------- Model helpers -------------------------


def make_crf(
        algorithm: str = "lbfgs",
        c1: float = 0.3,
        c2: float = 0.001,
        min_freq: float = 0.001,
        all_possible_transitions: bool = True,
        max_iterations: int | None = None,
        epsilon: float | None = None,
        verbose: bool = True,
        random_state: int | None = 42,  # will be filtered out if unsupported
        model_file_path: str | None = None,
):
    """
    Build a CRF instance but only pass kwargs your installed sklearn-crfsuite supports.
    This avoids errors like: TypeError: CRF.__init__() got an unexpected keyword 'random_state'
    """
    if model_file_path is None:
        model_file_path = ensure_model_dir()

    # Base params (some may be dropped below if unsupported)
    params = dict(
        algorithm=algorithm,
        c1=c1,  # used by lbfgs
        c2=c2,  # used by lbfgs
        min_freq=min_freq,
        all_possible_transitions=all_possible_transitions,
        keep_tempfiles=True,
        model_filename=model_file_path,
        verbose=verbose,
        random_state=random_state,  # some versions support this, some don't
    )

    # AP-specific (safe to include; will be filtered if not supported)
    if algorithm == "ap":
        params.update(dict(
            max_iterations=max_iterations or 5000,
            epsilon=epsilon or 1e-4,
        ))

    # Drop None values
    params = {k: v for k, v in params.items() if v is not None}

    # Keep only kwargs that __init__ actually accepts in your version
    sig = inspect.signature(sklearn_crfsuite.CRF.__init__)
    supported = {k: v for k, v in params.items() if k in sig.parameters}

    return sklearn_crfsuite.CRF(**supported)


def evaluate(crf, X_test, y_test, label_order=None, heading="Holdout performance"):
    print(f"\n{heading}:")
    y_pred = crf.predict(X_test)

    # Weighted token-level F1 over observed labels
    labels = label_order or list(crf.classes_)
    f1 = crf_metrics.flat_f1_score(y_test, y_pred, average="weighted", labels=labels)

    # Sequence accuracy
    seq_acc = metric.sequence_accuracy_score(y_test, y_pred)

    print(f"Weighted F1:      {f1:.4f}")
    print(f"Sequence accuracy:{seq_acc:.4f}\n")

    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    print(crf_metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
    return dict(f1=f1, sequence_accuracy=seq_acc)


# ------------------------- Training -------------------------

def train(
        training_xml: str,
        holdout_xml: str,
        algorithm: str = "lbfgs",
        c1: float = 0.3,
        c2: float = 0.001,
        min_freq: float = 0.001,
        all_possible_transitions: bool = True,
        random_state: int = 42,
        verbose: bool = True,
        train_subset: int | None = None,
        eval_subset: int | None = None,
):
    model_path = ensure_model_dir()
    X_train, y_train, X_test, y_test = read_data(training_xml, holdout_xml, verbose=verbose)

    X_train, y_train = maybe_subsample(X_train, y_train, train_subset, seed=random_state)
    X_test, y_test = maybe_subsample(X_test, y_test, eval_subset, seed=random_state)

    crf = make_crf(
        algorithm=algorithm,
        c1=c1, c2=c2, min_freq=min_freq,
        all_possible_transitions=all_possible_transitions,
        random_state=random_state,
        model_file_path=model_path,
        verbose=verbose,
    )

    print("\nStart training...")
    crf.fit(X_train, y_train)
    print("Finished training.")
    print(f"Model saved to: {model_path}")
    if hasattr(crf, "training_log_"):
        try:
            print("Training last iteration:", crf.training_log_.last_iteration)
        except Exception:
            pass

    # Evaluate
    labels = list(crf.classes_)  # learnt from data
    return evaluate(crf, X_test, y_test, label_order=labels)


# ------------------------- Hyperparameter search -------------------------

def tune(
        training_xml: str,
        holdout_xml: str,
        n_iter: int = 50,
        cv: int = 3,
        random_state: int = 42,
        sequence_optimisation: bool = True,
        min_freq: float = 0.001,
        all_possible_transitions: bool = True,
        train_subset: int | None = None,
        eval_subset: int | None = None,
        plot_path: str | None = None,
        pickle_path: str | None = None,
):
    """
    Randomised search over c1/c2 (LBFGS). Saves the best model to tok.MODEL_PATH/tok.MODEL_FILE.
    """
    model_path = ensure_model_dir()
    X_train, y_train, X_test, y_test = read_data(training_xml, holdout_xml, verbose=True)

    X_train, y_train = maybe_subsample(X_train, y_train, train_subset, seed=random_state)
    X_test, y_test = maybe_subsample(X_test, y_test, eval_subset, seed=random_state)

    base = make_crf(
        algorithm="lbfgs",
        min_freq=min_freq,
        all_possible_transitions=all_possible_transitions,
        random_state=random_state,
        verbose=False,  # quieter during CV
        model_file_path=model_path,  # ensures the refit best model is written here
    )

    # Search space
    param_space = {
        "c1": stats.expon(scale=0.5),  # L1
        "c2": stats.expon(scale=0.05),  # L2
    }

    if sequence_optimisation:
        scorer = make_scorer(metric.sequence_accuracy_score)
    else:
        # Token-level weighted F1 over labels found in training folds
        # (sklearn-crfsuite handles this internally)
        scorer = make_scorer(crf_metrics.flat_f1_score, average="weighted")

    print("\nRandomised search (LBFGS) starting...")
    rs = RandomizedSearchCV(
        base,
        param_distributions=param_space,
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        scoring=scorer,
        refit=True,  # refit best on full training set
    )
    rs.fit(X_train, y_train)

    best = rs.best_estimator_
    print("\nBest params:", rs.best_params_)
    print("Best CV score:", rs.best_score_)
    try:
        print("Model size: {:.2f}M".format(best.size_ / 1_000_000))
    except Exception:
        pass

    # Save the RandomizedSearchCV object (optional)
    if pickle_path is None:
        pickle_path = os.path.join(tok.MODEL_PATH, "optimisation.pickle")
    try:
        with open(pickle_path, "wb") as fh:
            pickle.dump(rs, fh)
        print(f"Search object pickled to: {pickle_path}")
    except Exception as e:
        print(f"Could not pickle search object: {e}")

    # Holdout eval
    labels = list(best.classes_)
    scores = evaluate(best, X_test, y_test, label_order=labels, heading="Holdout performance (best model)")

    # Optional plot of the search surface
    if plot_path:
        try:
            import matplotlib.pyplot as plt
            cvr = rs.cv_results_
            xs = np.array(cvr["param_c1"], dtype=float)
            ys = np.array(cvr["param_c2"], dtype=float)
            cs = np.array(cvr["mean_test_score"], dtype=float)
            fig, ax = plt.subplots()
            ax.set_xscale("log");
            ax.set_yscale("log")
            s = ax.scatter(xs, ys, c=cs, s=60, edgecolors="k", alpha=0.75)
            ax.set_xlabel("c1");
            ax.set_ylabel("c2")
            ax.set_title("Randomised Search CV Results")
            fig.colorbar(s, ax=ax, label="mean_test_score")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close(fig)
            print(f"Saved plot: {plot_path}")
        except Exception as e:
            print(f"Plotting skipped: {e}")

    print(f"Best model written to: {model_path}")
    return scores, rs.best_params_


# ------------------------- CLI -------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(description="Train or tune a CRF address parser (single script).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Common args
    def add_common(sp):
        sp.add_argument("--train-xml", required=True, help="Path to training XML")
        sp.add_argument("--holdout-xml", required=True, help="Path to holdout XML")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--train-subset", type=int, default=None, help="Use only N training sequences (speed-up)")
        sp.add_argument("--eval-subset", type=int, default=None, help="Use only N holdout sequences (speed-up)")

    # train
    sp_tr = sub.add_parser("train", help="Train a CRF with fixed hyperparameters")
    add_common(sp_tr)
    sp_tr.add_argument("--algo", choices=["lbfgs", "ap"], default="lbfgs")
    sp_tr.add_argument("--c1", type=float, default=0.3)
    sp_tr.add_argument("--c2", type=float, default=0.001)
    sp_tr.add_argument("--min-freq", type=float, default=0.001)
    sp_tr.add_argument("--no-all-transitions", action="store_true", help="Disable all_possible_transitions")
    sp_tr.add_argument("--quiet", action="store_true")

    # tune
    sp_tu = sub.add_parser("tune", help="Randomised search over c1/c2 (LBFGS)")
    add_common(sp_tu)
    sp_tu.add_argument("--n-iter", type=int, default=50)
    sp_tu.add_argument("--cv", type=int, default=3)
    sp_tu.add_argument("--seq-opt", action="store_true", help="Optimise for sequence accuracy (default token F1)")
    sp_tu.add_argument("--min-freq", type=float, default=0.001)
    sp_tu.add_argument("--plot", default=None, help="Path to save a PNG of the search surface")
    sp_tu.add_argument("--pickle", default=None, help="Where to pickle the RandomizedSearchCV object")
    return p


def main():
    args = build_arg_parser().parse_args()

    if args.cmd == "train":
        train(
            training_xml=args.train_xml,
            holdout_xml=args.holdout_xml,
            algorithm=args.algo,
            c1=args.c1,
            c2=args.c2,
            min_freq=args.min_freq,
            all_possible_transitions=not args.no_all_transitions,
            random_state=args.seed,
            verbose=not args.quiet,
            train_subset=args.train_subset,
            eval_subset=args.eval_subset,
        )
    elif args.cmd == "tune":
        tune(
            training_xml=args.train_xml,
            holdout_xml=args.holdout_xml,
            n_iter=args.n_iter,
            cv=args.cv,
            random_state=args.seed,
            sequence_optimisation=args.seq_opt,
            min_freq=args.min_freq,
            all_possible_transitions=True,
            train_subset=args.train_subset,
            eval_subset=args.eval_subset,
            plot_path=args.plot,
            pickle_path=args.pickle,
        )


if __name__ == "__main__":
    main()
