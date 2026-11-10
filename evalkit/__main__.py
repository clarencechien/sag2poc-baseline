"""CLI.

  python -m evalkit score   --dataset musique --split dev --run run.jsonl [--out scores.jsonl]
  python -m evalkit agg     --scores reference/musique/sag2_jev.jsonl --dataset musique --split all
  python -m evalkit compare --a a.jsonl --b b.jsonl --dataset musique --split dev [--metrics R@5 AE@10]
"""
import argparse
import json

from . import aggregate, load_questions, load_run, load_scores, load_split, paired_bootstrap, score_run


def main():
    parser = argparse.ArgumentParser(prog="evalkit")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("score", "agg", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True)
        p.add_argument("--split", default="all")
        p.add_argument("--root", default="data")
        if name == "score":
            p.add_argument("--run", required=True)
            p.add_argument("--out")
        if name == "agg":
            p.add_argument("--scores", required=True)
        if name == "compare":
            p.add_argument("--a", required=True)
            p.add_argument("--b", required=True)
            p.add_argument("--metrics", nargs="+", default=["R@5", "R@10", "AE@10"])
            p.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    qids = load_split(args.dataset, args.split, args.root)
    if args.cmd == "score":
        scores = score_run(load_questions(args.dataset, args.root), load_run(args.run), qids)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as stream:
                for qid in qids:
                    stream.write(json.dumps({"qid": qid, **scores[qid]}) + "\n")
        print(json.dumps(aggregate(scores, qids), indent=1))
    elif args.cmd == "agg":
        print(json.dumps(aggregate(load_scores(args.scores), qids), indent=1))
    else:
        a, b = load_scores(args.a), load_scores(args.b)
        for metric in args.metrics:
            print(json.dumps(paired_bootstrap(a, b, qids, metric, args.resamples)))


if __name__ == "__main__":
    main()
