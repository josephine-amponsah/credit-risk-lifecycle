"""Cut the 5.6 MB test-user file down to a small, representative demo sample.

Usage: python scripts/make_sample_users.py ../test_user_ids.json api/sample_users.json --n 50
"""
import argparse
import json
import random

parser = argparse.ArgumentParser()
parser.add_argument("src")
parser.add_argument("dst")
parser.add_argument("--n", type=int, default=50)
args = parser.parse_args()

users = json.load(open(args.src))
random.Random(42).shuffle(users)
json.dump(users[: args.n], open(args.dst, "w"), indent=1)
print(f"Wrote {min(args.n, len(users))} of {len(users)} users to {args.dst}")