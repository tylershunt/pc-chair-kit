#!/usr/bin/env python3
# Explore the PC partition space
import click
import csv
import unidecode
import itertools
import sys
import time
from tqdm import tqdm
from warnings import warn
from math import ceil, factorial as fac

def csv_to_dicts(fileobj, schema=None, sanitize=lambda x:x):
    result = []
    reader = csv.DictReader(fileobj, schema)
    return [sanitize(d) for d in csv.DictReader(fileobj, schema)]

def sanitize_score_input(d):
    d["email"] = d.get("email").lower()
    d["paper"] = int(d.get("paper"))
    d["score"] = float(d.get("score"))
    d["preference"] = float(d.get("preference"))
    d["topic_score"] = float(d.get("topic_score"))
    d["citations"] = float(d.get("topic_score"))
    return d

def build_affinities_dict(affinity_report, emails):
    scores = csv_to_dicts(affinity_report, sanitize=sanitize_score_input)
    for e in emails:
        pc_members = {e["email"] for e in scores if e["email"] in emails}
        papers     = {e["paper"] for e in scores}
        affinities = {(e["paper"], e["email"]):e["score"] for e in scores}

    return pc_members,papers,affinities

def iter_partitions(pc, seed):
    print("there are", len(pc), "in the pc")
    pc = {a for a in pc if a not in seed[0] and a not in seed[1]}
    print("there are", len(pc), "assignable members")
    n = int(ceil(len(pc) / 2))
    remaining = (fac(n * 2)) // ((fac(n)**2) * 2)

    print("trying", remaining, "possible partitions")

    combos = itertools.combinations(pc, n)
    for combo in combos:
        if remaining <= 0:
            break
        remaining -= 1
        yield remaining, (list(combo) + seed[0], list(pc - set(combo)) +
                          seed[1])

def score_group(g, paper, affinities):
    result = 0
    for mem in g:
        result += affinities.get((paper,mem), 0)

    return result

def score_partition(part, papers, affinities):
    # get the score for each paper in each partition, add the max
    result = 0
    for p in papers:
        result += max(score_group(part[0], p, affinities),
                      score_group(part[1], p, affinities))
    return result

def build_seed_part(fileobj):
    if not fileobj:
        return [[], []]
    d = csv_to_dicts(fileobj)
    partA = [e["email"].lower() for e in d if e["part"].lower() == 'a']
    partB = [e["email"].lower() for e in d if e["part"].lower() == 'b']
    return partA, partB

#pid,pc-email,affinity-score
@click.command()
@click.option("--seed-partition", type=click.File('r'))
@click.argument("pc-names", type=click.File('r'))
@click.argument("affinity-report", type=click.File('r'))
def build_assignment(affinity_report, pc_names, seed_partition):
    seed_part = build_seed_part(seed_partition)
    valid_emails = [entry["email"].lower() for entry in  csv_to_dicts(pc_names)]
    pc_members,papers,affinities = build_affinities_dict(affinity_report,
                                                         valid_emails)

    partitions = iter_partitions(valid_emails, seed_part)
    remaining, best_part = next(partitions)
    best_score = score_partition(best_part, papers, affinities)

    i = 0
    start = time.time()
    try:
        with tqdm(total=remaining) as pbar:
            for _,part in partitions:
                pbar.update(1)
                i += 1
                score = score_partition(part, papers, affinities)
                if score > best_score:
                    best_score = score
                    best_part = part
    except KeyboardInterrupt:
        print("interrupted!!!! After", i, "ierations")
    end = time.time()

    print("took", end - start, "seconds")
    print("best score is:", best_score, "produced by this split:\n", best_part)


if __name__ == '__main__':
    build_assignment()
