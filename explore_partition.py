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

def iterate_csv(fileobj, encoding=""):
    csv_it = csv.reader(fileobj)
    for r in csv_it:
        yield [unidecode.unidecode(i) for i in r]

def csv_to_dicts(fileobj, schema=None):
    result = []
    row_gen = iterate_csv(fileobj, schema)
    first_row = next(row_gen, None)
    if not schema:
        schema = first_row
    else:
        if schema != first_row:
            warn("csv schema does not match header in file")
    return [{k:v for k, v in zip(schema, row)} for row in row_gen]

def to_int(string):
    if string:
        return int(string)
    return 0

def build_affinities_dict(affinity_report, valid_emails):
    affinity_entries = csv_to_dicts(affinity_report)
    pc_members = {entry["email"].lower() for entry in affinity_entries \
                  if entry["email"].lower() in valid_emails}
    papers = {int(entry["paper"]) for entry in affinity_entries}
    affinities = {(int(entry["paper"]), entry["email"].lower()):to_int(entry["topic_score"]) \
                    for entry in affinity_entries}
    return pc_members,papers,affinities

def iter_partitions(pc, seed):
    print("there are", len(pc), "in the pc")
    pc = {a for a in pc if a not in seed[0] and a not in seed[1]}
    i = 0
    n = int(ceil(len(pc) / 2))
    stop_after = (fac(n * 2)) // ((fac(n)**2) * 2)

    print("trying", stop_after, "possible partitions")

    combos = itertools.combinations(pc, n)
    for combo in combos:
        if i == stop_after:
            break
        i += 1
        yield list(combo) + seed[0], list(pc - set(combo)) + seed[1]

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
    best_part = next(partitions)
    best_score = score_partition(best_part, papers, affinities)

    start = time.time()
    for part in partitions:
        score = score_partition(part, papers, affinities)
        if score > best_score:
            best_score = score
            best_part = part
    end = time.time()

    print("took", end - start, "seconds")
    print("best score is:", best_score, "produced by this split:\n", best_part)


if __name__ == '__main__':
    build_assignment()
