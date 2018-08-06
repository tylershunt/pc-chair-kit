#!/usr/bin/env python3
# Explore the PC partition space
import click
import csv
import unidecode
import itertools
import sys
import time
from multiprocessing import Pool
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

def build_affinities_dict(affinity_report, emails, seed_part):
    scores = csv_to_dicts(affinity_report, sanitize=sanitize_score_input)
    for e in emails:
        pc_members = {e["email"] for e in scores if e["email"] in emails}
        papers     = {e["paper"] for e in scores}
        affinities = {(e["paper"], e["email"]):e["score"] for e in scores}

    #pre compute these affinities since they never change
    for p in papers:
        affinities[(p, "part0")] = score_group(seed_part[0], p, affinities)
        affinities[(p, "part1")] = score_group(seed_part[1], p, affinities)

    return pc_members,papers,affinities

def iter_partitions(pc, seed):
    print("there are", len(pc), "in the pc")
    pc = {a for a in pc if a not in seed[0] and a not in seed[1]}
    print("there are", len(pc), "assignable members")
    k = int(ceil(len(pc) / 2))
    remaining = (fac(k * 2)) // ((fac(k)**2))

    print("trying", remaining, "possible partitions")

    combos = itertools.combinations(pc, k)
    for combo in combos:
        if remaining <= 0:
            break
        remaining -= 1
        yield remaining, (list(combo), list(pc - set(combo)))

def score_group(g, paper, affinities):
    result = 0
    for mem in g:
        result += affinities.get((paper,mem), 0)

    return result

def score_partition(part, papers, affinities):
    # get the score for each paper in each partition, add the max
    result = 0

    for p in papers:
        result += max(score_group(part[0], p, affinities) + \
                        affinities[(p, "part0")],
                      score_group(part[1], p, affinities) + \
                        affinities[(p, "part1")])
    return result

def build_seed_part(fileobj):
    if not fileobj:
        return [[], []]
    d = csv_to_dicts(fileobj)
    partA = [e["email"].lower() for e in d if e["part"].lower() == 'a']
    partB = [e["email"].lower() for e in d if e["part"].lower() == 'b']
    return partA, partB

class PartitionProcessor:
    def __init__(self, papers, affinities):
        self.papers = papers
        self.affinities = affinities
    def __call__(self, part):
        return score_partition(part[1], self.papers, self.affinities), part[1]

#pid,pc-email,affinity-score
@click.command()
@click.option("--seed-partition", type=click.File('r'))
@click.option("--full-report", type=click.File('w'))
@click.option("-j", type=int)
@click.argument("pc-names", type=click.File('r'))
@click.argument("affinity-report", type=click.File('r'))
def build_assignment(affinity_report, pc_names, seed_partition, full_report, j):
    seed_part = build_seed_part(seed_partition)
    valid_emails = [entry["email"].lower() for entry in  csv_to_dicts(pc_names)]
    pc_members,papers,affinities = build_affinities_dict(affinity_report,
                                                         valid_emails,
                                                         seed_part)

    partitions = iter_partitions(valid_emails, seed_part)
    remaining, best_part = next(partitions)
    best_score = score_partition(best_part, papers, affinities)
    all_parts = [(best_score, best_part)]

    i = 0
    start = time.time()
    pool = Pool(j)

    process_partition = PartitionProcessor(papers, affinities)
    process_it = pool.imap_unordered(process_partition, partitions, 1000)
    try:
        with tqdm(total=remaining) as pbar:
            for score,part in process_it:
                part[0].extend(seed_part[0])
                part[1].extend(seed_part[1])
                pbar.update(1)
                i += 1
                all_parts.append((score, part))
                if score > best_score:
                    best_score = score
                    best_part = part
    except KeyboardInterrupt:
        print("interrupted!!!! After", i, "ierations")
    end = time.time()

    print("took", end - start, "seconds")
    print("best score is:", best_score, "produced by this split:\n", best_part)
    if full_report:
        writer = csv.writer(full_report)
        writer.writerow(["score","partition"])
        for s,p in all_parts:
            writer.writerow([s, p])


if __name__ == '__main__':
    build_assignment()
