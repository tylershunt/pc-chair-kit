#!/usr/bin/env python3
# Explore the PC partition space
import click
import csv
import unidecode
import itertools
import sys
import time
import heapq
from multiprocessing import Pool
from tqdm import tqdm
from warnings import warn
from math import ceil, factorial as fac

def to_int(x):
    if x:
        return int(x)
    return 0

def csv_to_dicts(fileobj, schema=None, sanitize=lambda x:x):
    result = []
    reader = csv.DictReader(fileobj, schema)
    return [sanitize(d) for d in csv.DictReader(fileobj, schema)]

def sanitize_members(d):
    d["email"] = d.get("email").lower()
    return d

def sanitize_score_input(d):
    d["email"] = d.get("email").lower()
    d["paper"] = int(d.get("paper"))
    d["score"] = float(d.get("score"))
    d["preference"] = float(d.get("preference"))
    d["topic_score"] = float(d.get("topic_score"))
    d["citations"] = float(d.get("citations"))
    return d

def sanitize_seed_input(d):
    d["email"] = d.get("email").lower()
    d["part"] = d.get("part").lower()
    return d

def build_affinities_dict(affinity_report, valid_members):
    scores = csv_to_dicts(affinity_report, sanitize=sanitize_score_input)

    pc_members = set(e["email"] for e in valid_members)
    papers     = {e["paper"] for e in scores}
    affinities = {(e["paper"], e["email"]):e["score"] for e in scores}

    return pc_members,papers,affinities

def iter_partitions(pc, seed, n):
    k = int(ceil(len(pc) / 2))
    for combo in itertools.combinations(pc, k):
        yield Partition(Group(combo), Group(pc - set(combo)), seed, top_n=n)

def filter_out_seed(pc, seed):
    print("there are", len(pc), "in the pc")
    pc = {a for a in pc if a not in seed}
    print("there are", len(pc), "assignable members")
    k = int(ceil(len(pc) / 2))
    remaining = (fac(k * 2)) // ((fac(k)**2))
    print("trying", remaining, "possible partitions")
    return remaining, pc

class Group:
    def __init__(self, members):
        self.members = list(members)

    def __contains__(self, key):
        if key in self.members:
            return True

    def __iter__(self):
        return iter(self.members)

    def iter_scores(self, paper, affs):
        return (affs.get((paper,mem), 0) for mem in self)

    def top_n_scores(self, paper, affs, n):
        return heapq.nlargest(n, self.iter_scores(paper, affs))

    def score(self, paper, affs):
        return sum(self.iter_scores(paper, affs))

class MemoizedGroup(Group):
    def __init__(self, members, top_n=None):
        super(MemoizedGroup, self).__init__(members)
        self.scores = {}
        self.top_n_scores_store = {}

    def top_n_scores(self, paper, affs, n):
        if not (paper, n) in self.top_n_scores_store:
            self.top_n_scores_store[(paper,n)] = \
                    super(MemoizedGroup,self).top_n_scores(paper, affs, n)
        return self.top_n_scores_store[(paper,n)]

    def score(self, paper, affs):
        if not paper in self.scores:
            self.scores[paper] = super(MemoizedGroup,self).score(paper, affs)
        return self.scores[paper]

class MultiGroup:
    def __init__(self, *args, top_n=None):
        self.groups = list(args)
        self.papers = []
        self.top_n = top_n

    def assign(self, p):
        self.papers.append(p)

    def __iter__(self):
        return itertools.chain(*self.groups)

    def __str__(self):
        return str(list(self))

    def score(self, paper, affs):
        if self.top_n:
            it = (g.top_n_scores(paper, affs, self.top_n) for g in self.groups)
            it = heapq.nlargest(self.top_n, itertools.chain(*it))
        else:
            it = (g.score(paper, affs) for g in self.groups)
        return sum(it)

class Partition:
    def __init__(self, groupA, groupB, seed, top_n=None, save_papers=False):
        self.save_papers = save_papers
        if seed:
            self.groupA = MultiGroup(groupA, seed.groupA.groups[0], top_n=top_n)
            self.groupB = MultiGroup(groupB, seed.groupB.groups[0], top_n=top_n)
        else:
            self.groupA = MultiGroup(groupA, top_n=top_n)
            self.groupB = MultiGroup(groupB, top_n=top_n)

    def __contains__(self, key):
        return key in self.groupA or key in self.groupB

    def __str__(self):
        return "=====Group A======\n" + str(self.groupA) + \
               "\n=====Group B======\n" + str(self.groupB)
    def score(self, papers, affs):
        # get the score for each paper in each partition, add the max
        result = 0

        for p in papers:
            sA = self.groupA.score(p, affs)
            sB = self.groupB.score(p, affs)
            if self.save_papers:
                if sA > sB:
                    self.groupB.assign(p)
                else:
                    self.groupA.assign(p)
            result += max(sA, sB)
        return result

def build_seed_part(fileobj, save_papers=False, top_n=None):
    if fileobj:
        d = csv_to_dicts(fileobj, sanitize=sanitize_seed_input)
        groupA = MemoizedGroup(e["email"] for e in d if e["part"] == 'a')
        groupB = MemoizedGroup(e["email"] for e in d if e["part"] == 'b')
    else:
        groupA = MemoizedGroup([])
        groupB = MemoizedGroup([])

    return Partition(groupA, groupB, None, top_n=top_n, save_papers=save_papers)

# need this instead of closure for multiprocess
class PartitionProcessor:
    def __init__(self, papers, affinities):
        self.papers = papers
        self.affinities = affinities
    def __call__(self, part):
        try:
            return part.score(self.papers, self.affinities), part
        except KeyboardInterrupt:
            return None

@click.group()
@click.argument("pc-names", type=click.File('r'))
@click.argument("affinity-report", type=click.File('r'))
@click.pass_context
def cli(ctx, pc_names, affinity_report):
    valid_emails = csv_to_dicts(pc_names, sanitize=sanitize_members)
    pc_members, papers, affinities = build_affinities_dict(affinity_report, valid_emails)
    ctx.obj["pc_members"] = pc_members
    ctx.obj["papers"] = papers
    ctx.obj["affinities"] = affinities

#pid,pc-email,affinity-score
@click.command()
@click.option("--seed-partition", type=click.File('r'),
              help="csv file defining and initial fixed partition")
@click.option("-n", type=int,
              help="if given only consider the best n reviewers")
@click.option("--full-report", type=click.File('w'),
              help="dump every partition and score into this file")
@click.option("-j", type=int, help="number of worker processes")
@click.pass_context
def search(ctx, seed_partition, n, full_report, j):
    seed_part = build_seed_part(seed_partition)
    pc_members = ctx.obj["pc_members"]
    papers = ctx.obj["papers"]
    affinities = ctx.obj["affinities"]

    seed_part.score(papers, affinities)

    remaining, filtered_emails = filter_out_seed(pc_members, seed_part)
    partitions = iter_partitions(filtered_emails, seed_part, n)
    best_part = next(partitions)
    best_score = best_part.score(papers, affinities)
    all_parts = [(best_score, best_part)]

    if full_report:
        writer = csv.writer(full_report)
        writer.writerow(["score","partition"])

    i = 0
    start = time.time()
    pool = Pool(j)

    process_partition = PartitionProcessor(papers, affinities)
    process_it = pool.imap_unordered(process_partition, partitions, 1000)
    try:
        with tqdm(total=remaining) as pbar:
            for score,part in process_it:
                pbar.update(1)
                if score > best_score:
                    best_score = score
                    best_part = part
                if full_report:
                    writer.writerow((score, part))
    except KeyboardInterrupt:
        print("interrupted!!!! After", i, "ierations")

    end = time.time()

    print("took", end - start, "seconds")
    print("best score is:", best_score, "produced by this split:\n", best_part)

@click.command()
@click.option("-n", type=int,
              help="if given only consider the best n reviewers")
@click.argument("part_csv", type=click.File('r'))
@click.pass_context
def examine(ctx, part_csv, n):
    papers = ctx.obj["papers"]
    affinities = ctx.obj["affinities"]
    part = build_seed_part(part_csv, True, top_n=n)

    score = part.score(papers, affinities)

    print("Partition:\n", part)
    print("SCORE is", score)
    print("Count of papers:")
    print("GroupA:", len(part.groupA.papers), "GroupB:",
          len(part.groupB.papers))

@click.command()
@click.option("-n", type=int,
              help="if given only consider the best n reviewers")
@click.pass_context
def total(ctx, n):
    papers = ctx.obj["papers"]
    affinities = ctx.obj["affinities"]
    pc_members = ctx.obj["pc_members"]
    group = Group(pc_members)

    if not n:
        score = sum(group.score(p, affinities) for p in papers)
    else:
        score = sum(sum(group.top_n_scores(p, affinities, n)) for p in papers)

    print(len(pc_members), "reviewers, and", len(papers), "papers")
    print("Total affinity is", score)

cli.add_command(search)
cli.add_command(examine)
cli.add_command(total)
if __name__ == '__main__':
    cli(obj={})
