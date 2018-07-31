"""
    This script is used to calculate an affinity score between an reviewer and
    a paper. This script is based off of paper_affinity.py.
    The affinity score depends on 3 parameters
        1) Reviewer preference scores
        2) Topic score based on paper topics and reviewer expertise
        3) Number of citations a paper makes to the reviewer
"""
#!/usr/bin/env python3

import argparse
import csv
import unidecode
import re
from util import (get_dict_json)
from explore_partition import (to_int)
from paper_affinity import (get_citation_count)
from os import listdir
from os.path import join
from subprocess import Popen, PIPE
import statistics

DEF_CITATIONS_WEIGHT = 1.0
DEF_TOPIC_SCORE_WEIGHT = 1.0
DEF_PREFERENCE_WEIGHT = 2.0

def iterate_csv(filename, encoding=""):
    if not encoding:
        encoding = 'latin-1'
    with open(filename, newline='', encoding=encoding) as csvfile:
        csv_it = csv.reader(csvfile)
        for r in csv_it:
            yield [unidecode.unidecode(i) for i in r]

def csv_to_dicts(fileobj, schema=None):
    result = []
    row_gen = iterate_csv(fileobj)
    first_row = next(row_gen, None)
    if not schema:
        schema = first_row
    else:
        if schema != first_row:
            warn("csv schema does not match header in file")
    return [{k:v for k, v in zip(schema, row)} for row in row_gen]

def get_citations_by_pid_email(pids, emails, citationsList):
    citations = {}
    for pid in pids:
        pc_citations = (citationsList[pid]
                        if pid in citationsList else {})
        for email in emails:
            if email in pc_citations:
                citations[(pid,email)] = pc_citations[email]
    return citations

def get_ids_emails_and_scores(preference_report):
    report_entries = csv_to_dicts(preference_report)
    emails = {entry["email"].lower() for entry in report_entries}
    pids = {to_int(entry["paper"]) for entry in report_entries}
    prefs = {(int(entry["paper"]), entry["email"].lower()):to_int(entry["preference"]) \
                    for entry in report_entries}
    topics = {(int(entry["paper"]), entry["email"].lower()):to_int(entry["topic_score"]) \
                    for entry in report_entries}
    return emails,pids,prefs,topics

def get_standardized_score(curr_score, mean, stdev):
    # Bootstrap to 0 if stdev = 1
    if stdev == 0:
        return 0
    return (curr_score - mean)/stdev

def standardize(pids, emails, dict):
    standardized_scores = {}
    for email in emails:
        scores = [dict.get((pid,email), 0) for pid in pids]
        mean_score = statistics.mean(scores)
        stdev_score = statistics.stdev(scores)
        for pid in pids:
            key = (pid,email)
            standardized_score = get_standardized_score(dict.get(key, 0),
                                                        mean_score,
                                                        stdev_score)
            standardized_scores[key] = standardized_score
    return standardized_scores

def calculate_aggregate_score(pref_weight, pref_score, topic_weight, topic_score, citation_weight, citation_score):
    pref = pref_weight * pref_score
    topics = topic_weight * topic_score
    cites = citation_weight * citation_score
    score = pref + topics + cites
    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preference_report",
                        help='Reviewer preference report downloaded from HotCRP')
    parser.add_argument("--paper-json",
                        help='json with submitted papers and topics')
    parser.add_argument("--submissions",
                        help="Folder containing the submissions")
    parser.add_argument("--out_scores",
                        help="Scores report")
    parser.add_argument("--citations_weight",
                        default=DEF_CITATIONS_WEIGHT,
                        help="Citation score weight in final score, def: {}".format(DEF_CITATIONS_WEIGHT))
    parser.add_argument("--topics_weight",
                        default=DEF_TOPIC_SCORE_WEIGHT,
                        help="Topic score weight in final score, def: {}".format(DEF_TOPIC_SCORE_WEIGHT))
    parser.add_argument("--preference_weight",
                        default=DEF_PREFERENCE_WEIGHT,
                        help="Reviewer preference weight in final score, def: {}".format(DEF_PREFERENCE_WEIGHT))

    args = parser.parse_args()
    citations_weight = float(args.citations_weight)
    topics_weight = float(args.topics_weight)
    pref_weight = float(args.preference_weight)
    scores_csv = args.out_scores

    emails,pids,prefs,topics = get_ids_emails_and_scores(args.preference_report)
    submissionList = get_dict_json(args.paper_json)

    allFiles = listdir(args.submissions)
    justcsvs = list(filter(lambda x: x.endswith(".csv"), allFiles))
    refCounts = list(map(lambda x: join(args.submissions, x), justcsvs))

    citationsList = {}
    for ref in refCounts:
        get_citation_count(ref, citationsList)
    citations = get_citations_by_pid_email(pids, emails, citationsList)

    # The standardization is performed on data local to a PC member, e.g.,
    # preference, topic expertise, citations. This approach avoids comparisons
    # between arbitrary preference scales of PC members and enables assigning
    # scores to {paper, pc} tuples independent of the corresponding metrics of
    # other PC members.
    std_prefs = standardize(pids, emails, prefs)
    std_topics = standardize(pids, emails, topics)
    std_citations = standardize(pids, emails, citations)

    # Print csv
    schema = "paper,email,score,preference,topic_score,citations\n"
    with open(scores_csv, 'w') as f:
        f.write(schema)
        for pid in pids:
            for email in emails:
                key = (pid,email)
                score = calculate_aggregate_score(pref_weight, std_prefs.get(key, 0),
                                                  topics_weight, std_topics.get(key, 0),
                                                  citations_weight, std_citations.get(key, 0))
                s = "%d,%s,%.2f,%d,%d,%d\n" % (pid, email, score, prefs.get(key, 0),
                                               topics.get(key, 0), citations.get(key, 0))
                f.write(s)
    print("Done calculating scores for all {paper,pc} tuple")

if __name__ == '__main__':
    main()