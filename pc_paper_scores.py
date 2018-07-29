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
    preferences = {(int(entry["paper"]), entry["email"].lower()):to_int(entry["preference"]) \
                    for entry in report_entries}
    topic_scores = {(int(entry["paper"]), entry["email"].lower()):to_int(entry["topic_score"]) \
                    for entry in report_entries}
    return emails,pids,preferences,topic_scores

def calculate_aggregate_score(preference, topic_score, citations):
    # Placeholder for final score calculation, need to unify these different metrics
    return preference + topic_score + citations

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
    args = parser.parse_args()
    scores_csv = args.out_scores
    emails,pids,preferences,topic_scores = get_ids_emails_and_scores(
                                                        args.preference_report)

    submissionList = get_dict_json(args.paper_json)

    allFiles = listdir(args.submissions)
    justcsvs = list(filter(lambda x: x.endswith(".csv"), allFiles))
    refCounts = list(map(lambda x: join(args.submissions, x), justcsvs))

    citationsList = {}
    for ref in refCounts:
        get_citation_count(ref, citationsList)
    citations = get_citations_by_pid_email(pids, emails, citationsList)
    # Print csv
    schema = "paper,email,score,preference,topic_score,citations\n"
    with open(scores_csv, 'w') as f:
        f.write(schema)
        for pid in pids:
            for email in emails:
                preference = preferences.get((pid,email),0)
                topic_score = topic_scores.get((pid,email),0)
                citation_score = citations.get((pid,email), 0)
                score = calculate_aggregate_score(preference, topic_score, citation_score)
                s = "%d,%s,%d,%d,%d,%d\n" % (pid, email, score, preference, topic_score, citation_score)
                f.write(s)
    print("Done calculating scores for all {paper,pc} tuple")

if __name__ == '__main__':
    main()
