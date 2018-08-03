from dblp_crawler import Cache
from util import iterate_csv, get_dict_json
from tqdm import tqdm
from base import Institutions
from pc_members import Publication, Submission, PCMember
import sys


def print_conflict_list(l):
    s = ""
    for email, reasons in l.items():
        s += ("====> email: %s\n" % email)
        if reasons:
            s += str(reasons) + '\n'
    return s


def print_reports(sub_list, conflict_type, report_file):
    csv_list = []
    for s in tqdm(sub_list):
        csv_list += s.conflicts_csv(conflict_type)

    str_out = "valid, pid, email, reasons\n"
    str_out += "\n".join(csv_list)
    with open(report_file, 'w') as f:
        f.write(str_out)


def main():
    institutions_csv = sys.argv[1]
    submissions_json = sys.argv[2]
    hotcrp_pc_member_csv = sys.argv[3]
    pc_member_paper_db_csv = sys.argv[4]

    out_proper = sys.argv[5]
    out_paper_collabs_field = sys.argv[6]
    out_pc_collabs_field = sys.argv[7]
    out_dblp = sys.argv[8]
    out_fake = sys.argv[9]
    # Step 1: read all the inputs (institutions_csv, paper data from hotcrp,
    # pc info from hotcrp and paper db from dblp):
    print("Reading institutions csv...")
    institutions = Institutions(institutions_csv)

    print("Reading submissions:")
    d = get_dict_json(submissions_json)
    submissions = [Submission.from_json(p, institutions) for p in tqdm(d)]

    print("Reading hotcrp pc members:")
    hotcrp_pc_members = [PCMember.from_hotcrp_csv(line, institutions)
                         for line in tqdm(iterate_csv(hotcrp_pc_member_csv,
                                                      encoding='utf-8'))]
    hotcrp_pc_members = {p.email: p for p in hotcrp_pc_members}

    print("Reading pc papers:")
    dblp_pc_members = {k: p.copy_no_conflicts()
                       for k, p in hotcrp_pc_members.items()}

    for row in tqdm(iterate_csv(pc_member_paper_db_csv)):
        (email, id, firstname, lastname, keys, valid,
         pub_key, pub_title, pub_year, pub_authors) = row

        if valid == "x":
            pub = Publication.from_key(pub_key, institutions)
            dblp_pc_members[email].add_publication(pub)

    print("Cross referencing conflicts")
    for s in tqdm(submissions):
        # Step 2: list conflicts that are declared by authors properly
        # Step 3: list conflicts that are declared by authors
        #         in the collaborators field
        for k, v in hotcrp_pc_members.items():
            s.add_collaborator_conflict(v)

        # Step 4: list conflicts declared by pc members but undeclared
        # by paper authors
        for k, v in hotcrp_pc_members.items():
            s.add_conflicts_from_pc_member(v)

        # Step 5: list conflicts not declared by anyone but caught by dblp
        for k, v in dblp_pc_members.items():
            s.add_conflicts_from_dblp(v)

        # Step 6:
        for k, v in dblp_pc_members.items():
            s.add_fake_conflicts(hotcrp_pc_members[k], v)

    print_reports(submissions, 'proper', out_proper)
    print_reports(submissions, 'collaborators_field', out_paper_collabs_field)
    print_reports(submissions, 'declared_by_pc_members', out_pc_collabs_field)
    print_reports(submissions, 'dblp', out_dblp)
    print_reports(submissions, 'fake_conflicts', out_fake)



if __name__ == '__main__':
    main()

