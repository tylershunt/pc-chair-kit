# -*- coding: utf-8 -*-
from tqdm import tqdm
from util import iterate_csv, get_dict_json, save_dict_json
from dblp_crawler import (request_publication, sanitize_titles,
                          sanitize_coauthors, save_cache)
from base import Person, Institutions
from conflict import BaseConflicts
from copy import deepcopy
from functools import reduce


class Publication(object):
    def __init__(self, key, title, authors, year, institutions, valid=True):
        self.valid = valid
        self.key = key
        self.title = title
        self.insts = institutions
        self.authors = BaseConflicts(institutions)
        for author in authors:
            name, a_key = author
            self.authors.add_co_author(Person(name, a_key, key))
        self.year = year

    @classmethod
    def from_key(cls, key, institutions):
        xml, rdf = request_publication(key)
        if xml:
            return cls.from_xml(xml, rdf, institutions)
        else:
            return None

    @classmethod
    def invalid_pub(cls, title, key, year, institutions):
        return cls(key, title, [], year, institutions, False)

    @classmethod
    def from_xml(cls, xml, rdf, institutions):
        pub_type = list(xml['dblp'].keys())[0]
        key = xml['dblp'][pub_type]['@key']
        year = int(xml['dblp'][pub_type]['year'])
        title = sanitize_titles(xml['dblp'][pub_type]['title'])

        if 'dblp:editedBy' in rdf['rdf:RDF']['dblp:Publication'][1]:
            return cls.invalid_pub(title, key, year, institutions)

        if rdf:
            # Scrap authors from rdf file
            try:
                authors_list = rdf['rdf:RDF']['dblp:Publication'][1]['dblp:authoredBy']
                if isinstance(authors_list, list):
                    author_keys = [i['@rdf:resource'][21:]
                                   for i in authors_list
                                   if i['@rdf:resource'].startswith('http://dblp.org/pers/')]
                else:
                    author_keys = (authors_list['@rdf:resource'][21:]
                                   if authors_list['@rdf:resource'].startswith('http://dblp.org/pers/')
                                   else [])
            except Exception as ex:
                print("Bad rdf:")
                print(rdf['rdf:RDF']['dblp:Publication'][1])
                raise ex

        if 'author' in xml['dblp'][pub_type]:
            authors = sanitize_coauthors(xml['dblp'][pub_type]['author'])
        else:
            authors = []

        if len(author_keys) == len(authors):
            authors = zip(authors, author_keys)
        else:
            authors = [(a, None) for a in authors]

        return cls(key,
                   title,
                   authors,
                   year,
                   institutions)

    def get_authors(self):
        return self.authors

    def check_against_pc_member(self, pc_member):
        c = self.authors.find_collab_conflicts(pc_member.conflicts)
        return c if c else None

    def __contains__(self, author):
        for a in self.authors:
            if a.match(author):
                return True
        return False

    def __str__(self):
        return "Title: %s, key:%s" % (self.title, self.key)


class Submission(Publication):
    def __init__(self, pid, title, authors, affiliations,
                 institutions, pc_conflicts, collaborators):

        as_ = [(a, "") for a in authors]
        super().__init__("", title, as_, 2017, institutions, valid=True)
        self.pid = pid
        self.conflicts = BaseConflicts(institutions, collaborators)
        for a in affiliations:
            self.conflicts.add_institution(a)
        self.pc_member_collabs_field_cs = {}
        self.collabs_field_cs = {}
        self.dblp_cs = {}
        self.fake_conflicts = {}
        self.declared_pc = [email for email, _ in pc_conflicts]
        self.reviewers = []

    def __str__(self):
        return "%d " % self.pid + super().__str__()

    @classmethod
    def from_json(cls, json_dic, institutions):
        authors = []
        for a in json_dic['authors']:
            if 'first' in a and 'last' in a:
                authors.append("%s %s" % (a['first'], a['last']))
            elif 'last' in a:
                authors.append(a['last'])
            else:
                raise ValueError("Paper %d: author list is broken")

        affiliations = [a['affiliation']
                        for a in json_dic['authors']
                        if 'affiliation' in a]

        if 'pc_conflicts' in json_dic:
            pc_conflicts = json_dic['pc_conflicts'].items()
        else:
            pc_conflicts = {}

        collaborators = (json_dic['collaborators']
                         if 'collaborators' in json_dic else '')

        return cls(json_dic['pid'], json_dic['title'], authors, affiliations,
                   institutions, pc_conflicts, collaborators)

    def add_collaborator_conflict(self, pc_member):
        if pc_member.email in self.declared_pc:
            return

        # Check if pc_member is in paper conflict list
        match = self.conflicts.match_co_author(pc_member.name)
        if match:
            self.collabs_field_cs[pc_member.email] = match

    def get_conflicts_from_pc_member(self, pc_member,
                                     ignore_other_fields=True):
        c = BaseConflicts(self.insts, "")
        if (ignore_other_fields and ((pc_member.email in self.declared_pc) or
                                     (pc_member.email in self.collabs_field_cs))):
            return c

        # Check if pc_member institutions match authors institutions
        c.merge_inst_conflicts(self.conflicts.find_institution_conflicts(pc_member.conflicts))

        # Check if any author match something in the pc_member conflicts
        c.merge_collab_conflicts(self.authors.find_collab_conflicts(pc_member.conflicts))

        return c

    def add_conflicts_from_pc_member(self, pc_member):
        c = self.get_conflicts_from_pc_member(pc_member)
        if not c:
            return

        self.pc_member_collabs_field_cs[pc_member.email] = c

    def add_conflicts_from_dblp(self, pc_member):
        if pc_member.email in self.pc_member_collabs_field_cs:
            return

        c = self.get_conflicts_from_pc_member(pc_member)
        if not c:
            return

        self.dblp_cs[pc_member.email] = c

    def add_fake_conflicts(self, pc_member_hotcrp, pc_member_dblp):
        """
        A fake conflict is one where authors flag a script
        that cannot be verified from authors conflicts or
        from DBLP
        """
        golden_hotcrp = self.get_conflicts_from_pc_member(pc_member_hotcrp,
                                                          False)
        golden_dblp = self.get_conflicts_from_pc_member(pc_member_dblp, False)

        claimed_1 = self.declared_pc
        claimed_2 = self.collabs_field_cs

        email = pc_member_hotcrp.email
        if ((email in claimed_1 or
             email in claimed_2) and
           not (golden_hotcrp or golden_dblp)):

            if email in claimed_1:
                self.fake_conflicts[email] = "pc_conflicts"
            elif email in claimed_2:
                self.fake_conflicts[email] = "collaborators"
            else:
                raise ValueError("What is going on pal?")

    def list_conflicts(self, conflict_type):
        if conflict_type == 'proper':
            return {e: "" for e in self.declared_pc}
        elif conflict_type == 'collaborators_field':
            return self.collabs_field_cs
        elif conflict_type == 'declared_by_pc_members':
            return self.pc_member_collabs_field_cs
        elif conflict_type == 'dblp':
            return self.dblp_cs
        elif conflict_type == 'fake_conflicts':
            return self.fake_conflicts
        else:
            raise ValueError("don't know how to geneate this list")

    def conflicts_csv(self, conflict_type):
        l = self.list_conflicts(conflict_type)
        s = []
        for email, reasons in l.items():
            if isinstance(reasons, BaseConflicts):
                str_reasons = reasons.str_no_linebreaks()
            else:
                str_reasons = reasons
            s.append("x, %d, %s, %s" % (self.pid, email, str_reasons))
        return s


class PCMember(Person):
    def __init__(self, first, last, email, tags, affiliation,
                 topics, institutions, collaborators="", key=""):
        name = "%s %s" % (first, last)
        super().__init__(name, key)
        self.first = first
        self.last = last
        self.email = email
        self.tags = tags
        self.insts = institutions
        self.affiliation = institutions.get_inst(affiliation)
        self.conflicts = BaseConflicts(institutions, collaborators)
        self.conflicts.add_institution(affiliation)
        self.publications = []
        self.topics = topics

    @classmethod
    def from_hotcrp_csv(cls, line, insts):
        (first, last, email, roles, tags, affiliation,
         collaborators, follow, *topics) = line
        return cls(first, last, email, tags, affiliation,
                   topics, insts, collaborators)

    @classmethod
    def simple_pcmember(cls, first, last, key, insts):
        return cls(first, last, "", "", "", [], insts, key=key)

    def getPrettyName(self):
        return super().getTheirNames()
        #return last

    def copy_no_conflicts(self):
        return PCMember(self.first, self.last, self.email,
                        self.tags, str(self.affiliation),
                        self.topics, self.insts, key=self.key)

    def copy_full(self):
        return deepcopy(self)

    def add_co_author(self, a):
        self.conflicts.add_co_author(a)

    def add_publication(self, pub):
        self.publications.append(pub)
        print(pub)
        pub_as = pub.get_authors()
        self.conflicts.merge_collab_conflicts(pub_as)

    def conflicts_with_person(self, person):
        return self.conflicts.match_co_author(person)

    def conflicts_with_institution(self, person):
        return self.conflicts.match_institution(person)

    def check_conflict_set(self, other):
        """
        Returns:
        (in_this_but_not_other, in_other_but_not_this)
        """
        return self.conflicts.compare_co_authors(other.conflicts)
