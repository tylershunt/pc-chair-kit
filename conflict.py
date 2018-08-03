from base import Person, Institutions
import re


def parse_line(line):
    """
    Line formats supported:
    name [-;, ] [(]institution[)] [-;, ] reason (promptly ignored)
    """
    original_line = line
    blacklist = ['institution', 'advisor', 'other', 'students', 'coauthors', 'conflict',
                 'collaborators', 'coi', 'paper', 'institutional', 'student', 'co-authors']

    institution_indication = ['university', 'institute', 'college']
    
    line = line.strip('"')
    name_regex = "\s*\w+\-?'?\w*\.?\&?\s*"

    g = re.match("\s*\d*\s*\-?\.?(.*)", line)
    line = g.group(1)

    g = re.match(name_regex, line, re.IGNORECASE)
    names = ""
    # Match all names
    while g:
        s, e = g.span()
        name = line[s:e].strip()
        if name.lower() in blacklist:
            raise ValueError("Bad name in string: %s, line: %s" % (name, original_line))
        names += name + " "
        line = line[e:]
        g = re.match(name_regex, line)

    if names.strip().lower() == 'none':
        return None, None

    if re.match("\s*[,;:\-\(]", line, re.IGNORECASE):
        # This is a name, make sure there is no warning:
        #for i in institution_indication:
        #    if i.lower() in names.lower():
        #        raise ValueError("Conflict looks like university: %s, line: %s" % (name, original_line))
        return names.strip(), None
    else:
        return None, names.strip()

    raise ValueError("Can't parse line: %s" % original_line)


class ConflictSet(object):
    def __init__(self, value=None):
        self._d = []
        self._reasons = []
        if value:
            self._d.append(value)
            self._reasons.append("")

    def __contains__(self, item):
        for i in self._d:
            if i.match(item):
                return True

        return False

    def match(self, item):
        for i in self._d:
            if i.match(item):
                return "%s" % (str(item))

        return False

    def intersects_with(self, other):
        intersection = ConflictSet()
        intersection_str = []
        for i in self._d:
            if i in other:
                intersection.add(i)
                intersection_str.append(other.match(i))

        return zip(intersection, intersection_str)

    def merge(self, other):
        for c, r in zip(other._d, other._reasons):
            self.add(c, r)

    def add(self, item, reason=""):
        if not item:
            return

        if item not in self:
            self._d.append(item)
            self._reasons.append(reason)

    def __iter__(self):
        for d in self._d:
            yield d

    def __str__(self):
        if not self._d:
            return ""

        s = ""
        for i,r in zip(self._d, self._reasons):
            if r:
                s += ("- %s; %s\n" % (str(i), r))
            else:    
                s += ("- %s\n" % str(i))

        return s

    def str_no_linebreaks(self):
        if not self._d:
            return ""

        s = ""
        for i, r in zip(self._d, self._reasons):
            # if r:
            #    s += ("(%s: %s);" % (str(i), r))
            # else:
            s += ("(%s);" % str(i))

        return s

    def __bool__(self):
        return True if self._d else False

class BaseConflicts(object):
    def __init__(self, institution, collaborators_str=""):
        """ Parse collaborators string """
        self.collabs = ConflictSet()
        self.institutions = ConflictSet()
        self.insts = institution
        self.bad_data = False
        lines = collaborators_str.strip().splitlines()
        for line in lines:
            name, institution = parse_line(line)
            if institution:
                self.add_institution(institution)
            elif name:
                self.add_co_author(name)

    def __iter__(self):
        for d in self.collabs:
            yield d
                
    def add_institution(self, inst):
        if isinstance(inst, BaseConflicts):
            raise ValueError("Trying to add a conflict list to a conflict list")
        if isinstance(inst, ConflictSet):
            raise ValueError("Trying to add a conflict list to a conflict list")
        
        self.institutions.add(self.insts.get_inst(inst))

    def add_co_author(self, a):
        if isinstance(a, ConflictSet):
            raise ValueError("Trying to add a conflict list to a conflict list")
        if isinstance(a, BaseConflicts):
            raise ValueError("Trying to add a conflict list to a conflict list")
        if isinstance(a, Person):
            p = a
        else:
            p = Person(a)
        self.collabs.add(p)

    def __inst_conflicts(self, other):
        return self.institutions.intersects_with(other.institutions)

    def __collab_conflicts(self, other):
        return self.collabs.intersects_with(other.collabs)

    def find_institution_conflicts(self, other):
        out = BaseConflicts(self.insts)
        for i in self.__inst_conflicts(other):
            inst, r = i
            out.institutions.add(inst, r)
        return out

    def find_collab_conflicts(self, other):
        out = BaseConflicts(self.insts)
        for i in self.__collab_conflicts(other):
            (c, r) = i
            out.collabs.add(c, r)
        return out
    
    def find_conflicts(self, other):
        collabs = self.find_collab_conflicts(other)
        insts = self.find_institution_conflicts(other)

        collabs.merge_inst_conflicts(insts)
        return collabs

    def merge_inst_conflicts(self, other):
        self.institutions.merge(other.institutions)

    def merge_collab_conflicts(self, other):
        self.collabs.merge(other.collabs)
    
    def match_institution(self, inst):
        return self.institutions.match(inst)

    def match_co_author(self, co_author):
        return self.collabs.match(co_author)

    def compare_co_authors(self, other):
        """
        Returns:
        (in_this_but_not_other, in_other_but_not_this)
        """
        in_this = []
        in_other = []

        for coa in other.collabs:
            if coa not in self.collabs:
                in_other.append(coa)

        for coa in self.collabs:
            if coa not in other.collabs:
                in_this.append(coa)

        return in_this, in_other

    def __bool__(self):
        return True if self.collabs or self.institutions else False

    def __str__(self):
        s= ""
        if self.institutions:
            s+= "Institutions:\n"
            s+= str(self.institutions)
        if self.collabs:
            s+= "Collaborators:\n"
            s+= str(self.collabs)
        return s

    def str_no_linebreaks(self):
        s = ""
        if self.institutions:
            s += "instituitons:[%s];" % self.institutions.str_no_linebreaks()
        if self.collabs:
            s += "collaborators:[%s]" % self.collabs.str_no_linebreaks()

        return s
        
