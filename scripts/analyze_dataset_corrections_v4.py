#!/usr/bin/env python3
"""Experimental source-guarded refinements; no reference input to rules.
Reuses v3 frozen inputs and fresh correction->verification four-way evaluator.
"""
import re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze_dataset_corrections_v3 as base
OLD_FIELDS=base.corrected_fields
OLD_SEMANTICS=base.semantic_proposals
OLD_PROPOSE=base.propose
PERSON_RELATIONS=base.PERSON_SUBJECT_RELATIONS-{'occupation'}
TITLE_STOP={'the','currently','formerly','president','professor','dr','mr','mrs','ms','sir','his','her','their','university','college','january','february','march','april','may','june','july','august','september','october','november','december'}

def unique_surname(value,relation,text):
    if relation not in PERSON_RELATIONS or not re.fullmatch(base.NAME_WORD,value) or len(value)<3:return None
    if re.search(base.KINSHIP,text,re.I):return None
    # The short name is the grammatical subject of a supported leading clause.
    lead=r'\s*(?:Currently\s*,?\s*)?'+re.escape(value)+r'\s+(?:was|is|graduated|attended|competed|received|won|joined|served|has|had)\b'
    if not re.match(lead,text,re.I):return None
    token=r'(?:'+base.NAME_WORD+r'|[A-Z]\.)'
    pattern=r'(?<!\w)'+token+r'(?:\s+'+token+r'){0,2}\s+'+re.escape(value)+r'(?!\w)'
    found={}
    for m in re.finditer(pattern,text):
        full=m.group();words=full.split()
        if any(w.lower().strip('.') in TITLE_STOP for w in words[:-1]):continue
        if full!=value:found[full]=(full,{'start':m.start(),'end':m.end(),'span':m.group()})
    return next(iter(found.values())) if len(found)==1 else None

def corrected_fields(original,text):
    t,edits=OLD_FIELDS(original,text)
    if base.FEATURES['entity_representation']:
        found=unique_surname(t[0],t[1],text)
        if found and found[0]!=t[0]:
            old=list(t);t[0]=found[0]
            edits.append({'rule':'unique_source_surname_v4','field':0,'old':old,'new':list(t),'evidence':found[1]})
    return t,edits

def evidence_span(m,text,template):
    return {'start':m.start(),'end':m.end(),'span':text[m.start():m.end()],'template':template}

def direct_match(s,o,middle,text,template):
    pattern=r'(?<!\w)'+base.literal_pattern(s)+middle+base.literal_pattern(o)+r'(?!\w)'
    hits=[]
    for m in re.finditer(pattern,text,re.I):
        clean=re.sub(r'\([^)]*\)','',m.group())
        if re.search(base.UNSAFE_CLAUSE,clean,re.I):continue
        hits.append(evidence_span(m,text,template))
    return hits

def extra_proposals(t,text,relations):
    s,r,o=t;choices=[]
    def add(target,rule,hits):
        if target in relations and hits:choices.append(([s,target,o],rule,hits[0]))
    if r in {'publication date','point in time'} and base.date_key(o):
        hits=direct_match(s,o,r'\s*(?:\([^)]{0,100}\)\s*)?(?:(?:was|is)\s+)?(?:founded|established)\s+(?:in|on)\s+',text,'explicit_founding_v4')
        if not hits:
            intro=re.match(r'\s*'+base.literal_pattern(s)+r'\s+(?:is|was)\s+[^.!?\n]{0,160}\b(?:newspaper|journal|publication)\b[^.!?\n]*[.!?]\s*',text,re.I)
            if intro:
                m=re.match(r'The\s+(?:publication|newspaper|journal)\s+was\s+(?:founded|established)\s+(?:in|on)\s+'+base.literal_pattern(o)+r'(?!\w)',text[intro.end():],re.I)
                if m:hits=[{'start':0,'end':intro.end()+m.end(),'span':text[:intro.end()+m.end()],'template':'leading_publication_founding_v4'}]
        add('inception','explicit_founding_v4',hits)
    if r=='educated at':
        existing=base.explicit_event(s,o,r'\b(?:attended|graduated\s+from|studied\s+at|educated\s+at)\b',text,copula=True)
        if not existing:
            middle=r'\s*(?:\([^)]{0,140}\)\s*)?(?:is|was)\s+(?:an?\s+)?(?:longtime\s+|long[- ]serving\s+|emeritus\s+)?(?:professor|lecturer|researcher|teacher|faculty member)\b[^.;!?\n]{0,65}?\bat\s+'
            add('employer','explicit_academic_employment_v4',direct_match(s,o,middle,text,'academic_employment_v4'))
    if r in {'serves','location','located in the administrative territorial entity'}:
        middle=r'\s+(?:is|was)\s+(?:an?\s+)?(?:international\s+|regional\s+|domestic\s+)?(?:airport|railway station|railroad station|station|transport hub)\s+(?:serving|that serves|which serves)\s+'
        hits=direct_match(s,o,middle,text,'transport_service_v4')
        if r=='serves':hits+=direct_match(s,o,r'\s+serves\s+',text,'direct_transport_service_v4') if re.search(r'\b(?:airport|station|terminal|port)\b',s,re.I) else []
        add('place served by transport hub','explicit_transport_service_v4',hits)
    if r=='crosses body of water':
        add('crosses','schema_crosses_v4',direct_match(s,o,r'\s+(?:is|was)\s+(?:an?\s+)?[^.;!?\n]{0,55}?\bbridge\s+(?:spanning|crossing)\s+',text,'bridge_crossing_v4'))
    if r=='country' and o.lower() in (base.CONTINENTS-{'australia'}):
        middle=r'\s+(?:is|was)\s+(?:an?\s+)?(?:peak|mountain|lake|river|island|glacier|range)\b[^.;!?\n]{0,240}?(?:\bin\s+|\bon\s+|,\s*)'
        add('continent','typed_geography_continent_v4',direct_match(s,o,middle,text,'typed_geography_continent_v4'))
    return choices

def semantic_proposals(t,text,relations):
    options=OLD_SEMANTICS(t,text,relations)+extra_proposals(t,text,relations)
    if t[1]=='country' and t[2].lower()=='australia':
        options=[a for a in options if a[0][1]!='continent']
    return list({tuple(n):(n,r,e) for n,r,e in options}.values())

def install():
    base.corrected_fields=corrected_fields
    base.semantic_proposals=semantic_proposals
    base.VERSION='source_guarded_experimental_v4'
    base.ENTITY_RULES.add('unique_source_surname_v4')
    base.RULE_ORDER=['v3_surface_and_explicit_entity','unique_source_surname_v4','v3_and_v4_unique_semantic_proposal','event_date_repair','exact_deduplication']

def self_test():
    import unittest
    class Guards(unittest.TestCase):
        def correction(self,t,text,targets):return semantic_proposals(t,text,set(targets))
        def test_unique_short_name(self):
            self.assertEqual(corrected_fields(['Fletcher','educated at','University of Kentucky'],'Fletcher graduated from the University of Kentucky to pursue Ernie Fletcher dream.')[0][0],'Ernie Fletcher')
        def test_competing_names_preserved(self):
            text='Smith graduated from Yale. John Smith and Jane Smith later joined.'
            self.assertEqual(corrected_fields(['Smith','educated at','Yale'],text)[0][0],'Smith')
        def test_family_preserved(self):
            self.assertIsNone(unique_surname('Smith','employer','Smith was a teacher. John Smith was his father.'))
        def test_foundation(self):
            opts=self.correction(['Daily News','publication date','1976'],'Daily News is a local newspaper. The publication was founded in 1976.',['inception'])
            self.assertEqual(opts[0][0][1],'inception')
        def test_negative_foundation(self):
            self.assertFalse(self.correction(['Daily News','publication date','1976'],'Daily News was not founded in 1976.',['inception']))
        def test_academic_job(self):
            opts=self.correction(['John Smith','educated at','Columbia University'],'John Smith (1918-1994) was a longtime professor of Art History at Columbia University.',['employer'])
            self.assertEqual(opts[0][0][1],'employer')
        def test_education_preserved(self):
            text='John Smith attended Columbia University. John Smith was a professor at Columbia University.'
            self.assertFalse(self.correction(['John Smith','educated at','Columbia University'],text,['employer']))
        def test_transport(self):
            opts=self.correction(['Tres Arroyos Airport','serves','Tres Arroyos'],'Tres Arroyos Airport is an airport serving Tres Arroyos, Argentina.',['place served by transport hub'])
            self.assertEqual(opts[0][0][1],'place served by transport hub')
        def test_named_after_not_inferred(self):
            opts=self.correction(['Tres Arroyos Airport','serves','Tres Arroyos'],'Tres Arroyos Airport is an airport serving Tres Arroyos.',['named after','place served by transport hub'])
            self.assertTrue(all(x[0][1]!='named after' for x in opts))
        def test_schema_blocks(self):
            self.assertFalse(self.correction(['Daily News','publication date','1976'],'Daily News was founded in 1976.',[]))
        def test_supported_roles_preserved(self):
            self.assertFalse(self.correction(['Film X','screenwriter','John Smith'],'Film X is a film written and directed by John Smith.',['screenwriter','director']))
        def test_australia_country_preserved(self):
            self.assertFalse(self.correction(['Lake X','country','Australia'],'Lake X is a lake in Queensland, Australia.',['continent','country']))
        def test_operator_not_owner(self):
            self.assertFalse(self.correction(['Plant X','operator','Company Y'],'Plant X is operated by Company Y.',['owned by']))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Guards))
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':
    if '--self-test' in sys.argv:
        base.self_test();install();self_test()
    else:
        dependency=Path(base.__file__);install();base.__file__=__file__;base.main()
        if '--out' in sys.argv:
            out=base.resolve(Path.cwd(),sys.argv[sys.argv.index('--out')+1])
            for filename in ['summary.json','frozen_run_configuration.json']:
                p=out/filename
                if p.exists():
                    obj=base.json.loads(p.read_text());obj['implementation_dependencies']={str(dependency):base.sha(dependency)}
                    obj['entity_policy']='v3 explicit aliases plus unique source-attested surname resolution; ambiguous/family cases preserved'
                    obj['rule_origin']='Experimental rules informed by historical evaluation diagnostics; independent evaluation still required.'
                    base.dump(p,obj)
