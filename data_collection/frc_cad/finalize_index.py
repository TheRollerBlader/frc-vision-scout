import json, collections
p='robots_final.json'; m=json.load(open(p)); B={e['id']:e for e in m['robots']}
# 1) remove false-positive "team mismatch" flags where the detected 'team' is actually the season year
for e in m['robots']:
    ns=e.get('notes',[]); keep=[]
    for n in ns:
        if n.startswith('Possible team mismatch: GrabCAD model name/author mentions team(s) ['):
            t=n.split('[',1)[1].split(']')[0]
            if all(x.strip().isdigit() and 2000<=int(x)<=2030 for x in t.split(',')):
                continue  # year number, not a team
        if n not in keep: keep.append(n)   # also de-duplicates repeated split notes
    if len(keep)!=len(ns):
        if e.get('mismatch_flag') and not any('mismatch' in n.lower() for n in keep): e.pop('mismatch_flag',None)
    e['notes']=keep
def add(i, note, **flags):
    e=B[i]; e.setdefault('notes',[]).append(note); e.update(flags); e['needs_attention']=True
add('2026-2026-r36','ATTENTION: team mismatch - files are Team 3255 SuperNURDs 2026 robot, filed under page listing (2026 Senai).',mismatch_flag=True)
add('2024-3522-r266','ATTENTION: team mismatch - files are Team 3255 SuperNURDs 2024 robot, filed under page listing (3522 CBoT´S4).',mismatch_flag=True)
add('2019-2360-r577',"ATTENTION: team mismatch - GrabCAD model is 'FRC 2630 Grimlock'; the same model is also listed for 2019 team 2630 (2019-2630-r583, marked duplicate). Likely belongs to team 2630, filed under 2360 per page listing.",mismatch_flag=True)
add('2019-2630-r583',"Duplicate of 2019-2360-r577, but the GrabCAD model ('FRC 2630 Grimlock') actually appears to be this team's (2630) robot; files live in the 2019/2360 folder.")
add('2014-3339-r862',"ATTENTION: year mismatch - GrabCAD model is titled '2013 | 3339 - BumbleB' and the file is 3339BumbleB2013.stp; the same model is also listed as 2013-3339-r892 (duplicate). Likely the 2013 robot, filed under 2014 per page listing.",year_mismatch_flag=True)
add('2013-3339-r892',"Duplicate of 2014-3339-r862; the model appears to actually be the 2013 robot, so files live in the 2014/3339 folder.")
add('unknown_year-250-r997',"ATTENTION: year unknown on the source page; GrabCAD model titled '2023 | 250 - Dynamos' and file is '2023 Robot Export.stp' - likely the 2023 robot. Filed under unknown_year.",year_mismatch_flag=True,likely_year=2023)
add('2013-45-r866',"ATTENTION: possible year mismatch - GrabCAD model titled '2012 | 45 - Technokats' (bundle named 2012-45-technokats-2) though listed as 2013. A separate 2012-45 entry (r896) exists with a different model.",year_mismatch_flag=True)
add('2017-2168-r748',"ATTENTION: possible year mismatch - model is '2017 Off Season Prototype' but the STEP file is named '2018 Robot Assembly'.",year_mismatch_flag=True)
add('2013-2363-r888',"ATTENTION: partial - the only STEP is 'genome_epsilon_arm1.stp' (arm subassembly), not the full robot.",step_may_be_partial=True)
B['2011-45-r922']['notes'].append("ATTENTION: electronics only - the 3 STEPs are power distribution board, wireless bridge and digital sidecar; no robot assembly STEP."); B['2011-45-r922']['needs_attention']=True
add('2022-6404-r449',"ATTENTION: components only - STEPs are electronics/COTS parts (Talon SRX, VRM, breaker, REV parts); no robot assembly STEP.",step_may_be_partial=True)
add('2016-4001-r805',"Two near-identical files (2016_robot_asm.stp.zip and '2016_robot_asm.stp (2).zip', both ~39.8 MB) were on GrabCAD; both uploaded.")
for i in ('2016-1923-r796','2018-2655-r684','2018-751-r659','2018-2067-r678','2019-2881-r588','2018-2767-r685','2021-6404-r480','2019-6404-r625'):
    B[i]['needs_attention']=True
for i in ('2020-6995-r519','2019-6995-r631'):
    B[i]['notes'].append('Failed twice (stager pass 1 and retry pass 2): GrabCAD model returns 404 (removed/private/renamed).'); B[i]['needs_attention']=True
B['2018-1690-r671']['needs_attention']=True
L=m['robots']
s=m['summary']; s['status_counts']=dict(collections.Counter(e['status'] for e in L))
s['mismatch_flagged']=sum(1 for e in L if e.get('mismatch_flag'))
s['year_mismatch_flagged']=sum(1 for e in L if e.get('year_mismatch_flag'))
s['step_may_be_partial']=sum(1 for e in L if e.get('step_may_be_partial'))
s['needs_attention']=sorted(e['id'] for e in L if e.get('needs_attention'))
s['uploaded_gb']=round(s['uploaded_bytes']/1e9,2)
s['uploaded_per_year']=dict(sorted(collections.Counter(str(e['year']) if e['year'] else 'unknown_year' for e in L if e['status'] in('uploaded','uploaded_partial')).items(),reverse=True))
s['split_zip_robots']=sorted(e['id'] for e in L if e['status']=='uploaded' and any(str(f.get('compressed','')).endswith('split') for f in e['files']))
m['notes'].append("needs_attention / mismatch_flag / year_mismatch_flag / step_may_be_partial mark entries the user should review; team/year folders follow the source page listing and were never changed.")
json.dump(m,open(p,'w'),indent=1,ensure_ascii=False)
print(json.dumps(s,indent=1))
