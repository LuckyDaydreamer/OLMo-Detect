#!/usr/bin/env python3
"""Regenerate the two supervised main tables (sup_two) and apply the user's presentation edits (23 Sep): no vertical bars, FSD bases without MinK++
(original FSD setting), makecell row labels 'FSD\\(base)', captions verbatim from tables_latest/.caption_sup_{a,b}.txt. Writes tables_latest/table:olmo_detect_overall_supervised.tex"""
import re,subprocess,sys
src=subprocess.run([sys.executable,"analysis/paper_tables.py","sup_two"],capture_output=True,text=True).stdout
tabs=re.findall(r"\\begin\{table\*\}.*?\\end\{table\*\}",src,re.S); assert len(tabs)==2, len(tabs)
capa=open("tables/.caption_sup_a.txt").read(); capb=open("tables/.caption_sup_b.txt").read()
out=[]
for i,tb in enumerate(tabs):
    tb=re.sub(r"\\begin\{tabular\}\{ll\|cccc\|cccc\|cccc\|cccc\|cccc\}",r"\\begin{tabular}{llcccccccccccccccccccc}",tb)
    tb=re.sub(r"\\caption\{.*?\}\n\\label","\\\\caption{"+(capa if i==0 else capb).replace("\\","\\\\")+"}\n\\\\label",tb,flags=re.S)
    if i==0:
        # one bold per column across all four rows (Sup-CAMIA/MIA-Tuner x In-domain/LODO)
        lines=tb.split("\n"); idx=[k for k,l in enumerate(lines) if re.match(r"^(\\multirow\{2\}\{\*\}\{\\textbf\{(Sup-CAMIA|MIA-Tuner)\}\} & In-domain| & LODO)",l)]
        assert len(idx)==4, idx
        parsed=[]
        for k in idx:
            parts=lines[k].rstrip().rstrip("\\\\").split("&"); head=parts[:2]; cells=[c.strip() for c in parts[2:]]
            cells=[re.sub(r"\\textbf\{([^}]*)\}",r"\1",c) for c in cells]; parsed.append((k,head,cells))
        ncol=len(parsed[0][2])
        for j in range(ncol):
            vals=[]
            for r,(k,head,cells) in enumerate(parsed):
                m=re.match(r"^([0-9.]+)",cells[j]); vals.append(float(m.group(1)) if m else None)
            have=[v for v in vals if v is not None]
            if not have: continue
            b=vals.index(max(have)); c=parsed[b][2][j]; parsed[b][2][j]=re.sub(r"^([0-9.]+)",r"\\textbf{\1}",c,count=1)
        for k,head,cells in parsed: lines[k]=" & ".join([h.strip() for h in head]+cells)+" \\\\"
        tb="\n".join(lines)
    if i==1:
        # drop MinK++ block (multirow line + its LODO line + preceding \midrule)
        tb=re.sub(r"\\midrule\n\\multirow\{2\}\{\*\}\{\\textbf\{FSD \(MinK\+\+\)\}\}.*?\\\\\n & LODO.*?\\\\\n","",tb,flags=re.S)
        tb=re.sub(r"\\multirow\{2\}\{\*\}\{\\textbf\{FSD \(([^)]*)\)\}\}",lambda m:"\\multirow{2}{*}{\\makecell[l]{\\textbf{FSD}\\\\\\textbf{("+m.group(1)+")}}}",tb)
        assert "MinK++" not in tb
    out.append(tb)
open("tables/table:olmo_detect_overall_supervised.tex","w").write("\n".join(out)+"\n"); print("WRITTEN tables_latest/table:olmo_detect_overall_supervised.tex")
