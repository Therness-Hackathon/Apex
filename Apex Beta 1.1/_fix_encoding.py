# -*- coding: utf-8 -*-
"""
Fix all garbled mojibake + emoji strings in dashboard_app.py.
Strategy: read bytes as latin-1, re-encode to proper unicode aware replacements,
then write back as UTF-8. Also strips all emoji icon arguments from section_header()
and replaces garbled nav-link labels with clean ASCII.
"""
import re, pathlib

p = pathlib.Path(r'c:\New folder\Apex Beta 1.1\dashboard_app.py')
raw = p.read_bytes()

# ------------------------------------------------------------------
# 1. The file is valid UTF-8 but many string literals contain bytes
#    from emoji/box-drawing that get mangled by the editor display.
#    Re-read as UTF-8 (best effort, replacing undecodable bytes).
# ------------------------------------------------------------------
txt = raw.decode('utf-8', errors='replace')

# ------------------------------------------------------------------
# 2. Explicit string-level replacements for every visible mojibake
#    pattern that ends up in rendered HTML strings.
# ------------------------------------------------------------------
SUBS = [
    # Box-drawing in run-banner (comments — safe)
    ('\u2550', '='), ('\u2554', '+'), ('\u255a', '+'), ('\u2551', '|'), ('\u2500', '-'),
    # Typography
    ('\u2014', '--'),   # em dash
    ('\u2013', '-'),    # en dash
    ('\u00d7', 'x'),    # multiplication
    ('\u00b7', '.'),    # middle dot
    ('\u00a0', ' '),    # non-breaking space
    ('\u2192', '->'),   # right arrow
    # Bullet used in defect-code-map spans
    ('\u25cf', 'o'),    # filled circle
    # Info symbol
    ('\u2139\ufe0f', 'i'),
    ('\u2139', 'i'),
    # Lightning bolt
    ('\u26a1', '*'),
    # All remaining replacement-char placeholders
    ('\ufffd', ''),
]

for bad, good in SUBS:
    txt = txt.replace(bad, good)

# ------------------------------------------------------------------
# 3. Strip garbled / emoji icon from section_header() second argument.
#    Keep the title; replace icon with empty string.
#    Handles both  section_header("Title", "ICON")
#              and  section_header("Title", 'ICON')
# ------------------------------------------------------------------
# double-quoted icon
txt = re.sub(
    r'(section_header\("(?:[^"\\]|\\.)*",\s*)"(?:[^"\\]|\\.)*"(\))',
    r'\1""\2', txt
)
# single-quoted icon
txt = re.sub(
    r"(section_header\('(?:[^'\\]|\\.)*',\s*)'(?:[^'\\]|\\.)*'(\))",
    r"\1''\2", txt
)

# ------------------------------------------------------------------
# 4. Fix garbled nav-link labels — replace the whole label string
#    for the 4 tab nav items with clean plain text.
# ------------------------------------------------------------------
NAV_FIXES = [
    # pattern (partial original label with garbled prefix)  -> clean label
    (r'"[^"]*Overview[^"]*"(\s*,\s*href="#",\s*id="tab-overview")',
     r'"Overview"\1'),
    (r'"[^"]*Performance[^"]*"(\s*,\s*href="#",\s*id="tab-perf")',
     r'"Performance"\1'),
    (r'"[^"]*Submission[^"]*"(\s*,\s*href="#",\s*id="tab-sub")',
     r'"Submission"\1'),
    (r'"[^"]*About[^"]*"(\s*,\s*href="#",\s*id="tab-about")',
     r'"About"\1'),
]

for pat, repl in NAV_FIXES:
    txt = re.sub(pat, repl, txt)

# ------------------------------------------------------------------
# 5. Fix garbled subtitle / footer strings
# ------------------------------------------------------------------
# "AI-Powered Weld Defect Classification  Â·  Audio ML Pipeline  Â·  Hackathon 2026"
txt = re.sub(r'AI-Powered Weld Defect Classification\s+[^\w]*\s+Audio ML Pipeline\s+[^\w]*\s+Hackathon 2026',
             'AI-Powered Weld Defect Classification  ·  Audio ML Pipeline  ·  Hackathon 2026', txt)

# footer "Apex Weld Quality  Â·  Hackathon ..."
txt = re.sub(r'Apex Weld Quality\s+[^\w,.(0-9]*\s+Hackathon 2026\s+[^\w,.(0-9]*\s+LR \+ SVM Audio Pipeline\s+[^\w,.(0-9]*\s+FinalScore',
             'Apex Weld Quality  ·  Hackathon 2026  ·  LR + SVM Audio Pipeline  ·  FinalScore', txt)

# ------------------------------------------------------------------
# 6. Fix garbled Competition info table rows
# ------------------------------------------------------------------
txt = txt.replace('Hackathon â\x80\x94 Weld Quality Classification', 'Hackathon -- Weld Quality Classification')
txt = txt.replace('Hackathon -- Weld Quality Classification', 'Hackathon - Weld Quality Classification')

# ------------------------------------------------------------------
# 7. Fix garbled model architecture descriptions
# ------------------------------------------------------------------
txt = re.sub(r'LR\(C=7\)[^\w,.()\[\]]*SVM', 'LR(C=7) + SVM', txt)
txt = re.sub(r'zero-mean,[^\w]*unit-variance', 'zero-mean, unit-variance', txt)
txt = re.sub(r'115 rows[^\w]*sample_id', '115 rows -> sample_id', txt)
txt = re.sub(r'sample_id,[^\w]*pred_label_code,[^\w]*p_defect[^\w]*outputs/submission.csv',
             'sample_id, pred_label_code, p_defect -> outputs/submission.csv', txt)

# ------------------------------------------------------------------
# 8. Fix garbled strings in info_row calls (About tab table)
# ------------------------------------------------------------------
txt = re.sub(r'Hackathon[^\w]*Weld Quality Classification', 'Hackathon - Weld Quality Classification', txt)
txt = re.sub(r'Binary \+ 7-class', 'Binary + 7-class', txt)
txt = re.sub(r'FinalScore.*\(target > 0\.80\)', lambda m: re.sub(r'[^\x20-\x7e.]', '', m.group()), txt)

# ------------------------------------------------------------------
# 9. Fix docstring header
# ------------------------------------------------------------------
txt = txt.replace('Apex Weld Quality â\x80\x94 Hackathon Dashboard (v3)', 
                  'Apex Weld Quality - Hackathon Dashboard (v3)')
txt = re.sub(r'Apex Weld Quality[^\w\n]*Hackathon Dashboard \(v3\)',
             'Apex Weld Quality - Hackathon Dashboard (v3)', txt)

# ------------------------------------------------------------------
# 10. Generic cleanup: replace any remaining 3-char mojibake sequences
#     commonly seen: Ã  â  Â  followed by a non-ASCII char
# ------------------------------------------------------------------
txt = re.sub(r'[ÃâÂð][^\x00-\x7f\s]', '', txt)

# ------------------------------------------------------------------
# Write back
# ------------------------------------------------------------------
p.write_text(txt, encoding='utf-8')
print(f'Fixed. File size: {len(txt)} chars')

# Quick sanity - show any remaining suspicious sequences
import re as _re
suspects = _re.findall(r'[ðÃâÂ][^\x00-\x7f]', txt)
if suspects:
    print(f'Remaining suspect sequences ({len(suspects)}): {set(suspects[:20])}')
else:
    print('No remaining mojibake sequences found.')
