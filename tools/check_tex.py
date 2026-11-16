"""Static sanity checks for paper/main.tex (used because no TeX engine is installed on this machine)."""
import re
import sys
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "paper"
main = (P / "main.tex").read_text(encoding="utf-8")
numbers = (P / "numbers.tex").read_text(encoding="utf-8")
tables = {f.name: f.read_text(encoding="utf-8") for f in sorted((P / "tables").glob("*.tex"))}
problems = []


def strip_comments(s):
    return "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in s.splitlines())


body = strip_comments(main)
inputs = re.findall(r"\\input\{([^}]+)\}", body)
for i in inputs:
    if i != "numbers" and not (P / (i + ".tex")).exists():
        problems.append(f"missing \\input file: {i}")
full = body
for i in inputs:
    if (P / (i + ".tex")).exists():
        full = full.replace(f"\\input{{{i}}}", strip_comments((P / (i + ".tex")).read_text(encoding="utf-8")))

# macros from numbers.tex
defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", numbers))
used = set(re.findall(r"\\([A-Za-z]+)", full))
LATEX_KNOWN = set("""documentclass usepackage input begin end title author maketitle IEEEauthorblockN IEEEauthorblockA IEEEkeywords abstract section subsection
textbf emph texttt textit label ref cite caption centering includegraphics item bibitem url mbox ldots times leq geq sigma overline mathrm text frac max min
cmidrule midrule toprule bottomrule multicolumn footnotesize setlength tabcolsep rightarrow to omega alpha rho IEEEoverridecommandlockouts thebibliography
section  hline par textwidth linewidth tabular table figure itemize align equation cdots left right quad qquad percent ge le ldots  rm sf footnote
tabular  textcolor  small  cm mathbf boldsymbol  amp  nonumber in k  centerline  hspace vspace  dots  textless textgreater  textasciitilde  textbackslash""".split())
undefined = sorted(m for m in used if m not in defined and m not in LATEX_KNOWN and m != "newcommand")
if undefined:
    problems.append(f"macros that are neither in numbers.tex nor known LaTeX (review): {undefined}")

# refs / labels / cites
labels = set(re.findall(r"\\label\{([^}]+)\}", full))
for r in re.findall(r"\\ref\{([^}]+)\}", full):
    if r not in labels:
        problems.append(f"\\ref to undefined label: {r}")
bib = set(re.findall(r"\\bibitem\{([^}]+)\}", full))
cited = set()
for grp in re.findall(r"\\cite\{([^}]+)\}", full):
    cited |= {k.strip() for k in grp.split(",")}
for c in cited - bib:
    problems.append(f"\\cite key without \\bibitem: {c}")
if bib - cited:
    problems.append(f"bibliography items never cited: {sorted(bib - cited)}")

# figures
for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", full):
    if not any((P / f"{g}{ext}").exists() for ext in ("", ".pdf", ".png", ".jpg")):
        problems.append(f"missing figure file: {g}")

# environments & braces
for name, text in [("main(+tables)", full)]:
    envs = re.findall(r"\\(begin|end)\{([^}]+)\}", text)
    stack = []
    for kind, env in envs:
        if kind == "begin":
            stack.append(env)
        elif not stack or stack.pop() != env:
            problems.append(f"{name}: mismatched \\end{{{env}}}")
    if stack:
        problems.append(f"{name}: unclosed environments {stack}")
    clean = re.sub(r"\\[{}]", "", text)
    if clean.count("{") != clean.count("}"):
        problems.append(f"{name}: unbalanced braces ({clean.count('{')} vs {clean.count('}')})")

# unescaped special characters in running text (math, align environments and comments removed first)
text_only = re.sub(r"\\begin\{(align|equation)\}.*?\\end\{\1\}", " ", body, flags=re.S)
text_only = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", " ", text_only, flags=re.S)
text_only = re.sub(r"\\(input|includegraphics)(\[[^\]]*\])?\{[^}]*\}", " ", text_only)   # underscores in file paths are legal
text_only = re.sub(r"\$[^$]*\$", " ", text_only)
for ln_no, ln in enumerate(text_only.splitlines(), 1):
    if re.search(r"(?<!\\)&", ln):
        problems.append(f"main.tex(text) line {ln_no}: unescaped & : {ln.strip()[:70]}")
    if re.search(r"(?<!\\)_", ln):
        problems.append(f"main.tex(text) line {ln_no}: bare underscore: {ln.strip()[:70]}")
for name, t in tables.items():
    for ln_no, ln in enumerate(t.splitlines(), 1):
        if re.search(r"(?<!\\)_", re.sub(r"\$[^$]*\$", "", ln)) and "_\\text" not in ln:
            problems.append(f"{name}:{ln_no}: underscore outside math: {ln[:60]}")
        if re.search(r"(?<!\\)%", ln):
            problems.append(f"{name}:{ln_no}: unescaped % (would comment out the rest): {ln[:60]}")

# usage stats
unused = sorted(d for d in defined if d not in used)
print(f"macros defined: {len(defined)} | used in text/tables: {len(defined) - len(unused)} | labels: {len(labels)} | cites: {len(cited)} | tables: {len(tables)}")
print("figures referenced:", len(re.findall(r"\\includegraphics", full)))
if problems:
    print("\nPROBLEMS:"); [print(" -", p) for p in problems]; sys.exit(1)
print("static checks passed")
