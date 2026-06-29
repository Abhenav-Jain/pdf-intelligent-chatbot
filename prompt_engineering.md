# Prompt Engineering — Concepts Used in This Project (Deep Dive)

Har concept ke teen parts hain: **(a) General principle** (theory, kahin
bhi apply hoti hai), **(b) Humne yahan kaise use ki** (concrete example,
actual prompt text ke saath), **(c) Kya seekha / kyun zaroori thi**.

---

## 1. Role / Persona Definition

**(a) Principle:** Model ko ek specific ROLE assign karna uske output ke
"register" (tone, scope, behavior) ko anchor karta hai. Yeh prompt ka
SABSE PEHLA decision hota hai.

**(b) Humne kya kiya:**
```
"You are an advanced AI assistant responsible for providing responses to
user questions based ONLY on the given PDF context."
```
"ONLY" word yahan critical hai — yeh role definition ke ANDAR hi ek
constraint bhi daal deta hai (scope-limited assistant, general-knowledge
assistant nahi).

**(c) Lesson:** Role definition akela kaafi NAHI hota — humne dekha ki
"ONLY the given context" likhne ke baad bhi model occasionally external
knowledge use kar leta tha jab tak hamne ALAG, explicit "grounding rules"
(neeche #9) bhi add nahi kiye. Role sirf ek FRAME hai, har constraint
khud se explicitly bolna padta hai.

---

## 2. Structured Output Enforcement (JSON Schema in Prompt)

**(a) Principle:** Jab tumhe predictable, parseable output chahiye (free
text nahi), model ko EXACT schema dikhana chahiye — including jab schema
KAI VARIANTS mein badalta hai based on input type.

**(b) Humne kya kiya:** 4 alag shapes, explicitly likhe (curly braces
escape karke, kyunki LangChain templating `{}` ko variable samajhta hai):
```
1. RELEVANT QUERY: {{"Tag": "...", "Response": "..."}}
2. INCOMPLETE/VAGUE: {{"Response": "..."}}
3. NON-RELEVANT: {{"Response": "..."}}
4. GREETING: {{"Response": "..."}}
```

**(c) Lesson — ek genuine bug isi se mila:** Maine pehli baar yeh likha
tha bina escaping ke (`{"Tag": ...}` literal braces). LangChain ka
template-engine `{}` ko placeholder samajhta hai — **pehli hi query crash
ho jaati**. Lesson: jab tumhara INSTRUCTION text khud JSON/code-jaisa
dikhta hai, check karo ki tumhara templating-system uss syntax ko literal
text samjhega ya apna khud ka syntax.

---

## 3. Few-Shot Examples ("Show, Don't Just Tell")

**(a) Principle:** Abstract rule se zyada powerful hota hai EK CONCRETE,
worked example — especially jab rule "is row mein 3 columns hain, unhe
sahi se map karo" jaisa spatial/structural hota hai jo describe karna
mushkil hai.

**(b) Humne kya kiya:** Jab table-column-shifting bug mila (E-4 ki
"CPU TOO HOT" galat column mein chali gayi thi), abstract rule
("preserve columns") kaafi nahi thi — humne EXACT row diya:
```
Example: the source row "E-4" "CPU TOO HOT" Control board overheating
Turn switch to OFF... — has exactly THREE columns: Display = '"E-4"
"CPU TOO HOT"', Cause = 'Control board overheating', Correction = '...'
```

**(c) Lesson:** Worked examples sirf "helpful" nahi hote — kabhi-kabhi
woh hi REQUIRED hote hain jab rule itself ambiguous-sounding ho. Lekin
isme RISK bhi hai: humne dekha ki example SIRF Display/Cause/Correction
shape ke liye diya tha, toh model ne ek ALAG-shape table (Fig/Item
No./Description/Function) ko table samjha hi nahi — usko numbered list
bana diya. **Fix: example ko explicitly "this applies to ANY parallel-
column source, not just this one" se generalize kiya.** Lesson: ek
SPECIFIC example dene ke saath EXPLICIT generalization statement bhi
dena padta hai, warna model usse OVER-narrowly interpret kar leta hai.

---

## 4. Explicit Negative Instructions (Anti-Patterns)

**(a) Principle:** Kabhi-kabhi "yeh karo" se zyada zaroori hota hai "yeh
MAT karo" — especially jab koi specific, observed failure-pattern ho jo
model naturally karta hai.

**(b) Humne kya kiya:** Multiple jagah:
- "Never use a shorthand like '(Same as Error Code X)'" (verified bug:
  E-10C/E-10D ko galat se E-10B jaisa label kiya gaya tha)
- "Never blank out an entire row as 'not available' just because one
  column is thin"
- "Never restart numbering at 1 unless beginning a genuinely new,
  separately-headed list"

**(c) Lesson:** Negative instructions SABSE EFFECTIVE hoti hain jab woh
ek ACTUAL OBSERVED bug se aayi hain (generic "be careful" se kaam nahi
chalta) — har "never X" line is conversation mein kisi real, diagnosed
failure se aayi thi, koi speculative/hypothetical nahi.

---

## 5. Instruction Decoupling & Precedence Rules

**(a) Principle:** Jab do instructions interact kar sakti hain
(especially conflict-jaisi lag sakti hain), explicitly batana padta hai
KAUN OVERRIDE karta hai KISKO, aur KAUN INDEPENDENT hai.

**(b) Humne kya kiya:** "FORMAT REQUIREMENT" (table/comparison maango)
aur "not available" rule dono saath thay — model confuse ho gaya tha
("table format chahiye, but main perfect table nahi bana sakta, isliye
'not available' bol deta hoon" — yeh galat reasoning thi). Fix:
```
"A FORMAT REQUIREMENT is a presentation instruction ONLY: it never
changes whether information is available. Decide whether the context
contains the answer FIRST; only then apply the requested format."
```

**(c) Lesson:** Yeh exact bug ("table maango to 'not available' aata
tha") sirf ISI decoupling-statement se fix hua — sirf "do both things"
kehna kaafi nahi tha, explicitly batana pada ki yeh do SEPARATE
decisions hain, sequence mein.

---

## 6. Branching/Conditional Logic Inside a Single Prompt

**(a) Principle:** Ek hi prompt mein multiple "if this kind of input,
do X; if that kind, do Y" branches ho sakte hain — jaise ek decision tree.

**(b) Humne kya kiya:** Poora 4-category classification system
(Relevant/Vague/Non-Relevant/Greeting) khud ek branching structure hai —
model ko PEHLE classify karna hai, FIR uss category ke rules follow
karne hain.

**(c) Lesson:** Jitni zyada branches/categories, utna zyada **prompt
size** badhta hai (humne measure kiya — system prompt 700 tokens se
~2400 tokens tak pahunch gaya across iterations) — aur lambi prompt khud
ek NAYI problem ban sakti hai (next point dekho).

---

## 7. Token Budget Awareness in Prompt Design

**(a) Principle:** Prompt EK RESOURCE hai — context window ka hissa khata
hai, aur lamba/complex prompt khud bhi model ki "attention" ko dilute kar
sakta hai (zyada rules = har rule par kam focus).

**(b) Humne kya kiya:** Har baar jab prompt mein naya rule add kiya,
maine actual TOKEN COUNT measure kiya (`len(sys_msg)//4`) aur internal
budget-estimate ko sync kiya — kyunki ek hardcoded "700 tokens" estimate
asal mein 1270, fir 1750, fir 2200, fir 2450 tak drift kar gaya tha,
jisse context-budget calculations galat ho rahe the.

**(c) Lesson:** Ek important, counter-intuitive finding: jab maine prompt
ko bohot saari rules se overload kiya (table-fidelity + completeness +
numbering + stay-on-topic + paired-variants, sab ek saath), model ne
**content rows skip karna shuru kar diya** — possibly kyunki itni saari
COMPETING instructions follow karne mein "bandwidth" kam pad gayi.
Isliye humne SIMPLIFY bhi kiya beech mein (kuch redundant rules hataye)
— "zyada instructions = zyada better" hamesha sahi nahi hota.

---

## 8. Temperature as a Prompt-Adjacent Control Lever

**(a) Principle:** Temperature prompt TEXT nahi hai, lekin yeh prompt ke
SAATH milke decide karta hai ki model kitna "literal"/deterministic vs
"creative"/varied response dega — isliye yeh prompt-engineering ka hi
hissa hai.

**(b) Humne kya kiya:** **Differentiated temperature per task:**
- Final answer generation: **temp=0** hamesha (factual accuracy, JSON
  schema compliance ke liye determinism zaroori)
- Query rewriting (normal queries): **temp=0.1** (thodi diversity,
  retrieval ke liye alag phrasings try karne ke liye)
- Query rewriting (comprehensive/"all of X" queries): **temp=0**
  (determinism > diversity jab completeness critical ho — diversity ka
  fayda nahi jab tumhe HAR baar same, exhaustive result chahiye)

**(c) Lesson:** Temperature "ek size fits all" nahi hota — alag steps ke
alag goals hote hain (diversity vs determinism), aur SAME pipeline mein
DIFFERENT temperature use karna valid, deliberate design choice hai.

---

## 9. Grounding & Anti-Hallucination Rules

**(a) Principle:** RAG (Retrieval-Augmented Generation) systems mein
sabse critical rule: model ko explicitly batana ki sirf RETRIEVED context
use kare, apna "training data knowledge" nahi.

**(b) Humne kya kiya:**
```
"Never use external knowledge or information not present in the provided
context — if something isn't in the context, treat it as not available."
```
Plus uncertainty-signaling: "[Low Confidence]" prefix jab model genuinely
unsure ho.

**(c) Lesson:** Grounding rule SIRF top-level mein kaafi nahi — humne
PER-FIELD level tak le jaana pada ("agar Cause column missing hai but
Correction available hai, sirf Cause ko 'not specified' maro, pura row
blank mat karo") — generic "ground your answer" se zyada GRANULAR control
chahiye hota hai jab tumhara content STRUCTURED (table rows) ho.

---

## 10. Scoped Uncertainty Signaling

**(a) Principle:** Model ko apni confidence EXPRESS karne dena, lekin
SCOPE clearly define karna — warna model "all or nothing" confidence
dega (pura answer confident, ya pura "not sure").

**(b) Humne kya kiya:** "[Low Confidence]" ko explicitly PER-PIECE scope
kiya, na ki per-response:
```
"...scope this to the SPECIFIC missing or uncertain piece only... never
blank out an entire row as 'not available' just because one column is
thin."
```

**(c) Lesson:** Yeh fix directly E-84/E-85 series ke "Cause not
specified, but Correction text preserved" wale CORRECT behavior ko
enable kiya — pehle pura row "[Low Confidence] not available" ban jaata
tha jabki Correction part genuinely available tha.

---

## 11. Topic/Scope Boundary Instructions

**(a) Principle:** RAG mein retrieval kabhi "perfectly scoped" context
nahi deta — adjacent/unrelated content bhi aa sakta hai. Model ko
explicitly batana padta hai "context mein hone ka matlab 'answer mein
include karo' nahi hota."

**(b) Humne kya kiya:** "STAY ON TOPIC" rule, jo iteratively strengthen
hui:
- Pehla version: TOC-jaisi listing-content ko ignore karo
- Doosra version: poore ALAG sections (Error Codes, Programming) ko bhi
  ignore karo jab unrelated topic pucha gaya ho
- Teesra version: **immediately-adjacent** section bhi out-of-scope hai
  (Control Overview ke baad Display Options leak ho gaya tha, despite
  pehle do versions already present hone ke)

**(c) Lesson:** Yeh ek progressively-discovered rule thi — har round mein
ek NAYA tareeka mila jisse off-topic content leak ho sakta hai, aur har
baar rule ko THODA SPECIFIC karna pada. General principle: "stay on
topic" jaisi instruction abstract rehti hai jab tak tumhe EXACT failure
mode pata na chale jisko target karna hai.

---

## 12. Iterative Refinement Methodology (the meta-lesson)

**(a) Principle:** Prompt engineering EK-SHOT activity nahi hai — yeh
empirical, test-driven process hai: hypothesis → test → observe failure
→ diagnose ROOT CAUSE (na ki symptom) → targeted fix → re-test.

**(b) Humne kya kiya — poora yeh conversation isi pattern mein chala:**
Har round mein: ek specific query test ki → exact output ko ground-truth
se compare kiya → root cause diagnose kiya (jo kabhi prompt mein hota
tha, kabhi retrieval mein, kabhi dono mein) → fix likha → SYNTHETIC TEST
likha jo bug ko reproduce kare → fix verify kiya → real PDF pe phir test
kiya.

**(c) Lesson — sabse important wala:** **Prompt engineering ki apni
limits hain.** Kuch bugs (jaise "TOC page retrieve ho gaya kyunki word
'table' match ho gaya") prompt se fix NAHI ho sakte the — unhe
RETRIEVAL-level code-fix chahiye tha. Aur kuch bugs (numbering restart,
output truncation) jinko humne pehle PURE prompt-instruction se fix karne
ki koshish ki, woh PARTIALLY hi reliable nikle — isliye humne aakhir mein
CODE-LEVEL VERIFICATION add kiya (token-count check, numbering-pattern
regex check) jo model ke output ko POST-HOC verify karta hai aur zaroorat
padne par retry karta hai. **Yeh sabse bada lesson hai: prompt engineering
behtareen FIRST LINE OF DEFENSE hai, lekin production-grade reliability
ke liye CODE-LEVEL CHECKS bhi chahiye — sirf "model ko sahi se bolna" pe
100% depend nahi kar sakte.**

---

## Quick reference — kahan kya use hua (file mein dhoondne ke liye)

| Concept | Function/Section in `logic_file.py` |
|---|---|
| Role + JSON schema | `build_langchain_prompt()` — system_prompt |
| Few-shot examples | `rewrite_query()` prompt; TABLE COLUMN FIDELITY block |
| Negative instructions | "(Same as X)", "never restart numbering", etc. |
| Decoupling/precedence | FORMAT REQUIREMENT rules in `detect_response_format()` |
| Branching (4 categories) | system_prompt's RELEVANT/VAGUE/NON-RELEVANT/GREETING |
| Temperature control | `rewrite_query(temperature=...)`, `build_chain()` |
| Grounding rules | "ONLY the given PDF context", per-field confidence scoping |
| Scope boundaries | "STAY ON TOPIC" block |
| Code-level verification (the limit of prompting) | `_looks_truncated_by_token_limit()`, `_has_broken_numbering()`, `_looks_like_not_found()` |