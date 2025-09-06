import json
from tenacity import retry, wait_exponential, stop_after_attempt

def _oai_client():
    from openai import OpenAI
    return OpenAI()

def _model_name(args):
    return args.model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

ANALYSIS_SCHEMA = {
    "name": "EdTechIssueExtraction",
    "schema": {
        "type": "object",
        "properties": {
            "post_id": {"type": "string"},
            "subreddit": {"type": "string"},
            "title": {"type": "string"},
            "theme": {"type": "string"},
            "sub_theme": {"type": "string"},
            "stakeholders": {"type": "array", "items": {"type": "string"}},
            "problem_statements": {"type": "array", "items": {"type": "string"}},
            "questions_users_ask": {"type": "array", "items": {"type": "string"}},
            "proposed_solutions": {"type": "array", "items": {"type": "string"}},
            "useful_links": {"type": "array", "items": {"type": "string"}},
            "sentiment": {"type": "string", "enum": ["negative","mixed","neutral","positive"]},
            "severity_1to5": {"type": "integer", "minimum":1, "maximum":5},
            "dedupe_key": {"type": "string"}
        },
        "required": ["post_id","subreddit","title","theme","problem_statements","dedupe_key"]
    }
}


SYSTEM_PROMPT = """RAID PROMPT

R — ROLE
You are an EdTech industry analyst and information extractor. You read Reddit posts (and optional top comments) and return structured, decision-ready data about problems in the Education/EdTech space.

A — AUDIENCE
Your output is consumed by a product & research team building dashboards in CSV/Excel/BI to identify recurring issues, their severity, stakeholders, and actionable solution ideas.

I — INTENT
For each post, extract normalized fields that let us quantify and cluster problems:
- Categorize into a consistent EdTech taxonomy.
- Pull problem statements, user questions, proposed solutions, and useful links.
- Estimate sentiment and real-world severity.
- Provide a concise dedupe_key for grouping similar items.

D — DETAILS (RULES, TAXONOMY, RUBRICS, OUTPUT)
1) Taxonomy (theme): choose ONE primary theme from this list:
   Pricing; Certification Value; Content Quality; Outdated Curriculum; LMS/Platform UX; Engagement/Completion; 
   Assessment Integrity (cheating, proctoring); Teacher Workload; Student Motivation; Policy/Compliance; 
   Accessibility/Equity; Onboarding/Support; Performance/Scale/Reliability; Data/Privacy; Monetization for Creators; Other.
   - sub_theme: short phrase under the theme (e.g., “price hikes vs value”, “mobile app bugs”, “grading fairness”, “outdated Python version”).
2) Stakeholders (multi-select, pick those clearly affected): Students; Teachers; Parents; School/Uni Admins; Policymakers; EdTech Startups/Founders; Employers.
3) Problem extraction:
   - Write concrete, user-voiced problem_statements (not generic summaries).
   - Extract questions_users_ask (direct or implied “how do I…?”).
   - Extract proposed_solutions (from post/comments; if missing, infer practical next steps but label them as “suggested”).
4) Links:
   - Collect any URLs in post/comments that look relevant (docs, blogs, product pages, news).
   - Do NOT invent links. If none, return an empty list.
5) Sentiment & Severity:
   - sentiment ∈ {negative, mixed, neutral, positive}. Prefer negative/mixed when complaints dominate.
   - severity_1to5: 1 = minor annoyance; 3 = noticeable friction; 5 = major impact (learning outcomes, compliance, integrity, outages, widespread cost/credential concerns).
6) Dedupe:
   - dedupe_key = short canonical phrase suitable for grouping (e.g., “Certificates not recognized by employers”, “Course content outdated”, “Proctoring false positives”, “LMS mobile app crashes”).
7) Scope & noise:
   - Focus on Education/EdTech contexts; if off-topic, set theme = “Other” and still extract useful structure if any.
   - If content is mostly memes/low-signal, minimize problem_statements and keep severity ≤ 2.
8) Output format:
   - Return strictly JSON using the provided schema (the caller enforces it). 
   - Be concise but specific. Avoid repetition. No explanations outside JSON.
"""



def _to_prompt_items(batch):
    items = []
    for p in batch:
        body = (p.get("title","").strip() + "\n\n" + p.get("selftext","").strip()).strip()
        if not body:
            body = p.get("title","").strip()
        comments = p.get("top_comments", []) or []
        items.append({
            "post_id": p["id"],
            "subreddit": p["subreddit"],
            "title": p.get("title",""),
            "text": body,
            "comments": comments[:5],
            "permalink": p.get("permalink","")
        })
    return items

@retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(6))
def analyze_batch_with_openai(args, batch):
    client = _oai_client()
    model = _model_name(args)
    prompt_items = _to_prompt_items(batch)

    # We’ll ask for an array of per-post analyses
    response = client.responses.create(
        model=model,
        response_format={"type":"json_schema","json_schema":{"name":"BatchResult","schema":{
            "type":"object",
            "properties":{"results":{"type":"array","items":ANALYSIS_SCHEMA["schema"]}},
            "required":["results"]
        }}},
        input=[{
            "role":"system","content":SYSTEM_PROMPT
        },{
            "role":"user","content":
                "Analyze these posts and return a JSON object with 'results' (one element per post).\n" +
                json.dumps(prompt_items, ensure_ascii=False)
        }]
    )
    txt = response.output_text
    data = json.loads(txt)
    return data["results"]

def _flatten_for_row(a: dict) -> dict:
    def join(arr, sep=" | "):
        if not arr: return ""
        return sep.join([str(x).replace("\n"," ").strip() for x in arr if str(x).strip()])

    return {
        "post_id": a.get("post_id",""),
        "subreddit": a.get("subreddit",""),
        "title": (a.get("title","") or "").replace("\n"," ").strip(),
        "theme": a.get("theme",""),
        "sub_theme": a.get("sub_theme",""),
        "stakeholders": join(a.get("stakeholders",[]), sep="; "),
        "problem_statements": join(a.get("problem_statements", [])),
        "questions_users_ask": join(a.get("questions_users_ask", [])),
        "proposed_solutions": join(a.get("proposed_solutions", [])),
        "useful_links": join(a.get("useful_links", []), sep=" "),
        "sentiment": a.get("sentiment",""),
        "severity_1to5": a.get("severity_1to5",""),
        "dedupe_key": a.get("dedupe_key",""),
    }


def _write_per_post_csv(path, analyses: list[dict]):
    import csv, json
    rows = [_flatten_for_row(a) for a in analyses]
    if not rows:
        # still create a header-only file
        rows = [ {k:"" for k in [
            "post_id","subreddit","title","sentiment","severity_1to5","target_audience",
            "primary_topic_tags","problem_statements","questions_users_ask",
            "proposed_solutions","useful_links","dedupe_key"
        ]}]
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def _write_per_post_md(path, analyses: list[dict], top_n: int = 50):
    rows = [_flatten_for_row(a) for a in analyses][:max(1, int(top_n))]
    headers = ["post_id","subreddit","title","sentiment","severity_1to5",
               "target_audience","primary_topic_tags","problem_statements",
               "questions_users_ask","proposed_solutions","useful_links","dedupe_key"]
    def esc(x):  # keep MD readable
        return str(x).replace("|","\\|")
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"]*len(headers)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(esc(r.get(h,"")) for h in headers) + " |\n")


def run_openai_analysis(args, all_rows, outdir, prefix):
    # batch
    bs = max(1, int(args.batch_size))
    analyses = []
    for i in range(0, len(all_rows), bs):
        batch = all_rows[i:i+bs]
        try:
            results = analyze_batch_with_openai(args, batch)
            analyses.extend(results)
        except Exception as e:
            print(f"[error] OpenAI analysis failed for batch {i}-{i+bs}: {e}", file=sys.stderr)

    # write raw analyses
    raw_path = outdir / f"{prefix}_analysis.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in analyses:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # quick aggregation by dedupe_key
    buckets = {}
    for r in analyses:
        key = (r.get("dedupe_key") or "").lower().strip()
        if not key:
            continue
        buckets.setdefault(key, {"count":0, "examples":[], "severity_sum":0,
                                 "audiences":set(), "links":set(), "tags":set()})
        b = buckets[key]
        b["count"] += 1
        b["severity_sum"] += int(r.get("severity_1to5",3))
        b["audiences"].add(r.get("target_audience",""))
        b["tags"].update(set(r.get("primary_topic_tags",[])))
        for url in r.get("useful_links",[]):
            b["links"].add(url)
        if len(b["examples"]) < 5:
            b["examples"].append({"subreddit": r.get("subreddit"), "title": r.get("title")})

    # save an easy CSV for quick scanning
    import csv
    agg_path = outdir / f"{prefix}_issues_aggregated.csv"
    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["issue_key","posts","avg_severity","audiences",
                    "top_tags","example_titles","links_count"])
        for k, b in sorted(buckets.items(), key=lambda kv: kv[1]["count"], reverse=True):
            avg_sev = round(b["severity_sum"]/b["count"], 2)
            w.writerow([
                k, b["count"], avg_sev,
                "; ".join(sorted(a for a in b["audiences"] if a)),
                "; ".join(sorted(b["tags"]))[:300],
                "; ".join(e["title"] for e in b["examples"])[:300],
                len(b["links"]),
            ])

    # (optional) create a short MD summary via the model
    try:
        client = _oai_client()
        model = _model_name(args)
        bullet = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"You condense EdTech issues into crisp bullets for executives."},
                {"role":"user","content":f"Summarize the top 10 recurring issues from this JSON (each 1 line with a concrete action idea):\n{json.dumps(buckets)[:60_000]}"}
            ]
        ).choices[0].message.content
        (outdir / f"{prefix}_summary.md").write_text(bullet or "", encoding="utf-8")
    except Exception as e:
        print(f"[warn] summary generation failed: {e}", file=sys.stderr)

    # 👇 ADD THIS BLOCK
    per_post_csv = outdir / f"{prefix}_analysis_per_post.csv"
    _write_per_post_csv(per_post_csv, analyses)

    if getattr(args, "table", "csv") in ("md", "both"):
        per_post_md = outdir / f"{prefix}_analysis_per_post.md"
        _write_per_post_md(per_post_md, analyses, top_n=getattr(args, "md_top", 50))

    print("\nOpenAI analysis saved:")
    print(f"- {raw_path}")
    print(f"- {agg_path}")
    print(f"- {outdir / f'{prefix}_summary.md'}")
    print(f"- {per_post_csv}")
    if getattr(args, "table","csv") in ("md","both"):
        print(f"- {per_post_md}")
