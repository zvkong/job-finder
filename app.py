# -*- coding: utf-8 -*-
from __future__ import annotations

import random
import time
import urllib.parse
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable, List, Optional

import google.generativeai as genai
import requests
import streamlit as st
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from googlesearch import search


# ============================================================
# App Config
# ============================================================

APP_TITLE = "🎯 Postdoc Hunter (Direct Scraper Edition)"
PAGE_TITLE = "Postdoc Hunter Pro"

REQUEST_TIMEOUT = 15
MAX_JOBS_SENT_TO_GEMINI = 30
MAX_TITLE_LEN = 220
MAX_BODY_LEN = 350
GEMINI_MODEL_NAME = "gemini-2.5-flash"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


# ============================================================
# Data Model
# ============================================================

@dataclass
class JobEntry:
    title: str
    href: str
    body: str
    source: str

    def normalized(self) -> "JobEntry":
        return JobEntry(
            title=clean_text(self.title)[:MAX_TITLE_LEN],
            href=clean_text(self.href),
            body=clean_text(self.body)[:MAX_BODY_LEN],
            source=clean_text(self.source),
        )


# ============================================================
# Utilities
# ============================================================

def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(str(text).split())


def safe_get(
    url: str,
    params: Optional[dict] = None,
    timeout: int = REQUEST_TIMEOUT,
) -> Optional[requests.Response]:
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        return response
    except Exception:
        return None


def absolute_url(base_url: str, href: str) -> str:
    return urllib.parse.urljoin(base_url, href)


def truncate_jobs_for_prompt(jobs: List[JobEntry], max_jobs: int) -> List[JobEntry]:
    return jobs[:max_jobs]


def deduplicate_jobs(jobs: Iterable[JobEntry]) -> List[JobEntry]:
    """
    去重策略保持与之前一致的精神：
    - 优先按 href 去重
    - href 为空时，按 (title, source) 去重
    """
    seen_links = set()
    seen_fallback = set()
    deduped: List[JobEntry] = []

    for raw_job in jobs:
        job = raw_job.normalized()

        if job.href:
            if job.href in seen_links:
                continue
            seen_links.add(job.href)
            deduped.append(job)
        else:
            fallback_key = (job.title.lower(), job.source.lower())
            if fallback_key in seen_fallback:
                continue
            seen_fallback.add(fallback_key)
            deduped.append(job)

    return deduped


def jobs_to_prompt_text(jobs: List[JobEntry]) -> str:
    return "\n\n".join(
        [
            f"Source: {job.source}\n"
            f"Title: {job.title}\n"
            f"Link: {job.href}\n"
            f"Body: {job.body}"
            for job in jobs
        ]
    )


def jobs_to_jsonable(jobs: List[JobEntry]) -> List[dict]:
    return [asdict(job) for job in jobs]


# ============================================================
# Gemini Configuration
# ============================================================

def load_gemini_api_key() -> str:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return ""


def configure_gemini(api_key: str):
    genai.configure(api_key=api_key, transport="rest")
    return genai.GenerativeModel(GEMINI_MODEL_NAME)


# ============================================================
# Scrapers
# ============================================================

def fetch_umich_jobs(days_to_search: int) -> List[JobEntry]:
    jobs: List[JobEntry] = []
    url = "https://careers.umich.edu/search-jobs"
    params = {"keyword": "postdoc"}

    response = safe_get(url, params=params)
    if not response:
        return jobs

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", class_="cols-5")

        if not table or not table.tbody:
            return jobs

        today = datetime.now().date()

        for row in table.tbody.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) < 5:
                continue

            date_posted = clean_text(cols[0].get_text())
            title_tag = cols[1].find("a")
            dept_text = clean_text(cols[3].get_text()) if len(cols) > 3 else ""

            try:
                job_date = datetime.strptime(date_posted, "%m/%d/%Y").date()
                if (today - job_date).days > days_to_search:
                    continue
            except Exception:
                # 保持原逻辑：日期解析失败时不强行过滤
                pass

            title = clean_text(title_tag.get_text()) if title_tag else "Unknown"
            link = ""
            if title_tag and title_tag.has_attr("href"):
                link = absolute_url("https://careers.umich.edu", title_tag["href"])

            final_title = f"{title} ({dept_text})" if dept_text else title

            jobs.append(
                JobEntry(
                    title=final_title,
                    href=link,
                    body=f"Posted: {date_posted}",
                    source="UMich Portal",
                )
            )
    except Exception:
        pass

    return jobs


def fetch_other_priority_universities() -> List[JobEntry]:
    jobs: List[JobEntry] = []

    target_urls = [
        ("UW Statistics", "https://stat.uw.edu/news-resources/jobs"),
        ("NYU Careers", "https://www.nyu.edu/about/careers-at-nyu/faculty-and-researchers.html?keyword=postdoc"),
        ("UT Austin", "https://faculty.utexas.edu/career?q=postdoc&units=all"),
        ("Harvard", "https://academicpositions.harvard.edu/postings/search"),
    ]

    for uni_name, url in target_urls:
        response = safe_get(url)
        if not response:
            continue

        try:
            soup = BeautifulSoup(response.text, "html.parser")
            count = 0

            for a in soup.find_all("a", href=True):
                text = clean_text(a.get_text()).lower()
                if ("postdoc" in text or "fellow" in text) and len(text) > 12:
                    full_url = absolute_url(url, a["href"])
                    jobs.append(
                        JobEntry(
                            title=clean_text(a.get_text()),
                            href=full_url,
                            body=f"Source: {uni_name}",
                            source=f"{uni_name} Portal",
                        )
                    )
                    count += 1
                    if count >= 6:
                        break
        except Exception:
            pass

    return jobs


def fetch_mathjobs() -> List[JobEntry]:
    jobs: List[JobEntry] = []
    response = safe_get("https://www.mathjobs.org/jobs?joblist=0")
    if not response:
        return jobs

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        count = 0

        for a in soup.find_all("a", href=True):
            text = clean_text(a.get_text()).lower()
            href = a["href"]

            if "job" in href and ("postdoc" in text or "stat" in text or "research" in text):
                full_url = absolute_url("https://www.mathjobs.org", href)
                jobs.append(
                    JobEntry(
                        title=clean_text(a.get_text()),
                        href=full_url,
                        body="MathJobs Direct Entry",
                        source="MathJobs Direct",
                    )
                )
                count += 1
                if count >= 25:
                    break
    except Exception:
        pass

    return jobs


def fetch_nature_jobs() -> List[JobEntry]:
    jobs: List[JobEntry] = []
    url = "https://www.nature.com/naturecareers/jobs/science-jobs"

    response = safe_get(url, params={"q": "postdoc statistics"})
    if not response:
        return jobs

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        count = 0

        for a in soup.find_all("a", href=True):
            text = clean_text(a.get_text()).lower()
            href = a["href"]

            if "job" in href and ("postdoc" in text or "research" in text):
                full_url = absolute_url("https://www.nature.com", href)
                jobs.append(
                    JobEntry(
                        title=clean_text(a.get_text()),
                        href=full_url,
                        body="Nature Careers Direct Entry",
                        source="Nature Direct",
                    )
                )
                count += 1
                if count >= 20:
                    break
    except Exception:
        pass

    return jobs


def fetch_interfolio_via_ddg() -> List[JobEntry]:
    jobs: List[JobEntry] = []

    try:
        with DDGS() as ddgs:
            for result in ddgs.text("site:apply.interfolio.com postdoc statistics", max_results=10):
                jobs.append(
                    JobEntry(
                        title=clean_text(result.get("title", "Interfolio Position")),
                        href=clean_text(result.get("href", "")),
                        body=clean_text(result.get("body", "")),
                        source="Interfolio Scraper",
                    )
                )
    except Exception:
        pass

    return jobs


def fetch_direct_job_boards() -> List[JobEntry]:
    jobs: List[JobEntry] = []
    jobs.extend(fetch_mathjobs())
    jobs.extend(fetch_nature_jobs())
    jobs.extend(fetch_interfolio_via_ddg())
    return jobs


def fetch_google_jobs() -> List[JobEntry]:
    jobs: List[JobEntry] = []
    queries = [
        "site:apply.interfolio.com postdoc statistics USA",
        "site:linkedin.com/jobs postdoc statistics USA",
        "site:mathjobs.org postdoc statistics",
        "site:academicjobsonline.org postdoc statistics",
    ]

    for query in queries:
        try:
            for result in search(query, num_results=5, sleep_interval=4.0, advanced=True):
                jobs.append(
                    JobEntry(
                        title=clean_text(getattr(result, "title", "")),
                        href=clean_text(getattr(result, "url", "")),
                        body=clean_text(getattr(result, "description", "")),
                        source="Google Precision Search",
                    )
                )
            time.sleep(random.uniform(3, 5))
        except Exception:
            pass

    return jobs


# ============================================================
# Gemini Layer
# ============================================================

def build_system_prompt(
    editable_strategy: str,
    actual_days: int,
    today_str: str,
) -> str:
    hidden_prefix = """You are a rigorous factual academic career advisor.
ANTI-HALLUCINATION: Output ONLY jobs explicitly listed in the data. DO NOT invent URLs.

EXTRACTION STRATEGY:"""

    hidden_suffix = """
Format results exactly as:

### 🎯 Priority & Perfect Matches
- **[Job Title]** (Source: [Source])
- Link: [URL]
- Reason: [Why it matches]

### 💡 Worth Considering
- **[Job Title]** (Source: [Source])
- Link: [URL]
- Reason: [Alternative match explanation]
"""

    return (
        f"{hidden_prefix}\n"
        f"{editable_strategy}\n"
        f"{hidden_suffix}\n\n"
        f"[SYSTEM: Today is {today_str}. "
        f"Filter for last {actual_days} days where dates are explicitly available. "
        f"If no date is given, you may still include the job if it appears legitimate.]"
    )


def call_gemini_with_retry(model, content: str, max_retries: int = 5) -> str:
    last_error = None

    for attempt in range(max_retries):
        try:
            response = model.generate_content(content)
            return response.text
        except Exception as e:
            last_error = str(e)
            st.error(f"Gemini attempt {attempt + 1} failed: {last_error}")

            if "429" in last_error or "RESOURCE_EXHAUSTED" in last_error:
                sleep_time = min((2 ** attempt) + random.uniform(0.5, 1.5), 30)
                time.sleep(sleep_time)
                continue

            return f"Error: {last_error}"

    return f"### 🚨 Gemini quota or rate limit exceeded after retries.\n\nRaw error:\n{last_error}"

@st.cache_data(ttl=1800, show_spinner=False)
def cached_gemini_report(
    jobs_text: str,
    final_system_prompt: str,
    _cache_buster: str,
) -> str:
    api_key = load_gemini_api_key()
    if not api_key:
        return "Gemini API key not found."

    model = configure_gemini(api_key)

    if not jobs_text.strip():
        return "No data found."

    content = f"Raw Job Pool:\n\n{jobs_text}\n\nInstructions:\n{final_system_prompt}"
    return call_gemini_with_retry(model, content)


# ============================================================
# UI State
# ============================================================

def initialize_session_state():
    if "running" not in st.session_state:
        st.session_state.running = False
    if "gemini_call_count" not in st.session_state:
        st.session_state.gemini_call_count = 0
    if "last_prompt_chars" not in st.session_state:
        st.session_state.last_prompt_chars = 0

def estimate_prompt_size(text: str) -> dict:
    char_count = len(text)
    line_count = text.count("\n") + 1 if text else 0
    word_count = len(text.split()) if text else 0

    # 非精确估计，只用于诊断
    approx_tokens = max(1, int(char_count / 4)) if text else 0

    return {
        "characters": char_count,
        "lines": line_count,
        "words": word_count,
        "approx_tokens": approx_tokens,
    }
def render_sidebar():
    st.sidebar.header("Agent Settings")
    search_days = st.sidebar.radio(
        "Search Timeframe (Days)",
        options=[1, 3, 5],
        index=1,
        horizontal=True,
    )
    st.sidebar.markdown("---")

    editable_strategy = st.sidebar.text_area(
        "EXTRACTION STRATEGY",
        value="""1. Extract ALL legitimate Postdoc/Research Fellow jobs in Statistics/Biostatistics from the raw data.
2. Highlight and prioritize jobs mentioning Small Area Estimation (SAE), Spatial models, or Bayesian methodology.
3. Prioritize targeted universities: UMich, UW, NYU, UT Austin, and Harvard.
4. If there are fewer than 10 perfect matches, generously RELAX the research focus criteria to include other high-quality Statistics/Data Science postdoc roles. Include as many valid jobs as possible.""",
        height=300,
    )
    return search_days, editable_strategy


def render_buttons():
    btn_realtime = st.button(
        "⚡ Start Real-Time Web Scan (Direct Scraper + DDG)",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.running,
    )

    with st.expander("🛠️ Advanced / Daily Push Tools (Google Engine)"):
        st.warning("Warning: Use Google Engine sparingly to avoid IP block.")
        btn_google = st.button(
            "☕ Run Google Precision Scan (Slow)",
            use_container_width=True,
            disabled=st.session_state.running,
        )

    return btn_realtime, btn_google


# ============================================================
# Workflow
# ============================================================

def run_portal_scan(actual_days: int):
    p1 = fetch_umich_jobs(actual_days)
    p2 = fetch_other_priority_universities()
    return p1, p2


def run_engine_scan(use_google: bool):
    if use_google:
        return fetch_google_jobs()
    return fetch_direct_job_boards()


def build_final_job_pool(
    portal_jobs_a: List[JobEntry],
    portal_jobs_b: List[JobEntry],
    engine_jobs: List[JobEntry],
) -> List[JobEntry]:
    raw_all = portal_jobs_a + portal_jobs_b + engine_jobs
    return deduplicate_jobs(raw_all)


def main():
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(APP_TITLE)

    initialize_session_state()

    api_key = load_gemini_api_key()
    if not api_key:
        st.error("Gemini API key not found. Please set GEMINI_API_KEY in .streamlit/secrets.toml")
        st.stop()

    search_days, editable_strategy = render_sidebar()
    btn_realtime, btn_google = render_buttons()

    if not (btn_realtime or btn_google):
        return

    if st.session_state.running:
        st.warning("A scan is already running.")
        st.stop()

    st.session_state.running = True

    try:
        st.markdown("### 🛠️ System Diagnostic Monitor")
        actual_days = search_days

        with st.spinner("Searching Priority Portals..."):
            p1, p2 = run_portal_scan(actual_days)
            st.write(f"✔️ Portals: {len(p1) + len(p2)} items.")

        if btn_google:
            with st.spinner("Executing Google Search..."):
                engine_jobs = run_engine_scan(use_google=True)
                st.write(f"✔️ Google: {len(engine_jobs)} items.")
        else:
            with st.spinner("Direct Scraping MathJobs, Nature & Interfolio..."):
                engine_jobs = run_engine_scan(use_google=False)
                st.write(f"✔️ Direct Scrapers: {len(engine_jobs)} items.")

        all_jobs = build_final_job_pool(p1, p2, engine_jobs)
        st.write(f"**Total unique jobs found:** {len(all_jobs)}")

        if not all_jobs:
            st.error("No results found. Please check connection or site structure.")
            return

        prompt_jobs = truncate_jobs_for_prompt(all_jobs, MAX_JOBS_SENT_TO_GEMINI)
        jobs_text = jobs_to_prompt_text(prompt_jobs)
        today_str = datetime.now().strftime("%Y-%m-%d")
        final_system_prompt = build_system_prompt(editable_strategy, actual_days, today_str)

        full_prompt_preview = f"Raw Job Pool:\n\n{jobs_text}\n\nInstructions:\n{final_system_prompt}"
        prompt_stats = estimate_prompt_size(full_prompt_preview)
        st.session_state.last_prompt_chars = prompt_stats["characters"]

        with st.expander("📊 Gemini Request Diagnostics", expanded=True):
            st.write(f"Gemini call count in this session: {st.session_state.gemini_call_count}")
            st.write(f"Prompt characters: {prompt_stats['characters']}")
            st.write(f"Prompt words: {prompt_stats['words']}")
            st.write(f"Prompt lines: {prompt_stats['lines']}")
            st.write(f"Approx prompt tokens: {prompt_stats['approx_tokens']}")
            st.write(f"Jobs sent to Gemini: {len(prompt_jobs)} / {len(all_jobs)}")

        with st.spinner("Gemini is validating results..."):
            st.session_state.gemini_call_count += 1
            report = cached_gemini_report(
                jobs_text=jobs_text,
                final_system_prompt=final_system_prompt,
                _cache_buster=GEMINI_MODEL_NAME,
            )
        with st.expander("📈 Gemini Call Status", expanded=False):
            st.write(f"Total Gemini calls in this session: {st.session_state.gemini_call_count}")
            st.write(f"Last prompt characters: {st.session_state.last_prompt_chars}")
        st.success("Report Generated!")
        st.markdown("---")
        st.markdown(report)

        with st.expander("🔍 Debug Raw Data"):
            st.json(jobs_to_jsonable(all_jobs))

        with st.expander("🔍 Debug Prompt Sent to Gemini"):
            st.text(jobs_text)

    finally:
        st.session_state.running = False


if __name__ == "__main__":
    main()