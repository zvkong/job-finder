# -*- coding: utf-8 -*-
from __future__ import annotations

import random
import time
import urllib.parse
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

import google.generativeai as genai
import requests
import streamlit as st
from atproto import Client
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from googlesearch import search


# ============================================================
# App Config
# ============================================================

APP_TITLE = "🎯 Postdoc Hunter (Statistics)"
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

DEFAULT_PRIORITY_PORTALS = [
    {"name": "UW Statistics", "url": "https://stat.uw.edu/news-resources/jobs"},
    {
        "name": "NYU Careers",
        "url": "https://www.nyu.edu/about/careers-at-nyu/faculty-and-researchers.html?keyword=postdoc",
    },
    {"name": "UT Austin", "url": "https://faculty.utexas.edu/career?q=postdoc&units=all"},
    {"name": "Harvard", "url": "https://academicpositions.harvard.edu/postings/search"},
]

DEFAULT_SEARCH_SITES = [
    "apply.interfolio.com",
    "linkedin.com/jobs",
    "mathjobs.org",
    "academicjobsonline.org",
]


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


def estimate_prompt_size(text: str) -> dict:
    char_count = len(text)
    line_count = text.count("\n") + 1 if text else 0
    word_count = len(text.split()) if text else 0
    approx_tokens = max(1, int(char_count / 4)) if text else 0

    return {
        "characters": char_count,
        "lines": line_count,
        "words": word_count,
        "approx_tokens": approx_tokens,
    }


def make_bluesky_post_url(handle: str, uri: str) -> str:
    if not handle or not uri:
        return ""
    parts = uri.split("/")
    if len(parts) >= 5:
        rkey = parts[-1]
        return f"https://bsky.app/profile/{handle}/post/{rkey}"
    return ""


# ============================================================
# Gemini / Bluesky Configuration
# ============================================================

def load_gemini_api_key() -> str:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return ""


def load_bluesky_credentials() -> tuple[str, str]:
    try:
        handle = st.secrets["BLUESKY_HANDLE"]
        app_password = st.secrets["BLUESKY_APP_PASSWORD"]
        return handle, app_password
    except Exception:
        return "", ""


def configure_gemini(api_key: str):
    genai.configure(api_key=api_key, transport="rest")
    return genai.GenerativeModel(GEMINI_MODEL_NAME)


def configure_bluesky_client() -> Client | None:
    handle, app_password = load_bluesky_credentials()
    if not handle or not app_password:
        return None

    try:
        client = Client()
        client.login(handle, app_password)
        return client
    except Exception as e:
        st.error("Failed to authenticate with Bluesky.")
        st.error(str(e))
        return None


# ============================================================
# Scrapers: Portals / Direct Boards
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


def fetch_other_priority_universities(priority_portals: List[Tuple[str, str]]) -> List[JobEntry]:
    jobs: List[JobEntry] = []

    for uni_name, url in priority_portals:
        if not uni_name or not url:
            continue

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


# ============================================================
# Sidebar Controls
# ============================================================

def render_agent_settings():
    with st.sidebar.expander("Agent Settings", expanded=True):
        search_days = st.radio(
            "Search Timeframe (Days)",
            options=[1, 3, 5],
            index=1,
            horizontal=True,
            key="search_days_radio",
        )

        editable_strategy = st.text_area(
            "EXTRACTION STRATEGY",
            value="""1. Extract ALL legitimate Postdoc/Research Fellow jobs in Statistics/Biostatistics from the raw data.
2. Highlight and prioritize jobs mentioning Small Area Estimation (SAE), Spatial models, or Bayesian methodology.
3. Prioritize targeted universities: UMich, UW, NYU, UT Austin, and Harvard.
4. If there are fewer than 10 perfect matches, generously RELAX the research focus criteria to include other high-quality Statistics/Data Science postdoc roles. Include as many valid jobs as possible.""",
            height=300,
            key="editable_strategy_textarea",
        )
    return search_days, editable_strategy


def render_priority_portal_controls() -> List[Tuple[str, str]]:
    with st.sidebar.expander("Priority Portal Controls", expanded=False):
        if st.button("Add Priority Portal", key="add_priority_portal_btn"):
            st.session_state.priority_portals.append({"name": "", "url": ""})

        portals = []
        remove_index = None

        for i, portal in enumerate(st.session_state.priority_portals):
            st.markdown(f"**Priority Portal {i + 1}**")

            name = st.text_input(
                f"Priority Name {i + 1}",
                value=portal["name"],
                key=f"priority_name_{i}",
            )
            url = st.text_input(
                f"Priority URL {i + 1}",
                value=portal["url"],
                key=f"priority_url_{i}",
            )

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Remove", key=f"remove_priority_{i}"):
                    remove_index = i
            with col2:
                st.write("")

            st.session_state.priority_portals[i]["name"] = name.strip()
            st.session_state.priority_portals[i]["url"] = url.strip()

            if name.strip() and url.strip():
                portals.append((name.strip(), url.strip()))

        if remove_index is not None:
            st.session_state.priority_portals.pop(remove_index)
            st.rerun()

    return portals


def render_search_engine_controls():
    with st.sidebar.expander("Search Engine Query Controls", expanded=False):
        base_keyword = st.text_input(
            "Base keyword",
            value="postdoc statistics",
            key="search_base_keyword",
        )

        region_keyword = st.text_input(
            "Region keyword",
            value="USA",
            key="search_region_keyword",
        )

        extra_keyword = st.text_input(
            "Extra keyword",
            value="",
            help="例如 Bayesian OR spatial OR biostatistics",
            key="search_extra_keyword",
        )

        st.markdown("**Search Sites**")
        if st.button("Add Search Site", key="add_search_site_btn"):
            st.session_state.search_sites.append("")

        cleaned_sites = []
        remove_index = None

        for i, site in enumerate(st.session_state.search_sites):
            col1, col2 = st.columns([5, 1])

            with col1:
                new_site = st.text_input(
                    f"Site {i + 1}",
                    value=site,
                    key=f"search_site_{i}",
                )

            with col2:
                st.write("")
                st.write("")
                if st.button("X", key=f"remove_site_{i}"):
                    remove_index = i

            st.session_state.search_sites[i] = new_site.strip()
            if new_site.strip():
                cleaned_sites.append(new_site.strip())

        if remove_index is not None:
            st.session_state.search_sites.pop(remove_index)
            st.rerun()

        num_results = st.number_input(
            "Results per query",
            min_value=1,
            max_value=25,
            value=5,
            step=1,
            key="search_num_results",
        )

    return {
        "base_keyword": base_keyword.strip(),
        "region_keyword": region_keyword.strip(),
        "extra_keyword": extra_keyword.strip(),
        "sites": cleaned_sites,
        "num_results": int(num_results),
    }


def render_bluesky_controls():
    with st.sidebar.expander("Bluesky Keywords", expanded=False):
        bluesky_base = st.text_input(
            "Bluesky base keyword",
            value="postdoc",
            key="bluesky_base_keyword",
        )
        bluesky_region = st.text_input(
            "Bluesky region keyword",
            value="",
            key="bluesky_region_keyword",
        )
        bluesky_extra = st.text_input(
            "Bluesky extra keyword",
            value="lang:en",
            key="bluesky_extra_keyword",
        )
        bluesky_num_results = st.number_input(
            "Bluesky results",
            min_value=1,
            max_value=25,
            value=5,
            step=1,
            key="bluesky_num_results",
        )

    return {
        "base_keyword": bluesky_base.strip(),
        "region_keyword": bluesky_region.strip(),
        "extra_keyword": bluesky_extra.strip(),
        "num_results": int(bluesky_num_results),
    }


def render_buttons():
    btn_realtime = st.button(
        "⚡ Start Real-Time Web Scan (Direct Scraper + DDG Interfolio)",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.running,
    )

    with st.expander("🛠️ Search Engine Tools", expanded=True):
        st.warning("Google may fail due to anti-bot / IP rate limits. DDG and Bluesky are usually more stable.")
        btn_google = st.button(
            "☕ Run Google Precision Scan",
            use_container_width=True,
            disabled=st.session_state.running,
        )
        btn_ddg = st.button(
            "🦆 Run DDG Precision Scan",
            use_container_width=True,
            disabled=st.session_state.running,
        )
        btn_bluesky = st.button(
            "🦋 Run Bluesky Precision Scan",
            use_container_width=True,
            disabled=st.session_state.running,
        )

    if btn_realtime:
        st.session_state.should_run_scan = True
        st.session_state.selected_engine = "realtime"

    if btn_google:
        st.session_state.should_run_scan = True
        st.session_state.selected_engine = "google"

    if btn_ddg:
        st.session_state.should_run_scan = True
        st.session_state.selected_engine = "ddg"

    if btn_bluesky:
        st.session_state.should_run_scan = True
        st.session_state.selected_engine = "bluesky"


# ============================================================
# Query Builders
# ============================================================

def build_search_queries(config: dict) -> List[str]:
    base = config["base_keyword"]
    region = config["region_keyword"]
    extra = config["extra_keyword"]
    sites = [s for s in config["sites"] if s]

    tail = " ".join([x for x in [base, region, extra] if x]).strip()

    queries = []
    for site in sites:
        query = f"site:{site} {tail}".strip()
        if query:
            queries.append(query)
    return queries


def build_bluesky_queries(config: dict) -> List[str]:
    base = config["base_keyword"].strip()
    region = config["region_keyword"].strip()
    extra = config["extra_keyword"].strip()

    queries: List[str] = []

    if base:
        queries.append(base)

    if base and region:
        queries.append(f"{base} {region}".strip())

    if base and extra:
        queries.append(f"{base} {extra}".strip())

    if base and region and extra:
        queries.append(f"{base} {region} {extra}".strip())

    if base and "postdoc" not in base.lower():
        queries.append(f"{base} postdoc".strip())

    if region:
        queries.append(f"postdoc {region}".strip())

    if extra:
        queries.append(f"postdoc {extra}".strip())

    # Hard-coded broad fallback queries
    queries.append("postdoc")
    queries.append("hiring postdoc")
    queries.append("statistics postdoc")
    queries.append("biostatistics postdoc")
    queries.append("data science postdoc")
    queries.append("research fellow")
    queries.append("academic hiring")

    seen = set()
    result = []
    for q in queries:
        q2 = " ".join(q.split())
        if q2 and q2 not in seen:
            seen.add(q2)
            result.append(q2)

    return result


# ============================================================
# Search Engines
# ============================================================

def fetch_google_jobs(queries: List[str], num_results: int = 5) -> List[JobEntry]:
    jobs: List[JobEntry] = []

    for query in queries:
        st.write(f"Running Google query: {query}")
        try:
            raw_results = list(
                search(query, num_results=num_results, sleep_interval=4.0, advanced=True)
            )
            st.write(f"Raw results returned: {len(raw_results)}")

            for result in raw_results:
                jobs.append(
                    JobEntry(
                        title=clean_text(getattr(result, "title", "")),
                        href=clean_text(getattr(result, "url", "")),
                        body=clean_text(getattr(result, "description", "")),
                        source="Google Precision Search",
                    )
                )
            time.sleep(random.uniform(3, 5))

        except Exception as e:
            err = str(e)
            if "google.com/sorry" in err or "429" in err:
                st.warning(f"Google anti-bot triggered for query: {query}")
                st.error(err)
            else:
                st.error(f"Google query failed: {query}")
                st.error(err)

    return jobs


def fetch_ddg_jobs(queries: List[str], num_results: int = 5) -> List[JobEntry]:
    jobs: List[JobEntry] = []

    try:
        with DDGS() as ddgs:
            for query in queries:
                st.write(f"Running DDG query: {query}")
                try:
                    raw_results = list(ddgs.text(query, max_results=num_results))
                    st.write(f"Raw results returned: {len(raw_results)}")

                    for result in raw_results:
                        jobs.append(
                            JobEntry(
                                title=clean_text(result.get("title", "")),
                                href=clean_text(result.get("href", "")),
                                body=clean_text(result.get("body", "")),
                                source="DDG Precision Search",
                            )
                        )
                    time.sleep(random.uniform(1, 2))

                except Exception as e:
                    st.error(f"DDG query failed: {query}")
                    st.error(str(e))
    except Exception as e:
        st.error("Failed to initialize DuckDuckGo search client.")
        st.error(str(e))

    return jobs


def fetch_bluesky_jobs(queries: List[str], num_results: int = 5) -> List[JobEntry]:
    jobs: List[JobEntry] = []

    client = configure_bluesky_client()
    if client is None:
        st.error("Bluesky client is not available. Check BLUESKY_HANDLE and BLUESKY_APP_PASSWORD.")
        return jobs

    for query in queries:
        st.write(f"Running Bluesky query: {query}")

        try:
            response = client.app.bsky.feed.search_posts(
                params={
                    "q": query,
                    "limit": num_results,
                }
            )

            posts = getattr(response, "posts", []) or []
            st.write(f"Raw results returned: {len(posts)}")

            for post in posts:
                author = getattr(post, "author", None)
                record = getattr(post, "record", None)

                handle = clean_text(getattr(author, "handle", "") if author else "")
                uri = clean_text(getattr(post, "uri", ""))

                indexed_at = ""
                if hasattr(post, "indexed_at"):
                    indexed_at = clean_text(getattr(post, "indexed_at", ""))
                elif hasattr(post, "indexedAt"):
                    indexed_at = clean_text(getattr(post, "indexedAt", ""))

                text = ""
                if record is not None and hasattr(record, "text"):
                    text = clean_text(getattr(record, "text", ""))

                href = make_bluesky_post_url(handle, uri)

                body_parts = []
                if handle:
                    body_parts.append(f"Author: {handle}")
                if indexed_at:
                    body_parts.append(f"IndexedAt: {indexed_at}")
                if text:
                    body_parts.append(f"Post: {text[:300]}")

                jobs.append(
                    JobEntry(
                        title=text[:180] if text else "Bluesky Post",
                        href=href,
                        body=" | ".join(body_parts),
                        source="Bluesky Search",
                    )
                )

            time.sleep(random.uniform(1, 2))

        except Exception as e:
            st.error(f"Bluesky query failed: {query}")
            st.error(str(e))

    return deduplicate_jobs(jobs)


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
    if "priority_portals" not in st.session_state:
        st.session_state.priority_portals = [item.copy() for item in DEFAULT_PRIORITY_PORTALS]
    if "search_sites" not in st.session_state:
        st.session_state.search_sites = DEFAULT_SEARCH_SITES.copy()
    if "should_run_scan" not in st.session_state:
        st.session_state.should_run_scan = False
    if "selected_engine" not in st.session_state:
        st.session_state.selected_engine = None


# ============================================================
# Workflow Helpers
# ============================================================

def run_portal_scan(actual_days: int, priority_portals: List[Tuple[str, str]]):
    p1 = fetch_umich_jobs(actual_days)
    p2 = fetch_other_priority_universities(priority_portals)
    return p1, p2


def build_final_job_pool(
    portal_jobs_a: List[JobEntry],
    portal_jobs_b: List[JobEntry],
    engine_jobs: List[JobEntry],
) -> List[JobEntry]:
    raw_all = portal_jobs_a + portal_jobs_b + engine_jobs
    return deduplicate_jobs(raw_all)


# ============================================================
# Main
# ============================================================

def main():
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(APP_TITLE)

    initialize_session_state()

    api_key = load_gemini_api_key()
    if not api_key:
        st.error("Gemini API key not found. Please set GEMINI_API_KEY in .streamlit/secrets.toml")
        st.stop()

    search_days, editable_strategy = render_agent_settings()
    priority_portals = render_priority_portal_controls()
    search_engine_config = render_search_engine_controls()
    bluesky_config = render_bluesky_controls()

    search_queries = build_search_queries(search_engine_config)
    bluesky_queries = build_bluesky_queries(bluesky_config)

    render_buttons()

    if st.button("Test Bluesky Only", disabled=st.session_state.running, key="btn_test_bluesky"):
        client = configure_bluesky_client()
        if client is not None:
            try:
                response = client.app.bsky.feed.search_posts(
                    params={"q": "postdoc", "limit": 3}
                )
                posts = getattr(response, "posts", []) or []
                st.success(f"Bluesky auth OK. Retrieved {len(posts)} posts.")
            except Exception as e:
                st.error(f"Bluesky test failed: {e}")

    if not st.session_state.should_run_scan:
        return

    if st.session_state.running:
        st.warning("A scan is already running.")
        st.stop()

    st.session_state.running = True

    try:
        st.markdown("### 🛠️ System Diagnostic Monitor")
        actual_days = search_days

        with st.spinner("Searching Priority Portals..."):
            p1, p2 = run_portal_scan(actual_days, priority_portals)
            st.write(f"✔️ Portals: {len(p1) + len(p2)} items.")

        if st.session_state.selected_engine == "google":
            with st.spinner("Executing Google Search..."):
                engine_jobs = fetch_google_jobs(
                    queries=search_queries,
                    num_results=search_engine_config["num_results"],
                )
                st.write(f"✔️ Google: {len(engine_jobs)} items.")

        elif st.session_state.selected_engine == "ddg":
            with st.spinner("Executing DuckDuckGo Search..."):
                engine_jobs = fetch_ddg_jobs(
                    queries=search_queries,
                    num_results=search_engine_config["num_results"],
                )
                st.write(f"✔️ DDG: {len(engine_jobs)} items.")

        elif st.session_state.selected_engine == "bluesky":
            with st.spinner("Executing Bluesky Search..."):
                engine_jobs = fetch_bluesky_jobs(
                    queries=bluesky_queries,
                    num_results=bluesky_config["num_results"],
                )
                st.write(f"✔️ Bluesky: {len(engine_jobs)} items.")

        else:
            with st.spinner("Direct Scraping MathJobs, Nature & Interfolio..."):
                engine_jobs = fetch_direct_job_boards()
                st.write(f"✔️ Direct Scrapers: {len(engine_jobs)} items.")

        all_jobs = build_final_job_pool(p1, p2, engine_jobs)
        st.write(f"**Total unique jobs found:** {len(all_jobs)}")

        if not all_jobs:
            st.error("No results found. Please check connection, site structure, or search engine availability.")
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

        with st.expander("🔍 Debug Search Queries"):
            st.json(search_queries)

        with st.expander("🔍 Debug Bluesky Query"):
            st.json(bluesky_queries)

        with st.expander("🔍 Debug Priority Portals"):
            st.json([{"name": name, "url": url} for name, url in priority_portals])

    finally:
        st.session_state.running = False
        st.session_state.should_run_scan = False
        st.session_state.selected_engine = None


if __name__ == "__main__":
    main()