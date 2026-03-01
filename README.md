# job-finder
this is a job finder with agent 
Recommend conda to build the enviourment and install all packages
## conda create -n job_agent python=3.10 -y
## conda activate job_agent
## pip install streamlit google-generativeai duckduckgo-search response googlesearch-python

For safety, set a .streamlit folder and build a new secrets.toml file
in that file
GEMINI_API_KEY = "Your API-KEY"