"""
AI Study Coach entry point.

Run: streamlit run smart_study.py

This file stays at the repository root because Streamlit Community Cloud is configured
to run it. All application code lives in the ``study_coach`` package.
"""

import logging

from dotenv import load_dotenv

from study_coach.app import main

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    main()
