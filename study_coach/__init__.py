"""AI Study Coach: a Streamlit study planner backed by SQLite and the Groq API.

The package is split by responsibility:

- ``analytics``: pure functions (streaks, summaries, time split, CSV export)
- ``db``: SQLite persistence for logged study sessions
- ``ai``: Groq chat-completion calls for plan generation and feedback
- ``ui``: CSS and small HTML helpers for the pink theme
- ``app``: the Streamlit page (``main``)
"""

__version__ = "0.2.0"
