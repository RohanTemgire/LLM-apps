import logging
import queue
import threading
import time
import streamlit as st
import pandas as pd
from deal_agent_framework import DealAgentFramework
from log_utils import reformat
# import plotly.graph_objects as go  # Commented out: vector DB visualization disabled for now
from dotenv import load_dotenv

load_dotenv(override=True)


class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


def setup_logging(log_queue):
    handler = QueueHandler(log_queue)
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    # Avoid adding duplicate handlers on reruns
    for h in logger.handlers[:]:
        if isinstance(h, QueueHandler):
            logger.removeHandler(h)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def html_for(log_data):
    output = "<br>".join(log_data[-18:])
    return f"""
    <div style="height: 400px; overflow-y: auto; border: 1px solid #444; background-color: #222229;
                padding: 10px; border-radius: 8px; font-family: monospace; font-size: 13px; color: #ccc;">
    {output}
    </div>
    """


def get_agent_framework():
    """Get or create the DealAgentFramework singleton in session state."""
    if "agent_framework" not in st.session_state:
        st.session_state.agent_framework = DealAgentFramework()
    return st.session_state.agent_framework


def table_for(opps):
    """Convert opportunities list to a DataFrame."""
    rows = [
        {
            "Deals found so far": opp.deal.product_description,
            "Price": f"${opp.deal.price:.2f}",
            "Estimate": f"${opp.estimate:.2f}",
            "Discount": f"${opp.discount:.2f}",
            "URL": opp.deal.url,
        }
        for opp in opps
    ]
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Deals found so far", "Price", "Estimate", "Discount", "URL"]
    )


# ── Vector DB Visualization (commented out for faster loading) ──
# @st.cache_data(show_spinner="Computing vector visualization (t-SNE)...")
# def get_plot():
#     """Generate the 3D scatter plot of the vector DB (cached after first computation)."""
#     try:
#         documents, vectors, colors = DealAgentFramework.get_plot_data(max_datapoints=200)
#         fig = go.Figure(
#             data=[
#                 go.Scatter3d(
#                     x=vectors[:, 0], y=vectors[:, 1], z=vectors[:, 2],
#                     mode="markers",
#                     marker=dict(size=2, color=colors, opacity=0.7),
#                 )
#             ]
#         )
#         fig.update_layout(
#             scene=dict(
#                 xaxis_title="x", yaxis_title="y", zaxis_title="z",
#                 aspectmode="manual",
#                 aspectratio=dict(x=2.2, y=2.2, z=1),
#                 camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
#             ),
#             height=400, margin=dict(r=5, b=1, l=5, t=2),
#         )
#         return fig
#     except Exception:
#         fig = go.Figure()
#         fig.update_layout(title="Could not load vector DB visualization", height=400)
#         return fig


def run_agent():
    """Run the agent framework synchronously with live log streaming via placeholders."""
    st.session_state["is_running"] = True
    framework = get_agent_framework()
    log_queue = queue.Queue()
    result_queue = queue.Queue()
    setup_logging(log_queue)

    log_data = st.session_state.get("log_data", [])

    def worker():
        try:
            new_opportunities = framework.run()
            result_queue.put(("ok", new_opportunities))
        except Exception as e:
            result_queue.put(("error", e))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    log_placeholder = st.session_state["log_placeholder"]
    table_placeholder = st.session_state["table_placeholder"]

    while thread.is_alive() or not result_queue.empty() or not log_queue.empty():
        # Drain all available log messages first
        got_log = False
        while True:
            try:
                message = log_queue.get_nowait()
                log_data.append(reformat(message))
                got_log = True
            except queue.Empty:
                break

        if got_log:
            log_placeholder.markdown(html_for(log_data), unsafe_allow_html=True)
            table_placeholder.dataframe(
                table_for(framework.memory), width="stretch", hide_index=True
            )

        # Check for final result
        try:
            status, payload = result_queue.get_nowait()
            # Drain any remaining logs
            while True:
                try:
                    message = log_queue.get_nowait()
                    log_data.append(reformat(message))
                except queue.Empty:
                    break
            log_placeholder.markdown(html_for(log_data), unsafe_allow_html=True)

            if status == "error":
                st.error(f"Agent error: {payload}")
            else:
                table_placeholder.dataframe(
                    table_for(payload), width="stretch", hide_index=True
                )
            break
        except queue.Empty:
            time.sleep(0.1)

    st.session_state["log_data"] = log_data
    st.session_state["is_running"] = False


# ──────────────────────────────────────────────
# Streamlit Page Config & Layout
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="The Price is Right",
    page_icon="💰",
    layout="wide",
)

# ── Initialise session state defaults ──
if "log_data" not in st.session_state:
    st.session_state["log_data"] = []
if "has_auto_run" not in st.session_state:
    st.session_state["has_auto_run"] = False
if "is_running" not in st.session_state:
    st.session_state["is_running"] = False

# ── Custom CSS ──
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0;
        color: #ffffff;
    }
    .sub-title {
        text-align: center;
        font-size: 0.95rem;
        color: #aaaaaa;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ──
st.markdown(
    '<p class="main-title">💰 The Price is Right '
    '<span style="font-weight:400;font-size:1.1rem;">— Autonomous Agent Framework that hunts for deals</span></p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">A proprietary fine-tuned LLM deployed on Modal and a RAG pipeline '
    "with a frontier model collaborate to send push notifications with great online deals.</p>",
    unsafe_allow_html=True,
)

# ── Run / Re-run Button (placed near the top so it's always visible) ──
run_clicked = st.button("🚀 Run Agent Scan", use_container_width=True, type="primary")

# ── Deals Table ──
st.subheader("Deals")
table_placeholder = st.empty()
st.session_state["table_placeholder"] = table_placeholder

framework = get_agent_framework()
table_placeholder.dataframe(
    table_for(framework.memory), width="stretch", hide_index=True
)

# ── Agent Logs (full width) ──
st.subheader("Agent Logs")
log_placeholder = st.empty()
st.session_state["log_placeholder"] = log_placeholder
existing_logs = st.session_state.get("log_data", [])
if existing_logs:
    log_placeholder.markdown(html_for(existing_logs), unsafe_allow_html=True)

# ── Vector DB Visualization (commented out for faster loading) ──
# col_plot:
#     st.subheader("Vector DB Visualization")
#     plot_placeholder = st.empty()
#     plot_placeholder.plotly_chart(get_plot(), use_container_width=True)

# ── Trigger agent run (button click OR auto-run on first load) ──
should_run = run_clicked or (not st.session_state["has_auto_run"])

if should_run and not st.session_state["is_running"]:
    st.session_state["has_auto_run"] = True
    run_agent()
