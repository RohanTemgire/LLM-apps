import streamlit as st
import pandas as pd
from collections import defaultdict
# import config, os
import time

# Ensure this module exists in your folder structure or path
from evaluations.eval import evaluate_all_retrieval, evaluate_all_answers

# --- Configuration ---
# os.environ['GEMINI_API_KEY'] = config.GEMINI_API_KEY

# Set page layout to wide to match the dashboard feel
st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")

# --- Constants ---
# Color coding thresholds - Retrieval
MRR_GREEN = 0.9
MRR_AMBER = 0.75
NDCG_GREEN = 0.9
NDCG_AMBER = 0.75
COVERAGE_GREEN = 90.0
COVERAGE_AMBER = 75.0

# Color coding thresholds - Answer (1-5 scale)
ANSWER_GREEN = 4.5
ANSWER_AMBER = 4.0

# --- Helper Functions ---

def get_color(value: float, metric_type: str) -> str:
    """Get color based on metric value and type."""
    if metric_type == "mrr":
        if value >= MRR_GREEN: return "green"
        elif value >= MRR_AMBER: return "orange"
        else: return "red"
    elif metric_type == "ndcg":
        if value >= NDCG_GREEN: return "green"
        elif value >= NDCG_AMBER: return "orange"
        else: return "red"
    elif metric_type == "coverage":
        if value >= COVERAGE_GREEN: return "green"
        elif value >= COVERAGE_AMBER: return "orange"
        else: return "red"
    elif metric_type in ["accuracy", "completeness", "relevance"]:
        if value >= ANSWER_GREEN: return "green"
        elif value >= ANSWER_AMBER: return "orange"
        else: return "red"
    return "black"

def format_metric_html(label: str, value: float, metric_type: str, is_percentage: bool = False, score_format: bool = False) -> str:
    """Format a metric with color coding (adapted for Streamlit markdown)."""
    color = get_color(value, metric_type)
    
    # Map color names to hex for better UI consistency if needed, 
    # or rely on standard CSS colors. Green/Orange/Red work fine.
    border_color = color
    
    if is_percentage:
        value_str = f"{value:.1f}%"
    elif score_format:
        value_str = f"{value:.2f}/5"
    else:
        value_str = f"{value:.4f}"
        
    return f"""
    <div style="margin: 10px 0; padding: 15px; background-color: #f9f9f9; border-radius: 8px; border-left: 5px solid {border_color};">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">{label}</div>
        <div style="font-size: 28px; font-weight: bold; color: {color};">{value_str}</div>
    </div>
    """

# --- Main App ---

def main():
    st.title("📊 RAG Evaluation Dashboard")
    st.markdown("Evaluate retrieval and answer quality for the Insurellm RAG system")
    
    st.divider()

    # ==========================================
    # 1. RETRIEVAL SECTION
    # ==========================================
    st.header("🔍 Retrieval Evaluation")
    
    # We use session state to keep data persistent if the app reruns
    if "retrieval_df" not in st.session_state:
        st.session_state.retrieval_df = None
    if "retrieval_html" not in st.session_state:
        st.session_state.retrieval_html = None

    if st.button("Run Retrieval Evaluation", type="primary", use_container_width=False):
        # Progress indicators
        progress_bar = st.progress(0, text="Starting evaluation...")
        
        total_mrr = 0.0
        total_ndcg = 0.0
        total_coverage = 0.0
        category_mrr = defaultdict(list)
        count = 0

        # Run the evaluation generator
        for test, result, prog_value in evaluate_all_retrieval():
            count += 1
            total_mrr += result.mrr
            total_ndcg += result.ndcg
            total_coverage += result.keyword_coverage
            category_mrr[test.category].append(result.mrr)
            
            # Update progress
            progress_bar.progress(prog_value, text=f"Evaluating test {count}...")

        # Calculate final averages
        avg_mrr = total_mrr / count
        avg_ndcg = total_ndcg / count
        avg_coverage = total_coverage / count

        # Create final summary HTML
        st.session_state.retrieval_html = f"""
        <div style="padding: 0;">
            {format_metric_html("Mean Reciprocal Rank (MRR)", avg_mrr, "mrr")}
            {format_metric_html("Normalized DCG (nDCG)", avg_ndcg, "ndcg")}
            {format_metric_html("Keyword Coverage", avg_coverage, "coverage", is_percentage=True)}
            <div style="margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;">
                <span style="font-size: 14px; color: #155724; font-weight: bold;">✓ Evaluation Complete: {count} tests</span>
            </div>
        </div>
        """

        # Prepare chart data
        category_data = []
        for category, mrr_scores in category_mrr.items():
            avg_cat_mrr = sum(mrr_scores) / len(mrr_scores)
            category_data.append({"Category": category, "Average MRR": avg_cat_mrr})
        
        st.session_state.retrieval_df = pd.DataFrame(category_data)
        
        # Clear progress bar
        progress_bar.empty()

    # Display Retrieval Results if they exist
    if st.session_state.retrieval_df is not None:
        col1, col2 = st.columns([1, 1.5]) # Adjust ratio as needed
        
        with col1:
            st.markdown(st.session_state.retrieval_html, unsafe_allow_html=True)
            
        with col2:
            st.markdown("##### Average MRR by Category")
            # Streamlit bar chart requires setting the index for labels
            chart_df = st.session_state.retrieval_df.set_index("Category")
            st.bar_chart(chart_df["Average MRR"], color="#4A90E2")

    st.divider()

    # ==========================================
    # 2. ANSWERING SECTION
    # ==========================================
    st.header("💬 Answer Evaluation")

    if "answer_df" not in st.session_state:
        st.session_state.answer_df = None
    if "answer_html" not in st.session_state:
        st.session_state.answer_html = None

    if st.button("Run Answer Evaluation", type="primary"):
        progress_bar = st.progress(0, text="Starting evaluation...")
        
        total_accuracy = 0.0
        total_completeness = 0.0
        total_relevance = 0.0
        category_accuracy = defaultdict(list)
        count = 0

        for test, result, prog_value in evaluate_all_answers():
            count += 1
            total_accuracy += result.accuracy
            total_completeness += result.completeness
            total_relevance += result.relevance
            category_accuracy[test.category].append(result.accuracy)
            
            progress_bar.progress(prog_value, text=f"Evaluating test {count}...")

        avg_accuracy = total_accuracy / count
        avg_completeness = total_completeness / count
        avg_relevance = total_relevance / count

        st.session_state.answer_html = f"""
        <div style="padding: 0;">
            {format_metric_html("Accuracy", avg_accuracy, "accuracy", score_format=True)}
            {format_metric_html("Completeness", avg_completeness, "completeness", score_format=True)}
            {format_metric_html("Relevance", avg_relevance, "relevance", score_format=True)}
            <div style="margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;">
                <span style="font-size: 14px; color: #155724; font-weight: bold;">✓ Evaluation Complete: {count} tests</span>
            </div>
        </div>
        """

        category_data = []
        for category, accuracy_scores in category_accuracy.items():
            avg_cat_accuracy = sum(accuracy_scores) / len(accuracy_scores)
            category_data.append({"Category": category, "Average Accuracy": avg_cat_accuracy})
        
        st.session_state.answer_df = pd.DataFrame(category_data)
        progress_bar.empty()

    # Display Answer Results
    if st.session_state.answer_df is not None:
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown(st.session_state.answer_html, unsafe_allow_html=True)
            
        with col2:
            st.markdown("##### Average Accuracy by Category")
            chart_df = st.session_state.answer_df.set_index("Category")
            st.bar_chart(chart_df["Average Accuracy"], color="#FF6B6B")

if __name__ == "__main__":
    main()