"""Streamlit components that display the Ollama health check."""

from __future__ import annotations

import streamlit as st

from paperpilot.config import Settings
from paperpilot.health import HealthReport, check_ollama_health, normalize_model_name

HEALTH_CACHE_SECONDS = 30


@st.cache_data(ttl=HEALTH_CACHE_SECONDS, show_spinner=False)
def get_health_report(settings: Settings) -> HealthReport:
    """Run the health check, cached briefly so every click doesn't re-query Ollama."""
    return check_ollama_health(settings)


def model_status_rows(report: HealthReport) -> list[dict[str, str]]:
    """Turn a report into table rows: one row per required model."""
    missing = {normalize_model_name(name) for name in report.missing_models}
    rows = []
    for name in report.required_models:
        if not report.ollama_reachable:
            status = "❔ Unknown (Ollama unreachable)"
        elif normalize_model_name(name) in missing:
            status = "❌ Missing"
        else:
            status = "✅ Installed"
        rows.append({"Model": name, "Status": status})
    return rows


def render_health_summary(report: HealthReport) -> None:
    """Compact status for the sidebar."""
    st.subheader("System status")
    if report.ok:
        st.success("Ollama connected. All models ready.", icon="✅")
    elif report.ollama_reachable:
        st.warning(
            f"Missing model(s): {', '.join(report.missing_models)}", icon="⚠️"
        )
    else:
        st.error("Ollama is not reachable.", icon="🚫")

    if st.button("Re-check", width="stretch"):
        get_health_report.clear()
        st.rerun()


def render_health_details(report: HealthReport, settings: Settings) -> None:
    """Full health panel for the System status tab."""
    st.markdown(f"**Ollama server:** `{report.ollama_host}`")

    if report.error:
        st.error(report.error)
        st.markdown(
            "**How to fix:**\n"
            "1. Install Ollama from https://ollama.com/download\n"
            "2. Make sure the Ollama app is running (llama icon in the system tray)\n"
            "3. Click **Re-check** in the sidebar"
        )

    st.markdown("**Required models**")
    st.table(model_status_rows(report))

    if report.pull_commands:
        st.markdown("Install the missing models by running:")
        st.code("\n".join(report.pull_commands), language="powershell")

    if report.ollama_reachable:
        with st.expander(f"All installed models ({len(report.installed_models)})"):
            if report.installed_models:
                for name in report.installed_models:
                    st.markdown(f"- `{name}`")
            else:
                st.write("No models installed yet.")

    with st.expander("Storage locations"):
        st.markdown(
            f"- Uploads: `{settings.uploads_dir}`\n"
            f"- Vector store: `{settings.chroma_dir}`\n"
            f"- Exports: `{settings.exports_dir}`"
        )