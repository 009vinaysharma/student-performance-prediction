"""
components/cards.py
Reusable HTML/CSS card components rendered via st.markdown.
"""

import streamlit as st


def render_hero(title: str, subtitle: str, badge_text: str = "AI-Powered • Live Predictions"):
    st.markdown(f"""
    <div class="hero-header">
        <div class="hero-badge">✨ {badge_text}</div>
        <h1 class="hero-title">{title}</h1>
        <p class="hero-subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_card(label: str, value: str, icon: str = "📊"):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_row(metrics: dict):
    """metrics: dict of label -> (value_str, icon)"""
    cols = st.columns(len(metrics))
    for col, (label, (value, icon)) in zip(cols, metrics.items()):
        with col:
            render_kpi_card(label, value, icon)


def render_badge(status: str):
    if str(status).lower() == "pass":
        st.markdown('<span class="badge badge-pass">✅ PASS</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-fail">❌ FAIL</span>', unsafe_allow_html=True)


def render_insight_card(text: str):
    st.markdown(f'<div class="insight-card">{text}</div>', unsafe_allow_html=True)


def render_section_title(text: str):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def render_divider():
    st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)


def render_glass_card_open():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)


def render_glass_card_close():
    st.markdown('</div>', unsafe_allow_html=True)
