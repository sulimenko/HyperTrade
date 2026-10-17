"""Consistent research-terminal chart styling."""
import plotly.express as px
import streamlit as st

CYAN = '#37C7F3'
VIOLET = '#9263F4'
px.defaults.template = 'plotly_dark'
px.defaults.color_discrete_sequence = [CYAN,VIOLET,'#49D3A5','#F4B860']


def show(figure, **kwargs):
    figure.update_layout(template='plotly_dark', paper_bgcolor='#0B1220', plot_bgcolor='#18243A',
                         font=dict(color='#E8EEF8', family='Arial'), colorway=[CYAN, VIOLET, '#49D3A5', '#F4B860'],
                         margin=dict(l=30, r=25, t=40, b=35), legend=dict(orientation='h', y=1.12),
                         scene=dict(bgcolor='#18243A'), hoverlabel=dict(bgcolor='#18243A'))
    return st.plotly_chart(figure, width='stretch', theme=None, **kwargs)


def line(frame, x, y, title, **kwargs):
    if not frame.empty and y in frame:
        return show(px.line(frame, x=x, y=y, title=title, **kwargs))
