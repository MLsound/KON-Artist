"""
Global Continuous 3D Manifold Surface and Scatter Web Dashboard.

This script implements a reactive web dashboard using Streamlit and Plotly
to visualize global feature interactions and vulnerability boundaries
without partitioning data into separate acoustic clusters.

Usage:
    streamlit run src/diagnostics/dashboard_global_manifold.py
"""

import argparse
import glob
import os
import re
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import Rbf, griddata
import streamlit as st


def apply_academic_styling():
    """
    Injects custom CSS to enforce serif academic styling across the Streamlit app.
    """
    st.markdown(
        """
        <style>
        /* Force serif fonts on body, labels, headers, and markdown text */
        html, body, [class*="css"], [class*="st-"], .stMarkdown, p, div, span, label, h1, h2, h3, h4, h5, h6, input, button, select, textarea {
            font-family: "Times New Roman", Georgia, serif !important;
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def extract_timestamp(path):
    """
    Extracts the timestamp identifier from a file name.
    """
    filename = os.path.basename(path)
    match = re.search(r"\d{8}_\d{6}", filename)
    return match.group(0) if match else None


def scan_winners_files():
    """
    Scans outputs/ directory to find winners CSV files.
    """
    patterns = [
        "outputs/winners_*.csv",
        "outputs/**/winners_*.csv",
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))

    # Resolve absolute paths and remove duplicates
    resolved_files = sorted(list(set(os.path.abspath(f) for f in files)))
    return resolved_files


@st.cache_data
def load_data(winners_path):
    """
    Loads winners CSV file, wrapping with caching.
    """
    winners_df = pd.read_csv(winners_path)
    return winners_df


def interpolate_global_surface(df, x_col, y_col, z_col, grid_size=60):
    """
    Computes 2D mesh grid and interpolates Z values over the entire dataset.
    Also interpolates the score column if Z is not score, to lock the surfacecolor to the score gradient.
    """
    columns_to_agg = [z_col]
    if "score" not in columns_to_agg:
        columns_to_agg.append("score")

    # Group coordinates and take the mean to resolve duplicates
    agg_df = df.groupby([x_col, y_col])[columns_to_agg].mean().reset_index()

    x = agg_df[x_col].values
    y = agg_df[y_col].values
    z = agg_df[z_col].values
    s = agg_df["score"].values

    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()
    z_min, z_max = df[z_col].min(), df[z_col].max()
    s_min, s_max = df["score"].min(), df["score"].max()

    if x_max == x_min or y_max == y_min:
        return None, None, None, None

    grid_x = np.linspace(x_min, x_max, grid_size)
    grid_y = np.linspace(y_min, y_max, grid_size)
    gx, gy = np.meshgrid(grid_x, grid_y)

    def interp_channel(val_array, val_min, val_max):
        if len(x) < 4:
            try:
                gv = griddata((x, y), val_array, (gx, gy), method="linear")
                if np.isnan(gv).any():
                    gv_near = griddata((x, y), val_array, (gx, gy), method="nearest")
                    gv = np.where(np.isnan(gv), gv_near, gv)
                gv = np.clip(gv, val_min, val_max)
                return gv
            except Exception:
                return None
        try:
            rbf = Rbf(x, y, val_array, function="thin_plate", smooth=0.15)
            gv = rbf(gx, gy)
            gv = np.clip(gv, val_min, val_max)
            return gv
        except Exception:
            try:
                gv = griddata((x, y), val_array, (gx, gy), method="linear")
                if np.isnan(gv).any():
                    gv_near = griddata((x, y), val_array, (gx, gy), method="nearest")
                    gv = np.where(np.isnan(gv), gv_near, gv)
                gv = np.clip(gv, val_min, val_max)
                return gv
            except Exception:
                return None

    gz = interp_channel(z, z_min, z_max)
    gs = interp_channel(s, s_min, s_max)
    
    return gx, gy, gz, gs


def main():
    apply_academic_styling()

    # Dashboard Header (Serif typography, clean lines, no emojis)
    st.title("MTUCI/AASIST3 Global Adversarial Failure Manifold Dashboard")
    st.write(
        "Interactive web application visualizing global macro-vulnerabilities and decision "
        "boundaries across all acoustic profiles as a single unified manifold."
    )

    # 1. Global Workspace Data Scanner Sidebar
    st.sidebar.header("Data Ingestion Control Panel")
    all_files = scan_winners_files()

    if not all_files:
        st.sidebar.warning("No winners_*.csv files scanned in outputs/ directory.")
        manual_winners_path = st.sidebar.text_input(
            "Enter path to winners CSV file manually:"
        )
        if manual_winners_path and os.path.exists(manual_winners_path):
            selected_winners_path = manual_winners_path
        else:
            st.warning("Please locate or enter a valid winners_*.csv path.")
            return
    else:
        file_options = {os.path.basename(f): f for f in all_files}
        selected_file_name = st.sidebar.selectbox(
            "Select Run Output File (winners_*.csv):", list(file_options.keys())
        )
        selected_winners_path = file_options[selected_file_name]

    # Load file contents
    try:
        winners_df = load_data(selected_winners_path)
    except Exception as e:
        st.error(f"Error loading CSV files: {e}")
        return

    # Extract timestamp for layout naming
    session_id = extract_timestamp(selected_winners_path) or "unknown"

    # Display dataset dimensions information
    st.sidebar.info(
        f"Active Session: {session_id}\n"
        f"Total trajectory samples: {len(winners_df)}"
    )

    # Filter incoming rows to retain only valid model-bypass vectors (score >= 0.50)
    filtered_df = winners_df[winners_df["score"] >= 0.50].copy()

    if filtered_df.empty:
        st.warning("No successful model-bypass vectors (score >= 0.50) found in the selected file.")
        return

    # Isolate all columns starting with dsp_ prefix
    dsp_cols = sorted([col for col in filtered_df.columns if col.startswith("dsp_")])

    if not dsp_cols:
        st.error("No columns starting with 'dsp_' found in winners CSV.")
        return

    # 2. Unified Feature Selection Menu
    st.sidebar.header("Global Feature Mappings")

    # Select X-Axis (Default: dsp_tilt)
    default_x_idx = dsp_cols.index("dsp_tilt") if "dsp_tilt" in dsp_cols else 0
    x_axis_col = st.sidebar.selectbox(
        "X-Axis Feature Mapping:", dsp_cols, index=default_x_idx
    )

    # Select Y-Axis (Default: dsp_harmonics)
    default_y_idx = (
        dsp_cols.index("dsp_harmonics") if "dsp_harmonics" in dsp_cols else 0
    )
    y_axis_col = st.sidebar.selectbox(
        "Y-Axis Feature Mapping:", dsp_cols, index=default_y_idx
    )

    # Configurable Z-Axis response between score and reward (Default: score)
    z_axis_options = ["score", "reward"]
    z_axis_options = [opt for opt in z_axis_options if opt in filtered_df.columns]
    default_z_idx = z_axis_options.index("score") if "score" in z_axis_options else 0
    z_axis_col = st.sidebar.selectbox(
        "Z-Axis Response Metric:", z_axis_options, index=default_z_idx
    )

    # Color Scale Selector (score, reward, or any dsp_ metric; Default: score)
    color_options = []
    if "score" in filtered_df.columns:
        color_options.append("score")
    if "reward" in filtered_df.columns:
        color_options.append("reward")
    color_options.extend(dsp_cols)
    color_options = list(dict.fromkeys(color_options))

    default_color_idx = color_options.index("score") if "score" in color_options else 0
    color_col = st.sidebar.selectbox(
        "Color-Scale Variable Mapping:", color_options, index=default_color_idx
    )

    # Additional style choices
    st.sidebar.header("Visual Options")
    colorscale_choice = st.sidebar.selectbox(
        "Colormap Style Schema:", ["Viridis", "Cividis", "Plasma", "Inferno"]
    )

    show_global_surface = st.sidebar.checkbox(
        "Plot Global Surface Mesh (Opacity: 0.45)", value=True
    )

    # 3. Unified Global Plotly Graph Construction
    fig = go.Figure()

    # Calculate global range bounds for colorbar mapping
    c_min = filtered_df[color_col].min()
    c_max = filtered_df[color_col].max()
    if c_min == c_max:
        c_min -= 0.1
        c_max += 0.1

    # Plot continuous macroscopic vulnerability surface mesh if enabled
    if show_global_surface:
        gx, gy, gz, gs = interpolate_global_surface(
            filtered_df, x_axis_col, y_axis_col, z_axis_col, grid_size=60
        )

        if gx is not None and gz is not None and gs is not None:
            fig.add_trace(
                go.Surface(
                    x=gx,
                    y=gy,
                    z=gz,
                    surfacecolor=gs,  # Lock colorscale to continuous score gradient
                    opacity=0.45,
                    colorscale=colorscale_choice,
                    cmin=filtered_df["score"].min(),
                    cmax=filtered_df["score"].max(),
                    showscale=False,
                    name="Global Vulnerability Surface",
                    hoverinfo="skip",
                )
            )

    # Plot unified single global Scatter3d trace
    # Marker symbols are uniform (circles) with colors tied reactively to color_col
    hover_text = []
    for idx, row in filtered_df.iterrows():
        # Include cluster info dynamically if present, otherwise omit
        cluster_info_str = ""
        if "cluster" in filtered_df.columns:
            cluster_info_str = f"Acoustic Cluster ID: {int(row['cluster'])}<br>"
        
        hover_text.append(
            f"Sample Info:<br>"
            f"{cluster_info_str}"
            f"{x_axis_col}: {row[x_axis_col]:.4f}<br>"
            f"{y_axis_col}: {row[y_axis_col]:.4f}<br>"
            f"{z_axis_col}: {row[z_axis_col]:.4f}<br>"
            f"Color ({color_col}): {row[color_col]:.4f}"
        )

    fig.add_trace(
        go.Scatter3d(
            x=filtered_df[x_axis_col],
            y=filtered_df[y_axis_col],
            z=filtered_df[z_axis_col],
            mode="markers",
            name="Successful Deception Vectors",
            text=hover_text,
            hoverinfo="text",
            marker=dict(
                size=6,
                color=filtered_df[color_col],
                colorscale=colorscale_choice,
                cmin=c_min,
                cmax=c_max,
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text=color_col,
                        font=dict(size=12, family="serif"),
                    ),
                    tickfont=dict(size=10, family="serif"),
                    x=1.15,
                ),
                symbol="circle",
                line=dict(color="rgba(30, 30, 30, 0.8)", width=1),
            ),
        )
    )

    # Determine dynamic coordinate intervals
    x_range = [filtered_df[x_axis_col].min(), filtered_df[x_axis_col].max()]
    y_range = [filtered_df[y_axis_col].min(), filtered_df[y_axis_col].max()]
    z_range = [filtered_df[z_axis_col].min(), filtered_df[z_axis_col].max()]

    # Standard axis paddings
    for r in [x_range, y_range, z_range]:
        if r[1] != r[0]:
            diff = r[1] - r[0]
            r[0] -= diff * 0.05
            r[1] += diff * 0.05
        else:
            r[0] -= 0.5
            r[1] += 0.5

    # Structure plot layout conforming to academic formatting
    fig.update_layout(
        margin=dict(l=40, r=40, b=40, t=60),
        legend=dict(
            font=dict(size=10, family="serif"),
            x=0.01,
            y=0.99,
            bgcolor="rgba(255, 255, 255, 0.6)",
        ),
        scene=dict(
            xaxis=dict(
                title=dict(
                    text=f"{x_axis_col}", font=dict(size=11, family="serif")
                ),
                range=x_range,
                tickfont=dict(size=9, family="serif"),
                backgroundcolor="rgba(245, 245, 245, 0.5)",
                showbackground=True,
            ),
            yaxis=dict(
                title=dict(
                    text=f"{y_axis_col}", font=dict(size=11, family="serif")
                ),
                range=y_range,
                tickfont=dict(size=9, family="serif"),
                backgroundcolor="rgba(245, 245, 245, 0.5)",
                showbackground=True,
            ),
            zaxis=dict(
                title=dict(
                    text=f"{z_axis_col}", font=dict(size=11, family="serif")
                ),
                range=z_range,
                tickfont=dict(size=9, family="serif"),
                backgroundcolor="rgba(245, 245, 245, 0.5)",
                showbackground=True,
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=0.8),
            camera=dict(
                eye=dict(x=1.65, y=1.65, z=1.2),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.05),
            ),
        ),
    )

    # Render web graph
    st.plotly_chart(fig, width="stretch")

    # 4. Informational Table Panel
    st.markdown("### Successful Deception DSP Parameters Reference Table")
    st.write("Summary statistics computed directly from the successful model-bypass trajectory vectors:")
    dsp_summary = filtered_df[dsp_cols].describe().transpose()[["mean", "std", "min", "max"]]
    st.dataframe(dsp_summary, width="stretch")


if __name__ == "__main__":
    main()
