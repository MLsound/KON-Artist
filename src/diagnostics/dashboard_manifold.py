"""
Global Continuous 3D Manifold Surface and Scatter Web Dashboard.

This script implements a reactive web dashboard using Streamlit and Plotly
to visualize global feature interactions and vulnerability boundaries
without partitioning data into separate acoustic clusters.

Usage: 
    streamlit run src/diagnostics/dashboard_manifold.py
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


# Inject custom CSS to enforce serif academic styling across the Streamlit app
def apply_academic_styling():
    st.markdown(
        """
        <style>
        /* Force serif fonts on body, labels, headers, and markdown text */
        html, body, [class*="css"], .stMarkdown, p, div, span, label, h1, h2, h3, h4, h5, h6 {
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
    Scans outputs/ and relative training folders to find winners CSV files.
    """
    patterns = [
        "outputs/**/winners_*.csv",
        "outputs/winners_*.csv",
        "../training/**/outputs/winners_*.csv",
        "../training/**/winners_*.csv",
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))

    # Resolve absolute paths and remove duplicates
    resolved_files = sorted(list(set(os.path.abspath(f) for f in files)))
    return resolved_files


def locate_paired_analysis_file(winners_path):
    """
    Attempts to locate the paired cluster analysis file in the same directory.
    """
    timestamp = extract_timestamp(winners_path)
    if not timestamp:
        return None

    dir_name = os.path.dirname(winners_path)
    possible_names = [
        f"analysis_clusters_{timestamp}.csv",
        f"analysis_clusters_{timestamp}_thresh0.00.csv",
        f"analysis_clusters_{timestamp}_thresh0.csv",
    ]
    for name in possible_names:
        path = os.path.join(dir_name, name)
        if os.path.exists(path):
            return path
    return None


def get_hyperparameters(active_session_id):
    """
    Scans the session log file to extract training hyperparameters.
    """
    if not active_session_id or active_session_id == "unknown":
        return None

    log_patterns = [
        f"outputs/train_session_{active_session_id}.log",
        f"outputs/logs/train_session_{active_session_id}.log",
    ]
    log_path = None
    for pat in log_patterns:
        abs_pat = os.path.abspath(pat)
        if os.path.exists(abs_pat):
            log_path = abs_pat
            break

    if not log_path:
        # Fallback using glob search
        matches = glob.glob(f"**/train_session_{active_session_id}.log", recursive=True)
        if matches:
            log_path = os.path.abspath(matches[0])

    if not log_path:
        return None

    params = {
        "TOTAL_TIMESTEPS": "X",
        "N_STEPS": "X",
        "TOTAL_UPDATES": "X",
        "EPOCHS": "X",
        "BATCH": "X",
    }

    try:
        pat_re = re.compile(
            r"TOTAL_TIMESTEPS\s*=\s*(?P<t>\d+)|"
            r"N_STEPS\s*=\s*(?P<n>\d+)|"
            r"TOTAL_UPDATES\s*=\s*(?P<u>\d+)|"
            r"EPOCHS\s*=\s*(?P<e>\d+)|"
            r"BATCH\s*=\s*(?P<b>\d+)"
        )
        with open(log_path, "r", encoding="utf-8") as f:
            for _ in range(5000):
                line = f.readline()
                if not line:
                    break
                for match in pat_re.finditer(line):
                    d = match.groupdict()
                    if d.get("t") is not None:
                        params["TOTAL_TIMESTEPS"] = d["t"]
                    if d.get("n") is not None:
                        params["N_STEPS"] = d["n"]
                    if d.get("u") is not None:
                        params["TOTAL_UPDATES"] = d["u"]
                    if d.get("e") is not None:
                        params["EPOCHS"] = d["e"]
                    if d.get("b") is not None:
                        params["BATCH"] = d["b"]
    except Exception:
        return None

    return params


def find_convergence_image(active_session_id):
    """
    Checks for convergence progress graphs matching the active session ID.
    """
    if not active_session_id or active_session_id == "unknown":
        return None

    extensions = [".png", ".jpg", ".jpeg"]
    search_paths = []
    for ext in extensions:
        search_paths.append(f"outputs/convergence_{active_session_id}{ext}")
        search_paths.append(f"outputs/plots/convergence_{active_session_id}{ext}")
        search_paths.append(f"outputs/history/convergence_{active_session_id}{ext}")

    for path in search_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path

    # Glob fallback search
    for ext in extensions:
        matches = glob.glob(f"**/convergence_{active_session_id}{ext}", recursive=True)
        if matches:
            return os.path.abspath(matches[0])

    return None


@st.cache_data
def load_data(winners_path, analysis_path):
    """
    Loads winners and analysis CSV files, wrapping with caching.
    """
    winners_df = pd.read_csv(winners_path)
    analysis_df = None
    if analysis_path and os.path.exists(analysis_path):
        analysis_df = pd.read_csv(analysis_path)
    return winners_df, analysis_df


def interpolate_global_surface(df, x_col, y_col, z_col, grid_size=50):
    """
    Computes 2D mesh grid and interpolates Z values over the entire dataset.
    """
    # Group coordinates and take the mean to resolve duplicates
    agg_df = df.groupby([x_col, y_col])[z_col].mean().reset_index()

    x = agg_df[x_col].values
    y = agg_df[y_col].values
    z = agg_df[z_col].values

    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()
    z_min, z_max = df[z_col].min(), df[z_col].max()

    if x_max == x_min or y_max == y_min:
        return None, None, None

    grid_x = np.linspace(x_min, x_max, grid_size)
    grid_y = np.linspace(y_min, y_max, grid_size)
    gx, gy = np.meshgrid(grid_x, grid_y)

    if len(x) < 4:
        try:
            gz = griddata((x, y), z, (gx, gy), method="linear")
            gz = np.clip(gz, z_min, z_max)
            return gx, gy, gz
        except Exception:
            return None, None, None

    try:
        rbf = Rbf(x, y, z, function="thin_plate", smooth=0.15)
        gz = rbf(gx, gy)
        gz = np.clip(gz, z_min, z_max)
        return gx, gy, gz
    except Exception:
        try:
            gz = griddata((x, y), z, (gx, gy), method="linear")
            gz = np.clip(gz, z_min, z_max)
            return gx, gy, gz
        except Exception:
            return None, None, None


def main():
    apply_academic_styling()

    # Dashboard Header (Serif typography, clean lines)
    st.title("MTUCI/AASIST3 Global Adversarial Failure Manifold Dashboard")
    st.write(
        "Interactive web application visualizing global macro-vulnerabilities and decision "
        "boundaries across all acoustic profiles as a single unified manifold."
    )

    # 1. Global Workspace Data Scanner Sidebar
    st.sidebar.header("Data Ingestion Control Panel")
    all_files = scan_winners_files()

    if not all_files:
        st.sidebar.warning("No winners_*.csv files scanned in outputs/ or training/.")
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

    # Automatically locate paired analysis file
    selected_analysis_path = locate_paired_analysis_file(selected_winners_path)

    # Load file contents
    try:
        winners_df, analysis_df = load_data(
            selected_winners_path, selected_analysis_path
        )
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

    # 2. Unified Feature Selection Menu
    st.sidebar.header("Global Feature Mappings")

    # Isolate all columns starting with dsp_ prefix
    dsp_cols = sorted([col for col in filtered_df.columns if col.startswith("dsp_")])

    if not dsp_cols:
        st.error("No columns starting with 'dsp_' found in winners CSV.")
        return

    # Select X-Axis
    default_x_idx = dsp_cols.index("dsp_tilt") if "dsp_tilt" in dsp_cols else 0
    x_axis_col = st.sidebar.selectbox(
        "X-Axis Feature Mapping:", dsp_cols, index=default_x_idx
    )

    # Select Y-Axis
    default_y_idx = (
        dsp_cols.index("dsp_harmonics") if "dsp_harmonics" in dsp_cols else 0
    )
    y_axis_col = st.sidebar.selectbox(
        "Y-Axis Feature Mapping:", dsp_cols, index=default_y_idx
    )

    # Configurable Z-Axis response between score and reward
    z_axis_col = st.sidebar.selectbox(
        "Z-Axis Response Metric:", ["score", "reward"], index=0
    )

    # Color Scale Selector (score, reward, or any dsp_ metric)
    color_options = ["score", "reward"] + dsp_cols
    default_color_idx = color_options.index("score")
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
        gx, gy, gz = interpolate_global_surface(
            filtered_df, x_axis_col, y_axis_col, z_axis_col, grid_size=60
        )

        if gx is not None and gz is not None:
            fig.add_trace(
                go.Surface(
                    x=gx,
                    y=gy,
                    z=gz,
                    opacity=0.45,
                    colorscale=colorscale_choice,
                    cmin=c_min,
                    cmax=c_max,
                    showscale=False,
                    name="Global Vulnerability Surface",
                    hoverinfo="skip",
                )
            )

    # Plot unified single global Scatter3d trace
    # Marker symbols are uniform (circles) with colors tied reactively to color_col
    hover_text = []
    for _, row in filtered_df.iterrows():
        # Include cluster info as supplementary metadata, but do not split trace
        cluster_info = (
            f"Cluster ID: {int(row['cluster'])}"
            if "cluster" in filtered_df.columns
            else "N/A"
        )
        hover_text.append(
            f"Sample Info:<br>"
            f"Acoustic {cluster_info}<br>"
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
    st.plotly_chart(fig, use_container_width=True)

    # 4. Informational Table Panel
    st.markdown("### Workspace Aggregate Metrics Reference Table")
    if analysis_df is not None:
        st.write("Summary statistics loaded directly from the paired run execution:")
        st.dataframe(analysis_df, use_container_width=True)
    else:
        st.warning(
            "Paired analysis_clusters_*.csv not found. Re-run training loops to create."
        )


if __name__ == "__main__":
    main()
