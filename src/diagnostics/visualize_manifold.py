"""
Production-Grade 3D Manifold Surface and Scatter Visualization Script.

This script maps the empirical failure manifolds of the MTUCI/AASIST3 model
using successful adversarial DSP configurations. It outputs an interactive
HTML visualization and a publication-quality static PNG visualization.

Usage:
    python -m src.diagnostics.visualize_manifold \
        --winners_csv path/to/winners.csv \
        --analysis_csv path/to/analysis_clusters.csv \
        --output_dir outputs/diagnostics/
"""

import argparse
import os
import re
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import Rbf, griddata

# Import matplotlib for robust high-resolution static rendering fallback
try:
    import matplotlib
    matplotlib.use('Agg')  # Headless backend
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ManifoldVisualizer:
    """
    Manages loading, filtering, interpolation, and visualization of 3D
    acoustic manifolds for model deception analysis.
    """

    def __init__(self, winners_csv_path, analysis_csv_path, output_dir):
        """
        Initializes the visualizer with paths and configurations.

        Parameters:
            winners_csv_path (str): Path to winners trajectory log CSV.
            analysis_csv_path (str): Path to aggregate cluster metrics CSV.
            output_dir (str): Base directory for visualization outputs.
        """
        self.winners_csv_path = winners_csv_path
        self.analysis_csv_path = analysis_csv_path
        self.output_dir = output_dir

        self.winners_df = None
        self.analysis_df = None
        self.session_id = "default"

        # Define markers for the specific clusters to show clear distinction
        # 0: circle, 1: square, 4: diamond, 8: cross (x in matplotlib), 9: x (plus in matplotlib)
        self.cluster_markers_plotly = {
            0: "circle",
            1: "square",
            4: "diamond",
            8: "cross",
            9: "x"
        }
        self.cluster_markers_matplotlib = {
            0: "o",
            1: "s",
            4: "D",
            8: "P",
            9: "X"
        }

    def extract_session_id(self):
        """
        Extracts the timestamp session ID from file names to align outputs.
        """
        filename = os.path.basename(self.winners_csv_path)
        match = re.search(r"\d{8}_\d{6}", filename)
        if match:
            self.session_id = match.group(0)
        else:
            self.session_id = "20260616_134732"

    def load_data(self):
        """
        Loads CSV files and filters records for model-bypass vectors.
        """
        if not os.path.exists(self.winners_csv_path):
            raise FileNotFoundError(
                f"Winners CSV not found: {self.winners_csv_path}"
            )
        if not os.path.exists(self.analysis_csv_path):
            raise FileNotFoundError(
                f"Analysis CSV not found: {self.analysis_csv_path}"
            )

        print(f"Loading winners from: {self.winners_csv_path}")
        self.winners_df = pd.read_csv(self.winners_csv_path)

        print(f"Loading cluster analysis from: {self.analysis_csv_path}")
        self.analysis_df = pd.read_csv(self.analysis_csv_path)

        self.extract_session_id()

        # Validate mandatory columns
        required_cols = ["dsp_tilt", "dsp_harmonics", "dsp_jitter", "score", "cluster"]
        for col in required_cols:
            if col not in self.winners_df.columns:
                raise ValueError(
                    f"Required column '{col}' is missing from winners CSV."
                )

        # Print raw metrics information
        total_rows = len(self.winners_df)
        print(f"Loaded {total_rows} total rows from winners CSV.")

        # Filter out rows where score is less than the success threshold (score < 0.50)
        # to ensure only valid model-bypass vectors are used
        self.winners_df = self.winners_df[self.winners_df["score"] >= 0.50].copy()
        filtered_rows = len(self.winners_df)
        print(
            f"Filtered data to score >= 0.50: {filtered_rows} of {total_rows} "
            f"rows retained ({filtered_rows/total_rows*100:.2f}%)."
        )

        # Standardize cluster representation to integer
        self.winners_df["cluster"] = self.winners_df["cluster"].astype(int)

    def interpolate_surface(self, df_cluster, grid_size=50):
        """
        Constructs an empirical vulnerability surface mesh using thin-plate spline
        or linear bivariate interpolation over the bounded X (tilt) and Y (harmonics) domain.

        Parameters:
            df_cluster (pd.DataFrame): Subset of data for a specific cluster.
            grid_size (int): Size of the 2D mesh grid.

        Returns:
            gx (np.ndarray): 2D grid coordinates for X-axis.
            gy (np.ndarray): 2D grid coordinates for Y-axis.
            gz (np.ndarray): Interpolated Z values (score).
        """
        # Aggregate duplicates by taking the mean score to prevent singular interpolation matrix
        agg_df = (
            df_cluster.groupby(["dsp_tilt", "dsp_harmonics"])["score"]
            .mean()
            .reset_index()
        )

        x = agg_df["dsp_tilt"].values
        y = agg_df["dsp_harmonics"].values
        z = agg_df["score"].values

        # Generate meshgrid bounded on [-1.0, 1.0] for tilt and harmonics
        grid_x = np.linspace(-1.0, 1.0, grid_size)
        grid_y = np.linspace(-1.0, 1.0, grid_size)
        gx, gy = np.meshgrid(grid_x, grid_y)

        # We need at least 4 unique coordinates for thin-plate spline interpolation
        if len(x) < 4:
            print(
                f"Fewer than 4 unique spatial coordinates for cluster "
                f"(count={len(x)}). Using linear bivariate griddata interpolation."
            )
            try:
                gz = griddata((x, y), z, (gx, gy), method="linear")
                gz = np.clip(gz, 0.5, 1.0)
                return gx, gy, gz
            except Exception as e:
                print(f"Linear griddata interpolation failed: {e}")
                return None, None, None

        # Attempt Thin-Plate Spline (Rbf) interpolation
        try:
            # smooth=0.1 avoids exact interpolation constraints and handles noise/singularities
            rbf = Rbf(x, y, z, function="thin_plate", smooth=0.1)
            gz = rbf(gx, gy)
            # Clip surface heights to valid score bounds [0.5, 1.0]
            gz = np.clip(gz, 0.5, 1.0)
            return gx, gy, gz
        except Exception as e:
            print(
                f"Thin-plate spline Rbf interpolation failed. "
                f"Falling back to linear griddata. Error: {e}"
            )
            try:
                gz = griddata((x, y), z, (gx, gy), method="linear")
                gz = np.clip(gz, 0.5, 1.0)
                return gx, gy, gz
            except Exception as ex:
                print(f"Linear fallback interpolation failed: {ex}")
                return None, None, None

    def build_plotly_figure(self, colorscale="Cividis"):
        """
        Constructs the multi-layered 3D interactive Plotly scene containing
        point vectors and continuous vulnerability surfaces.
        """
        fig = go.Figure()

        unique_clusters = sorted(self.winners_df["cluster"].unique())
        print(f"Plotting clusters: {unique_clusters}")

        # Keep track of which trace shows the color scale to avoid duplicates
        colorbar_shown = False

        # 1. Plot the continuous vulnerability surface meshes for Cluster 1 and Cluster 4
        dense_surface_clusters = [1, 4]
        for cid in dense_surface_clusters:
            if cid in unique_clusters:
                cluster_df = self.winners_df[self.winners_df["cluster"] == cid]
                gx, gy, gz = self.interpolate_surface(cluster_df, grid_size=50)

                if gx is not None and gz is not None:
                    print(
                        f"Adding 3D empirical surface mesh for Cluster {cid} "
                        f"(Opacity: 0.4)."
                    )
                    # Use a dedicated colorscale mapping or custom mesh layout
                    fig.add_trace(
                        go.Surface(
                            x=gx,
                            y=gy,
                            z=gz,
                            opacity=0.4,
                            colorscale=colorscale,
                            cmin=0.5,
                            cmax=1.0,
                            showscale=False,
                            name=f"Cluster {cid} Surface",
                            hoverinfo="skip",  # Do not clutter hover cards
                        )
                    )

        # 2. Add scatter layers for each unique cluster
        for cid in unique_clusters:
            cluster_df = self.winners_df[self.winners_df["cluster"] == cid]
            marker_symbol = self.cluster_markers_plotly.get(cid, "circle")

            # Display colorbar only for the first added scatter trace to prevent overlap
            show_scale = False
            if not colorbar_shown:
                show_scale = True
                colorbar_shown = True

            print(
                f"Adding 3D Scatter layer for Cluster {cid} with marker "
                f"'{marker_symbol}' ({len(cluster_df)} points)."
            )

            # Custom hover text including dsp_jitter metadata
            hover_text = []
            for _, row in cluster_df.iterrows():
                hover_text.append(
                    f"Cluster: {row['cluster']}<br>"
                    f"Tilt: {row['dsp_tilt']:.4f}<br>"
                    f"Harmonics: {row['dsp_harmonics']:.4f}<br>"
                    f"Jitter: {row['dsp_jitter']:.4f}<br>"
                    f"Score: {row['score']:.4f}"
                )

            fig.add_trace(
                go.Scatter3d(
                    x=cluster_df["dsp_tilt"],
                    y=cluster_df["dsp_harmonics"],
                    z=cluster_df["score"],
                    mode="markers",
                    name=f"Cluster {cid} (Scatter)",
                    text=hover_text,
                    hoverinfo="text",
                    marker=dict(
                        size=6,
                        color=cluster_df["score"],
                        colorscale=colorscale,
                        cmin=0.5,
                        cmax=1.0,
                        showscale=show_scale,
                        colorbar=(
                            dict(
                                title=dict(
                                    text="AASIST3 Deception Score P(BonaFide)",
                                    font=dict(size=12, family="serif"),
                                ),
                                tickfont=dict(size=10, family="serif"),
                                x=1.15,
                            )
                            if show_scale
                            else None
                        ),
                        symbol=marker_symbol,
                        line=dict(color="rgba(30, 30, 30, 0.8)", width=1),
                    ),
                )
            )

        # Update scene formatting to avoid overlaps and use academic Serif font
        fig.update_layout(
            title=dict(
                text=(
                    f"MTUCI/AASIST3 Adversarial Failure Manifold Surface Mapping "
                    f"(Session {self.session_id})"
                ),
                font=dict(size=14, family="serif"),
                x=0.5,
                y=0.95,
                xanchor="center",
            ),
            margin=dict(l=60, r=100, b=60, t=80),
            legend=dict(
                font=dict(size=10, family="serif"),
                x=0.02,
                y=0.98,
                bgcolor="rgba(255, 255, 255, 0.6)",
                bordercolor="rgba(200, 200, 200, 0.5)",
                borderwidth=1,
            ),
            scene=dict(
                xaxis=dict(
                    title=dict(
                        text="Spectral Tilt (dsp_tilt)",
                        font=dict(size=12, family="serif"),
                    ),
                    range=[-1.0, 1.0],
                    tickfont=dict(size=10, family="serif"),
                    gridcolor="rgba(220, 220, 220, 0.8)",
                    zerolinecolor="rgba(180, 180, 180, 0.8)",
                    backgroundcolor="rgba(245, 245, 245, 0.5)",
                    showbackground=True,
                ),
                yaxis=dict(
                    title=dict(
                        text="Harmonic Injection (dsp_harmonics)",
                        font=dict(size=12, family="serif"),
                    ),
                    range=[-1.0, 1.0],
                    tickfont=dict(size=10, family="serif"),
                    gridcolor="rgba(220, 220, 220, 0.8)",
                    zerolinecolor="rgba(180, 180, 180, 0.8)",
                    backgroundcolor="rgba(245, 245, 245, 0.5)",
                    showbackground=True,
                ),
                zaxis=dict(
                    title=dict(
                        text="AASIST3 Deception Score P(BonaFide)",
                        font=dict(size=12, family="serif"),
                    ),
                    range=[0.5, 1.0],
                    tickfont=dict(size=10, family="serif"),
                    gridcolor="rgba(220, 220, 220, 0.8)",
                    zerolinecolor="rgba(180, 180, 180, 0.8)",
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

        return fig

    def generate_matplotlib_fallback(self, output_png_path, colorscale="cividis"):
        """
        Fallback implementation using Matplotlib to generate a publication-quality
        3D visualization. Saves directly to the requested output file at 300 DPI.
        """
        if not HAS_MATPLOTLIB:
            print(
                "Matplotlib fallback requested but matplotlib is not installed. "
                "Skipping fallback rendering."
            )
            return False

        print("Generating high-resolution static PNG using Matplotlib.")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Map string colorscales to matplotlib colormaps
        cmap_name = colorscale.lower()
        if cmap_name not in ["viridis", "cividis", "plasma", "inferno"]:
            cmap_name = "cividis"
        cmap = cm.get_cmap(cmap_name)

        unique_clusters = sorted(self.winners_df["cluster"].unique())

        # 1. Interpolate and plot surfaces for Cluster 1 and Cluster 4
        dense_surface_clusters = [1, 4]
        for cid in dense_surface_clusters:
            if cid in unique_clusters:
                cluster_df = self.winners_df[self.winners_df["cluster"] == cid]
                gx, gy, gz = self.interpolate_surface(cluster_df, grid_size=40)

                if gx is not None and gz is not None:
                    # Mask out NaNs if any occurred during linear interpolation
                    # matplotlib plot_surface handles masked arrays well
                    masked_gz = np.ma.masked_invalid(gz)

                    # Plot surface with alpha 0.4
                    # We specify colormap for heights
                    surf = ax.plot_surface(
                        gx,
                        gy,
                        masked_gz,
                        cmap=cmap,
                        alpha=0.4,
                        linewidth=0.1,
                        edgecolors="rgba(50,50,50,0.3)",
                        vmin=0.5,
                        vmax=1.0,
                    )

        # 2. Add scatter layers for each cluster
        scatters = []
        for cid in unique_clusters:
            cluster_df = self.winners_df[self.winners_df["cluster"] == cid]
            marker_symbol = self.cluster_markers_matplotlib.get(cid, "o")

            x = cluster_df["dsp_tilt"].values
            y = cluster_df["dsp_harmonics"].values
            z = cluster_df["score"].values

            sc = ax.scatter(
                x,
                y,
                z,
                c=z,
                cmap=cmap,
                marker=marker_symbol,
                s=40,
                vmin=0.5,
                vmax=1.0,
                edgecolors="rgba(30,30,30,0.8)",
                linewidths=0.7,
                label=f"Cluster {cid} (Scatter)",
            )
            scatters.append(sc)

        # Plot labels with LaTeX-style layout styling
        ax.set_xlabel("Spectral Tilt (dsp_tilt)", fontfamily="serif", fontsize=11, labelpad=10)
        ax.set_ylabel(
            "Harmonic Injection (dsp_harmonics)",
            fontfamily="serif",
            fontsize=11,
            labelpad=10,
        )
        ax.set_zlabel(
            "AASIST3 Deception Score P(BonaFide)",
            fontfamily="serif",
            fontsize=11,
            labelpad=10,
        )

        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(0.5, 1.0)

        # Apply serif font to ticks
        for label in (
            ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()
        ):
            label.set_fontfamily("serif")
            label.set_fontsize(9)

        # Configure view angle (matching Plotly's perspective)
        ax.view_init(elev=25, azim=-45)

        # Add single colorbar mapping scores
        mappable = cm.ScalarMappable(cmap=cmap)
        mappable.set_array([])
        mappable.set_clim(0.5, 1.0)
        cbar = fig.colorbar(
            mappable,
            ax=ax,
            shrink=0.5,
            aspect=15,
            pad=0.1,
            format="%.1f",
        )
        cbar.set_label(
            "AASIST3 Deception Score P(BonaFide)",
            fontfamily="serif",
            fontsize=10,
        )
        for label in cbar.ax.get_yticklabels():
            label.set_fontfamily("serif")
            label.set_fontsize(9)

        # Add title and legend
        ax.set_title(
            f"Adversarial Failure Manifold Surface Mapping\n(Session {self.session_id})",
            fontfamily="serif",
            fontsize=13,
            pad=10,
        )
        ax.legend(
            loc="upper left",
            prop={"family": "serif", "size": 8},
            framealpha=0.6,
        )

        plt.tight_layout()
        fig.savefig(output_png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Successfully saved Matplotlib fallback PNG to: {output_png_path}")
        return True

    def save_visualizations(self, colorscale="Cividis"):
        """
        Generates and saves the HTML and high-resolution PNG plots.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        # Generate paths
        html_filename = "manifold_alignment_surface.html"
        png_filename = f"cluster_vulnerability_score_{self.session_id}.png"

        html_path = os.path.join(self.output_dir, html_filename)
        png_path = os.path.join(self.output_dir, png_filename)

        # Create Plotly interactive object
        fig = self.build_plotly_figure(colorscale=colorscale)

        # Save standalone HTML
        print(f"Saving interactive HTML to: {html_path}")
        fig.write_html(html_path, include_plotlyjs="cdn")

        # Save static PNG using plotly's write_image with scale parameter
        # to ensure publication-ready output
        print(f"Exporting static PNG to: {png_path}")
        try:
            # Width and height adjusted to standard academic layout specs (4:3 ratio)
            fig.write_image(png_path, scale=3, width=1000, height=750)
            print(f"Successfully exported high-resolution Plotly PNG to: {png_path}")
        except Exception as e:
            print(
                f"Plotly static PNG export failed. Error details: {e}. "
                f"Invoking Matplotlib fallback render engine."
            )
            # Run the Matplotlib fallback
            success = self.generate_matplotlib_fallback(
                png_path, colorscale=colorscale
            )
            if not success:
                print(
                    "Warning: Could not export static PNG image. "
                    "Make sure kaleido or matplotlib is installed."
                )


def main():
    """
    Main CLI entry point for the manifold visualization tool.
    """
    parser = argparse.ArgumentParser(
        description="Visualize the 3D adversarial decision manifolds of AASIST3."
    )
    parser.add_argument(
        "--winners_csv",
        type=str,
        default="../training/KONArtist_20260616_134705/outputs/winners_20260616_134732.csv",
        help="Path to the winners csv trajectory log.",
    )
    parser.add_argument(
        "--analysis_csv",
        type=str,
        default="../training/KONArtist_20260616_134705/outputs/analysis_clusters_20260616_134732.csv",
        help="Path to the cluster analysis csv file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/diagnostics",
        help="Directory where output files will be written.",
    )
    parser.add_argument(
        "--colorscale",
        type=str,
        default="Cividis",
        choices=["Cividis", "Viridis"],
        help="Continuous colorscale for scoring gradient.",
    )

    args = parser.parse_args()

    # Convert paths to absolute paths to guarantee script runs from any directory
    winners_path = os.path.abspath(args.winners_csv)
    analysis_path = os.path.abspath(args.analysis_csv)
    output_directory = os.path.abspath(args.output_dir)

    print("--- MTUCI/AASIST3 Manifold Visualization Tool ---")
    visualizer = ManifoldVisualizer(
        winners_csv_path=winners_path,
        analysis_csv_path=analysis_path,
        output_dir=output_directory,
    )

    try:
        visualizer.load_data()
        visualizer.save_visualizations(colorscale=args.colorscale)
        print("Visualizations created successfully.")
    except Exception as e:
        print(f"Error during execution: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
