"""Figure style. Uses the Quantis Research Figure module when available, otherwise a plain matplotlib fallback."""
import os, sys, textwrap
sys.path.insert(0, os.path.expanduser("~/Desktop/Quantis Design System/templates"))
try:
    from quantis_research_figure import *          # noqa: F401,F403
except ImportError:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    TEXT_DARK, TEXT_MID, TEXT_LIGHT = "#080808", "#222222", "#565656"
    STRONG_GRAY, MID_GRAY, LIGHT_GRAY = "#606060", "#b8b8b8", "#d8d8d8"
    GREEN, AMBER, OXBLOOD, ACCENT_SOFT = "#2d6a2d", "#c87a1a", "#8B1A1A", "#f5e8e8"

    def frame(title, subtitle, category, provenance, w=10.0, h=6.25):
        fig = plt.figure(figsize=(w, h))
        fig.text(0.045, 0.955, title, weight="bold", fontsize=16, ha="left", va="top")
        fig.text(0.045, 0.893, subtitle, style="italic", fontsize=10.5, color=TEXT_LIGHT, ha="left", va="top")
        fig.text(0.045, 0.03, category + "   " + "\n".join(textwrap.wrap(provenance, 150)), fontsize=6.8, color=STRONG_GRAY, va="bottom")
        return fig

    def tidy(ax, grid_y=True):
        ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(length=0)
        if grid_y:
            ax.yaxis.grid(True, color=LIGHT_GRAY, lw=0.6); ax.set_axisbelow(True)

    def save(fig, name, outdir):
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, name + ".png"), dpi=220); plt.close(fig)
