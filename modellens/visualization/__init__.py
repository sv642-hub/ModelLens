"""
Presentation-ready Plotly figures built from ModelLens analysis outputs.

Design notes live in ``modellens/visualization/DESIGN.md``. Prefer importing
specific plot functions from submodules for tree-shaking in notebooks.
"""

# ``showfig`` lives in a tiny module (no deps on ``common``) to avoid import-order issues.
from .plotly_display import showfig

from modellens.visualization.common import (
    default_plotly_layout,
    default_plotly_template,
    tensor_to_dataframe,
    to_numpy,
    truncate_label,
    truncate_labels,
)
from modellens.visualization.activation_patching import (
    format_patching_summary_html,
    plot_patching_importance_bar,
    plot_patching_importance_heatmap,
    plot_patching_recovery_fraction,
)
from modellens.visualization.backward_flow import plot_module_gradient_norms
from modellens.visualization.forward_flow import (
    plot_forward_trace_norms,
    plot_last_token_hidden_norm,
)
from modellens.visualization.logit_evolution import plot_logit_lens_confidence_panel
from modellens.visualization.overview import (
    model_info_markdown,
    parameter_summary_by_prefix,
    plot_parameter_sunburst_or_bar,
)
from modellens.visualization.training_curves import plot_snapshot_metric
from modellens.visualization.attention import (
    plot_attention_head_grid,
    plot_attention_heatmap,
)
from modellens.visualization.embeddings import (
    plot_embedding_norms,
    plot_embedding_similarity_heatmap,
)
from modellens.visualization.logit_lens import (
    plot_logit_lens_evolution,
    plot_logit_lens_heatmap,
    plot_logit_lens_top_token_bars,
)
from modellens.visualization.residuals import (
    plot_residual_contributions,
    plot_residual_lines,
)
from modellens.visualization.schemas import (
    AttentionVizData,
    EmbeddingVizData,
    LogitLensVizData,
    PatchingVizData,
    ResidualVizData,
    patching_dict_to_viz,
    residual_dict_to_viz,
)
from modellens.visualization.shapes import (
    compute_shape_trace,
    plot_shape_trace_table,
    shape_trace_mermaid,
    shape_trace_to_dataframe,
)

__all__ = [
    "showfig",
    "AttentionVizData",
    "EmbeddingVizData",
    "LogitLensVizData",
    "PatchingVizData",
    "ResidualVizData",
    "compute_shape_trace",
    "default_plotly_layout",
    "default_plotly_template",
    "format_patching_summary_html",
    "model_info_markdown",
    "parameter_summary_by_prefix",
    "patching_dict_to_viz",
    "plot_attention_head_grid",
    "plot_attention_heatmap",
    "plot_embedding_norms",
    "plot_embedding_similarity_heatmap",
    "plot_forward_trace_norms",
    "plot_last_token_hidden_norm",
    "plot_logit_lens_confidence_panel",
    "plot_logit_lens_evolution",
    "plot_logit_lens_heatmap",
    "plot_logit_lens_top_token_bars",
    "plot_module_gradient_norms",
    "plot_parameter_sunburst_or_bar",
    "plot_patching_importance_bar",
    "plot_patching_importance_heatmap",
    "plot_patching_recovery_fraction",
    "plot_residual_contributions",
    "plot_residual_lines",
    "plot_shape_trace_table",
    "plot_snapshot_metric",
    "residual_dict_to_viz",
    "shape_trace_mermaid",
    "shape_trace_to_dataframe",
    "tensor_to_dataframe",
    "to_numpy",
    "truncate_label",
    "truncate_labels",
]
