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
    plot_patching_family_effect_recovery_heatmap,
)
from modellens.visualization.backward_flow import (
    plot_gradient_norm_distribution_by_family,
    plot_gradient_norm_family_aggregate,
    plot_gradient_norm_top_n,
    plot_module_gradient_norms,
)
from modellens.visualization.forward_flow import (
    plot_activation_norm_distribution_by_family,
    plot_forward_family_aggregate,
    plot_forward_trace_norms,
    plot_forward_trace_top_n,
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
    plot_attention_head_entropy,
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
from modellens.visualization.module_families import infer_module_family
from modellens.visualization.comparison_story import (
    format_comparison_summary_html,
    format_patching_story_html,
    plot_attention_comparison_heatmaps,
    plot_attention_entropy_delta_heads,
    plot_divergence_by_module,
    plot_family_divergence,
    plot_logit_lens_comparison_trajectories,
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
    "plot_attention_head_entropy",
    "plot_embedding_norms",
    "plot_embedding_similarity_heatmap",
    "plot_forward_trace_norms",
    "plot_forward_trace_top_n",
    "plot_forward_family_aggregate",
    "plot_activation_norm_distribution_by_family",
    "plot_last_token_hidden_norm",
    "plot_logit_lens_confidence_panel",
    "plot_logit_lens_evolution",
    "plot_logit_lens_heatmap",
    "plot_logit_lens_top_token_bars",
    "plot_module_gradient_norms",
    "plot_gradient_norm_top_n",
    "plot_gradient_norm_family_aggregate",
    "plot_gradient_norm_distribution_by_family",
    "plot_parameter_sunburst_or_bar",
    "plot_patching_importance_bar",
    "plot_patching_importance_heatmap",
    "plot_patching_recovery_fraction",
    "plot_patching_family_effect_recovery_heatmap",
    "plot_residual_contributions",
    "plot_residual_lines",
    "plot_shape_trace_table",
    "plot_snapshot_metric",
    "residual_dict_to_viz",
    "infer_module_family",
    "format_comparison_summary_html",
    "format_patching_story_html",
    "plot_attention_comparison_heatmaps",
    "plot_attention_entropy_delta_heads",
    "plot_divergence_by_module",
    "plot_family_divergence",
    "plot_logit_lens_comparison_trajectories",
    "shape_trace_mermaid",
    "shape_trace_to_dataframe",
    "tensor_to_dataframe",
    "to_numpy",
    "truncate_label",
    "truncate_labels",
]
