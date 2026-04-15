# Plot Configuration for SitToStand Biomechanics Project
# This file contains all styling parameters for consistent plotting across the project

import plotly.graph_objects as go

# ==================== COLOR PALETTE ====================
COLORS = {
    # Primary condition colors
    'normal': 'skyblue',           # Normal/Natural sit-to-stand
    'weak': 'lightcoral',          # Weak/Trunk flexion sit-to-stand
    
    # OpenCap vs Motion Capture colors
    'opencap': '#1f77b4',          # Blue for OpenCap
    'mocap': '#ff7f0e',            # Orange for Motion Capture
    
    # Method comparison colors
    'mono': '#1f77b4',             # Blue for Mono (same as OpenCap)
    'wham': '#d62728',             # Red for WHAM
    'twocam': '#ff7f0e',           # Orange for 2CAMS (same as mocap)
    
    # Joint colors for moment plots
    'knee': '#2ca02c',             # Green
    'hip': '#d62728',              # Red  
    'ankle': '#9467bd',            # Purple
    
    # Secondary colors
    'line_unity': 'grey',          # Line of unity/reference lines
    'grid': '#e6e6e6',             # Grid lines (if used)
    'text': 'black',               # Primary text
    'background': 'white',         # Plot background
    
    # Accent colors with transparency
    'normal_fill': 'rgba(135, 206, 235, 0.7)',     # skyblue with alpha
    'weak_fill': 'rgba(240, 128, 128, 0.7)',       # lightcoral with alpha
}

# Darker versions for borders/outlines
BORDER_COLORS = {
    'normal': 'darkblue',
    'weak': 'darkred',
    'opencap': '#1f77b4',
    'mocap': '#ff7f0e',
    'knee': 'darkgreen',
    'hip': 'darkred',
    'ankle': 'darkviolet',
}

# ==================== FONTS ====================
FONTS = {
    'family': 'sans-serif',  # Use system default sans-serif font
    'title_size': 18,
    'axis_title_size': 16,
    'tick_size': 14,
    'legend_size': 14,
    'annotation_size': 14,
}

# ==================== LAYOUT SETTINGS ====================
LAYOUT = {
    'width': 800,
    'height': 600,
    'margin': dict(l=80, r=50, t=80, b=80),
    'plot_bgcolor': COLORS['background'],
    'paper_bgcolor': COLORS['background'],
}

# ==================== MARKER SETTINGS ====================
MARKERS = {
    'size': 8,
    'opacity': 0.7,
    'line_width': 1,
}

# ==================== LINE SETTINGS ====================
LINES = {
    'width': 2,
    'dash_unity': 'dash',  # For line of unity
    'dash_reference': 'dot',  # For reference lines
}

# ==================== AXIS SETTINGS ====================
AXIS_CONFIG = {
    'showgrid': False,
    'zeroline': False,
    'linecolor': 'black',
    'linewidth': 1,
    'mirror': False,
    'ticks': 'outside',
    'tickfont': {'size': FONTS['tick_size'], 'family': FONTS['family']},
}

# ==================== LEGEND SETTINGS ====================
LEGEND_CONFIG = {
    'x': 0.98,
    'y': 0.98,
    'xanchor': 'right',
    'yanchor': 'top',
    'bgcolor': 'rgba(255,255,255,0)',  # Transparent
    'bordercolor': 'rgba(0,0,0,0)',    # Transparent border
    'borderwidth': 0,
    'font': {'size': FONTS['legend_size'], 'family': FONTS['family']},
}

# ==================== ANNOTATION SETTINGS ====================
ANNOTATION_CONFIG = {
    'showarrow': False,
    'font': {'size': FONTS['annotation_size'], 'color': COLORS['text'], 'family': FONTS['family']},
    'bgcolor': 'rgba(255,255,255,0)',  # Transparent background
    'bordercolor': 'rgba(0,0,0,0)',    # Transparent border
    'borderwidth': 0,
    'align': 'left',
}

# ==================== CONDITION LABELS ====================
CONDITION_LABELS = {
    'normal': 'Natural sit-to-stand',
    'weak': 'Trunk flexion sit-to-stand',
}

# ==================== JOINT LABELS ====================
JOINT_LABELS = ['Knee', 'Hip', 'Ankle']

# ==================== HELPER FUNCTIONS ====================

def get_standard_layout(title=None, xaxis_title=None, yaxis_title=None):
    """Get standard layout configuration with optional titles"""
    layout = {
        'font': {'size': FONTS['axis_title_size'], 'family': FONTS['family']},
        'plot_bgcolor': LAYOUT['plot_bgcolor'],
        'paper_bgcolor': LAYOUT['paper_bgcolor'],
        'width': LAYOUT['width'],
        'height': LAYOUT['height'],
        'margin': LAYOUT['margin'],
        'legend': LEGEND_CONFIG.copy(),
    }
    
    if title:
        layout['title'] = {
            'text': title,
            'font': {'size': FONTS['title_size'], 'family': FONTS['family']},
            'x': 0.5,  # Center title
        }
    
    if xaxis_title:
        layout['xaxis_title'] = {
            'text': xaxis_title,
            'font': {'size': FONTS['axis_title_size'], 'family': FONTS['family']},
        }
    
    if yaxis_title:
        layout['yaxis_title'] = {
            'text': yaxis_title,
            'font': {'size': FONTS['axis_title_size'], 'family': FONTS['family']},
        }
    
    return layout

def get_standard_axes():
    """Get standard axis configuration"""
    return AXIS_CONFIG.copy()

def get_condition_trace(x_data, y_data, condition, name=None, hovertemplate=None):
    """Create a standardized scatter trace for normal/weak conditions"""
    if name is None:
        name = CONDITION_LABELS[condition]
    
    return go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers',
        name=name,
        marker=dict(
            color=COLORS[condition],
            size=MARKERS['size'],
            opacity=MARKERS['opacity'],
            line=dict(width=MARKERS['line_width'], color=BORDER_COLORS[condition])
        ),
        hovertemplate=hovertemplate
    )

def get_unity_line(min_val, max_val, name='Line of Unity', showlegend=False):
    """Create a standardized line of unity"""
    return go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name=name,
        showlegend=showlegend,
        line=dict(
            color=COLORS['line_unity'], 
            dash=LINES['dash_unity'], 
            width=LINES['width']
        ),
        hovertemplate='Perfect Agreement<extra></extra>'
    )

def add_stats_annotation(fig, stats_text, x=0.98, y=0.02):
    """Add standardized statistics annotation to figure"""
    annotation_config = ANNOTATION_CONFIG.copy()
    annotation_config.update({
        'text': stats_text,
        'xref': 'paper',
        'yref': 'paper',
        'x': x,
        'y': y,
    })
    fig.add_annotation(**annotation_config)

def apply_standard_layout(fig, title=None, xaxis_title=None, yaxis_title=None):
    """Apply standard layout and axis styling to a figure"""
    layout_config = get_standard_layout(title, xaxis_title, yaxis_title)
    axis_config = get_standard_axes()
    
    fig.update_layout(**layout_config)
    fig.update_xaxes(**axis_config)
    fig.update_yaxes(**axis_config)
    
    return fig

# ==================== EXAMPLE USAGE ====================
"""
# Example usage in a plotting script:

import plot_config as pc
import plotly.graph_objects as go

# Create figure
fig = go.Figure()

# Add traces with consistent styling
fig.add_trace(pc.get_condition_trace(
    x_data=normal_x, 
    y_data=normal_y, 
    condition='normal',
    hovertemplate='Normal<br>X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>'
))

fig.add_trace(pc.get_condition_trace(
    x_data=weak_x, 
    y_data=weak_y, 
    condition='weak',
    hovertemplate='Weak<br>X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>'
))

# Add line of unity if needed
min_val = min(min(all_x), min(all_y))
max_val = max(max(all_x), max(all_y))
fig.add_trace(pc.get_unity_line(min_val, max_val))

# Apply standard styling
pc.apply_standard_layout(
    fig, 
    title="My Plot Title",
    xaxis_title="X Axis Label (%BW×H)",
    yaxis_title="Y Axis Label (%BW×H)"
)

# Add statistics if needed
stats_text = f"MAE = {mae:.2f}%BW×H<br>R² = {r2:.3f}"
pc.add_stats_annotation(fig, stats_text)

# Save
fig.write_html('my_plot.html')
fig.write_image('my_plot.svg', format='svg')
"""