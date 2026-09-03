def score_color(s):
    if s >= 4: return "score-high"
    if s >= 3: return "score-mid"
    return "score-low"

def status_badge(status):
    return f'<span class="badge badge-{status}">{status}</span>'

def metric_card(label, value, color="gray"):
    return f"""
    <div class="metric-card">
        <div class="m-label">{label}</div>
        <div class="m-value {color}">{value}</div>
    </div>
    """