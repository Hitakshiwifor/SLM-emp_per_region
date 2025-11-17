from dash import html, dcc
import dash_bootstrap_components as dbc
from ..config import APP_TITLE  # make sure this import exists

def _hero_card():
    return dbc.Card(
        dbc.CardBody(
            [
                html.H2(
                    "Welcome to the Skilled Labor Monitor",
                    style={"color": "#003662", "marginBottom": "8px"},
                ),
                html.P(
                    "Explore where talent is today and how the workforce could evolve across Europe. "
                    "Use the map, flow visualisations and time-series views to answer strategic labour market questions.",
                    className="lead",
                    style={"fontSize": "15px", "marginBottom": "12px"},
                ),
                html.Div(
                    [
                        dbc.Badge("Europe-wide labour market", color="primary", pill=True, className="me-2 mb-2"),
                        dbc.Badge("NUTS regions", color="secondary", pill=True, className="me-2 mb-2"),
                        dbc.Badge("2025–2040 outlook", color="info", pill=True, className="me-2 mb-2"),
                    ],
                    className="mb-2",
                ),
                html.P(
                    "Start here to get a quick feeling for what the tool offers, then jump into the Map, Flows or Trends tabs.",
                    style={"fontSize": "13px", "color": "#6c757d", "marginBottom": 0},
                ),
            ]
        ),
        style={
            "borderRadius": "20px",
            "boxShadow": "0 4px 20px rgba(0,0,0,.10)",
            "backgroundColor": "#ffffff",
        },
        className="mb-3",
    )

def _persona_card(title, icon_class, lines):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.I(className=f"bi {icon_class} me-2", style={"fontSize": "20px", "color": "#003662"}),
                        html.Strong(title),
                    ],
                    className="d-flex align-items-center mb-2",
                ),
                html.Ul(
                    [html.Li(l, style={"fontSize": "13px"}) for l in lines],
                    style={"paddingLeft": "18px", "marginBottom": 0},
                ),
            ]
        ),
        style={
            "borderRadius": "16px",
            "boxShadow": "0 2px 12px rgba(0,0,0,.06)",
            "backgroundColor": "#ffffff",
            "height": "100%",
        },
    )

def _who_is_this_for_row():
    return dbc.Row(
        [
            dbc.Col(
                _persona_card(
                    "For CEOs",
                    "bi-briefcase",
                    [
                        "Identify regions with strong talent pools.",
                        "Compare countries and regions at a glance.",
                    ],
                ),
                md=4,
                className="mb-3",
            ),
            dbc.Col(
                _persona_card(
                    "For HR leaders",
                    "bi-people",
                    [
                        "Plan recruitment where supply is high.",
                        "Spot areas with latent talent and lower competition.",
                    ],
                ),
                md=4,
                className="mb-3",
            ),
            dbc.Col(
                _persona_card(
                    "For Analysts",
                    "bi-bar-chart",
                    [
                        "Drill into composition by age and sex.",
                        "Track employment trends over time per region.",
                    ],
                ),
                md=4,
                className="mb-3",
            ),
        ],
        className="g-3",
    )

def _nav_card(title, subtitle, icon_class, href):
    return dcc.Link(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.I(className=f"bi {icon_class} me-2", style={"fontSize": "22px"}),
                            html.Span(title, className="fw-semibold"),
                        ],
                        className="d-flex align-items-center mb-1",
                        style={"color": "#003662"},
                    ),
                    html.P(
                        subtitle,
                        style={"fontSize": "13px", "color": "#6c757d", "marginBottom": 0},
                    ),
                ]
            ),
            style={
                "borderRadius": "16px",
                "boxShadow": "0 2px 12px rgba(0,0,0,.06)",
                "backgroundColor": "#ffffff",
                "transition": "transform 0.15s ease, box-shadow 0.15s ease",
            },
            className="intro-nav-card h-100",
        ),
        href=href,
        style={"textDecoration": "none", "color": "inherit"},
    )

def _navigation_row():
    return dbc.Row(
        [
            dbc.Col(
                _nav_card(
                    "Map – Where is the talent?",
                    "Explore employment across Europe’s regions and see hotspots for skilled labour.",
                    "bi-geo-alt",
                    "/map",
                ),
                md=4,
                className="mb-3",
            ),
            dbc.Col(
                _nav_card(
                    "Flows – How is the workforce composed?",
                    "Use the Sankey view to see how employment splits by sex or age groups.",
                    "bi-diagram-3",
                    "/sankey",
                ),
                md=4,
                className="mb-3",
            ),
            dbc.Col(
                _nav_card(
                    "Trends – How are regions evolving?",
                    "Compare regions with time-series sparklines and current vs. future employment levels.",
                    "bi-table",
                    "/tables",
                ),
                md=4,
                className="mb-3",
            ),
        ],
        className="g-3",
    )

def _how_to_use_card():
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("How to use this tool in 3 simple steps", style={"color": "#003662"}, className="mb-2"),
                html.Ol(
                    [
                        html.Li("Pick a country or region you care about."),
                        html.Li("Use the Map to see where talent is concentrated."),
                        html.Li("Open Flows or Trends to explain the ‘why’ behind the numbers."),
                    ],
                    style={"fontSize": "13px", "marginBottom": 0},
                ),
            ]
        ),
        style={
            "borderRadius": "16px",
            "boxShadow": "0 2px 12px rgba(0,0,0,.06)",
            "backgroundColor": "#ffffff",
        },
        className="mb-3",
    )

def layout():
    return dbc.Container(
        [
            _hero_card(),
            _who_is_this_for_row(),
            _navigation_row(),
            _how_to_use_card(),
            html.Div(style={"height": "12px"}),
        ],
        fluid=True,
    )
