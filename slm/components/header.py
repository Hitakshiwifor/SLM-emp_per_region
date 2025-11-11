from dash import html, dcc
import dash_bootstrap_components as dbc
from ..config import APP_TITLE

NAV_ITEMS = [
    (" Introduction", "bi-house-door", "/"),
    (" Sankey", "bi-diagram-3", "/sankey"),
    (" Tables", "bi-table", "/tables"),
    (" Map", "bi-geo-alt", "/map"),
]

def header(pathname="/"):
    def link(lbl, icon, href):
        active = (href == pathname) or (href == "/" and pathname in ["/","",None])
        cls = "nav-pill" + (" active" if active else "")
        return dcc.Link(html.Span([html.I(className=f"bi {icon}"), lbl]), href=href, className=cls)

    return html.Div(
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.Img(src="/assets/SLM Dark.png",
                                         style={"height":"38px","marginRight":"14px"}),
                                html.Span(APP_TITLE, className="fw-semibold",
                                          style={"fontSize":"22px","color":"#003662","whiteSpace":"nowrap"}),
                            ],
                            className="d-flex align-items-center"
                        ),
                        md="auto",
                    ),
                    dbc.Col(
                        html.Div([link(l,i,h) for (l,i,h) in NAV_ITEMS],
                                 className="d-flex justify-content-end align-items-center flex-wrap")
                    ),
                ],
                className="align-items-center g-2"
            ),
            fluid=True,
            style={"paddingTop":"10px","paddingBottom":"10px"},
        ),
        className="glass",
        style={"position":"sticky","top":"8px","zIndex":"30","borderRadius":"16px","marginBottom":"10px"},
    )
