from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from .config import APP_TITLE, THEME, PORT
from .pages import intro, sankey, tables, map as map_page

app = Dash(__name__, external_stylesheets=[THEME], suppress_callback_exceptions=True, title=APP_TITLE)
server = app.server

# bring Bootstrap Icons + rely on assets/styles.css for the rest
app.index_string = """
<!DOCTYPE html>
<html>
<head>
  {%metas%}
  <title>{%title%}</title>
  {%favicon%}
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css">
  {%css%}
</head>
<body>
  {%app_entry%}
  <footer>
    {%config%}
    {%scripts%}
    {%renderer%}
  </footer>
</body>
</html>
"""

NAV_ITEMS = [
    (" Overview", "bi-house-door", "/"),
    (" Workforce Composition", "bi-diagram-3", "/sankey"),
    (" Regional trends", "bi-table", "/tables"),
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
                                html.Span(
                                    APP_TITLE,
                                    className="fw-semibold",
                                    style={"fontSize":"26px","color":"#003662","whiteSpace":"nowrap"},
                                ),
                            ],
                            className="d-flex align-items-center"
                        ),
                        md="auto"
                    ),
                    dbc.Col(
                        html.Div([link(l,i,h) for (l,i,h) in NAV_ITEMS],
                                 className="d-flex justify-content-end align-items-center flex-wrap"),
                        align="center"
                    ),
                ],
                className="align-items-center g-2"
            ),
            fluid=True,
            className= "header-bar",
            style={"paddingTop":"10px","paddingBottom":"10px"}
        ),
        className="glass",
        style={"position":"sticky","top":"8px","zIndex":"30","borderRadius":"16px","marginBottom":"10px"}
    )

# master layout
app.layout = html.Div([
    dcc.Location(id="url"),
    html.Div(id="header-slot"),
    html.Div(id="page-slot"),
])

# router
@app.callback(
    Output("header-slot","children"),
    Output("page-slot","children"),
    Input("url","pathname"),
)
def route(pathname):
    hdr = header(pathname or "/")
    if pathname in ["/", "", "/intro", "/home"]:
        return hdr, intro.layout()
    if pathname == "/sankey":
        return hdr, sankey.layout()
    if pathname == "/tables":
        return hdr, tables.layout()
    if pathname == "/map":
        return hdr, dbc.Container(map_page.layout(), fluid=True)
    return hdr, dbc.Container(html.H4("404 — Page not found", className="mt-4"), fluid=True)

# register callbacks for the map page
map_page.register_callbacks(app)

sankey.register_callbacks(app)

tables.register_callbacks(app)


# ---------- Run server 
if __name__ == "__main__":
    app.run(debug=False, port=PORT)
